import argparse
import json
import logging
from pathlib import Path

from soccer_clips.config import get_settings
from soccer_clips.scan import scan
from soccer_clips.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_scan(store: Store, args: argparse.Namespace) -> None:
    added = scan(store.settings, store)
    print(f"registered {len(added)} new game(s)")
    for g in added:
        print(f"  {g.id}")


def cmd_list(store: Store, args: argparse.Namespace) -> None:
    for g in store.list_games():
        played = g.played_on.isoformat() if g.played_on else "????-??-??"
        print(f"{played}  {g.duration_s / 60:5.1f} min  {g.width}x{g.height}  {g.id}")


def cmd_serve(store: Store, args: argparse.Namespace) -> None:
    s = store.settings
    host, port = args.host or s.host, args.port or s.port
    if not args.no_reload:
        return _serve_reloading(host, port)
    import uvicorn

    from soccer_clips.server.app import create_app

    uvicorn.run(create_app(store), host=host, port=port)


def _serve_reloading(host: str, port: int) -> None:
    """
    Run the server in a child process and restart it when the code changes, so fixes made in
    the tagger always use the current ball filter. Exports and reel renders live in the
    server's memory, so a restart waits until none is queued or running.
    """
    import subprocess
    import sys
    import urllib.request
    from pathlib import Path

    from watchfiles import watch

    log = logging.getLogger("serve")
    src = Path(__file__).parent
    cmd = [sys.executable, "-m", "soccer_clips.cli", "serve", "--no-reload", "--host", host, "--port", str(port)]
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/api/jobs"

    def busy() -> bool:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return any(j["status"] in ("queued", "running") for j in json.load(r))
        except Exception:
            return False  # not up (starting, or crashed): nothing to lose

    def start() -> subprocess.Popen:
        return subprocess.Popen(cmd)

    def stop(child: subprocess.Popen) -> None:
        child.terminate()
        try:
            child.wait(15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    child, pending, said = start(), False, False
    log.info("watching %s for code changes (restarts wait for exports/renders to finish; --no-reload to turn off)", src)
    try:
        for changes in watch(src, watch_filter=lambda _change, path: path.endswith(".py"),
                             rust_timeout=5000, yield_on_timeout=True):
            if changes:
                pending = True
            if not pending:
                continue
            if busy():
                if not said:
                    log.info("code changed: restarting when the running exports/renders finish")
                    said = True
                continue
            log.info("code changed: restarting the server")
            stop(child)
            child, pending, said = start(), False, False
    finally:
        stop(child)


def cmd_bench(store: Store, args: argparse.Namespace) -> None:
    from soccer_clips.track import bench

    if args.bench_command == "add":
        c = bench.add_clip(store, args.game_id, args.clip_id, args.name, args.why or "")
        print(f"added {c.name}: {c.game_id} {c.start_s:.1f}-{c.end_s:.1f}s")
    elif args.bench_command == "cut":
        names = bench.cut(store, force=args.force)
        print(f"cut {len(names)} clip(s) into {bench.bench_dir(store.settings) / 'clips'}")
    elif args.bench_command == "run":
        from soccer_clips.track import benchmark

        benchmark.run(store, args.variants or list(benchmark.VARIANTS))
    elif args.bench_command == "draw":
        from soccer_clips.track import benchmark

        names = args.clips or [c.name for c in bench.load_set(store.settings).clips]
        for name in names:
            print(benchmark.draw(store, args.variant, name, ball_conf=args.ball_conf, results=args.results))
    else:
        for c in bench.load_set(store.settings).clips:
            who = f"  callout #{c.callout_player}" if c.callout_player is not None else ("  callout" if c.callout else "")
            print(f"{c.name:24} {c.end_s - c.start_s:5.1f}s  {c.game_id}{who}  {c.why}")


def cmd_ball(store: Store, args: argparse.Namespace) -> None:
    from soccer_clips.track import ball_labels

    if args.ball_command == "sample":
        new = (ball_labels.sample_bench(store, args.n or 150) if args.set == "bench"
               else ball_labels.sample_train(store, args.n or 200, seed=args.seed))
        added = ball_labels.add_tasks(store, args.set, new)
        print(f"added {added} frame(s) to the {args.set} set")
        if args.set != "bench" and added:
            print(f"run `soccer-clips ball guess {args.set}` to pre-fill the model's guesses")
    elif args.ball_command == "guess":
        n = ball_labels.make_guesses(store, args.set)
        print(f"guessed {n} frame(s)")
    elif args.ball_command == "others":
        from collections import Counter

        from soccer_clips.track import ball_others

        if args.others_command == "find":
            n = ball_others.find(store, model_name=args.model, imgsz=args.imgsz, log=lambda m: print(m, flush=True))
            print(f"{n} new candidate(s) to review at /review-balls")
        else:
            entries = [b for bs in ball_others.load(store.settings).frames.values() for b in bs]
            c = Counter("manual" if b.source == "manual" else (b.verdict or "to review") for b in entries)
            print(", ".join(f"{v} {k}" for k, v in c.items()) or "no other balls yet")
    elif args.ball_command == "track":
        from soccer_clips.track import ball_track
        from soccer_clips.track.bench import bench_dir

        results = bench_dir(store.settings) / args.results
        print("variant          method   found  wrong  missed  (filled)  hidden-near  claimed-when-none")
        for v in args.variants:
            for method in ("top1", "track"):
                s = ball_track.evaluate(store, results, v, method=method)
                f = lambda k: "   –" if s[k] is None else f"{s[k]:4.0%}"
                print(f"{v:16} {method:6}  {f('found')}   {f('wrong')}   {f('missed')}     {f('filled_share_of_found')}"
                      f"        {f('hidden_near')}         {f('claimed_when_none')}", flush=True)
            if args.draw:
                from soccer_clips.track.bench import load_set

                for c in load_set(store.settings).clips:
                    print(ball_track.draw(store, results, v, c.name))
    elif args.ball_command == "fixed-objects":
        from soccer_clips.track import ball_game, world

        ids = [g.id for g in store.list_games()] if args.game == "all" else [args.game]
        for gid in ids:
            game = store.load_game(gid)
            if not (ball_game.tracks_dir(store, game) / "ball_raw.npz").exists():
                print(f"{gid}: no ball detections yet, skipped")
                continue
            n = world.find_fixed(store, game)
            ball_game.retrack(store, game)
            print(f"{gid}: {n} detections on fixed objects blocked; path redone", flush=True)
    elif args.ball_command == "eval-games":
        from soccer_clips.track import ball_game

        per = ball_game.evaluate(store, fixes=args.fixes)
        tot = {k: sum(n[k] for n in per.values()) for k in next(iter(per.values()), {})}
        print("game                                        found  wrong  missed  claimed-when-none  (visible, none)")
        for gid, n in list(per.items()) + [("all", tot)]:
            v, z = max(1, n["visible"]), max(1, n["none"])
            print(f"{gid:42}  {n['found'] / v:5.0%}  {n['wrong'] / v:5.0%}  {n['missed'] / v:6.0%}  "
                  f"{n['claimed'] / z:17.0%}  ({n['visible']}, {n['none']})")
    elif args.ball_command == "track-game":
        from soccer_clips.track import ball_game

        ids = [g.id for g in store.list_games()] if args.game == "all" else [args.game]
        for gid in ids:
            if args.retrack:
                n = ball_game.retrack(store, store.load_game(gid))
            else:
                ball_game.track_game(store, gid, mode=args.mode, model_name=args.model, workers=args.workers,
                                     log=lambda m: print(m, flush=True))
                n = len((ball_game.load_path(store, store.load_game(gid)) or {"frame": []})["frame"])
            print(f"{gid}: ball in {n} frames", flush=True)
    elif args.ball_command == "export":
        from soccer_clips.track import ball_train

        ball_train.export(store)
    elif args.ball_command == "train":
        from soccer_clips.track import ball_train

        for base in args.bases:
            ball_train.train(store, base, epochs=args.epochs, batch=args.batch)
    elif args.ball_command == "eval":
        from pathlib import Path

        from soccer_clips.track import ball_train
        from soccer_clips.track.benchmark import models_dir

        scores = {}
        frames = [tuple(x) for x in json.loads(Path(args.frames_file).read_text())] if args.frames_file else None
        for name in args.models:
            weights = Path(name) if Path(name).exists() else models_dir(store) / f"{name}.pt"
            label = name + ("" if args.full else "-centre")
            scores[label] = ball_train.evaluate(store, weights, crop=not args.full, imgsz=args.imgsz, frames=frames)
            print(ball_labels.score_table({label: scores[label]}).split("\n", 1)[1], flush=True)
            if args.json:
                Path(args.json).write_text(json.dumps(scores, indent=2))
    elif args.ball_command == "score":
        from soccer_clips.track.bench import bench_dir

        scores = ball_labels.score(store, bench_dir(store.settings) / args.results)
        print(ball_labels.score_table(scores))
    else:
        for s in ball_labels.summary(store):
            print(f"{s['name']:8} {s['labelled']}/{s['total']} labelled")


def cmd_people(store: Store, args: argparse.Namespace) -> None:
    from soccer_clips.track import people

    def ids(game: str) -> list[str]:
        return [g.id for g in store.list_games()] if game == "all" else [game]

    log = lambda m: print(m, flush=True)  # noqa: E731
    if args.people_command == "detect":
        for gid in ids(args.game):
            n = people.detect_game(store, gid, mode=args.mode, model_name=args.model, workers=args.workers, log=log)
            print(f"{gid}: {args.model} detected {n} new frame(s)", flush=True)
        return
    if args.people_command == "kit":
        from soccer_clips.track import kit

        for gid in ids(args.game):
            game = store.load_game(gid)
            if not people.raw_path(store, game, args.model).exists():
                print(f"{gid}: no {args.model} boxes yet, skipped")
                continue
            n = kit.compute(store, game, args.model, workers=args.workers, log=log)
            print(f"{gid}: kit colours for {n} boxes", flush=True)
        return
    if args.people_command == "teams":
        for gid in ids(args.game):
            game = store.load_game(gid)
            n = people.recluster(store, game, args.model)
            roles = people.team_roles(people.tracks(store, game, args.model), people.load_picks(store, game), game,
                                      store.settings.default_player)
            print(f"{gid}: {n} kit clusters: " + ", ".join(f"{k} {v}" for k, v in sorted(roles.items())), flush=True)
        return
    p = people.Params(tracker=args.tracker, track_buffer=args.buffer, kit_gate=args.kit_gate, split_kit=args.split_kit,
                      link_gap_s=args.link_gap, link_kit=args.link_kit, link_heights=args.link_heights)
    if args.people_command == "eval-labels":
        r = people.evaluate_labels(store, args.model, p, player=args.player, saved=args.saved, games=args.games, log=log)
        print("game                                      points  found  hold median/mean s  conflicts  swaps left")
        for gid, g in r["games"].items():
            hs = sorted(g["holds"])
            med = hs[len(hs) // 2] if hs else 0
            print(f"{gid:42} {g['points']:5}  {g['found']:5}  {med:8.1f} / {sum(hs) / max(1, len(hs)):5.1f}"
                  f"  {g['conflicts']:9}  {g['swaps_left']:3}/{g['wrong']}")
        print(f"{'all':42} {r['points']:5}  {r['found']:5}  {r['hold_median_s']:8} / {r['hold_mean_s']:5}"
              f"  {r['conflicts']:9}  {r['swaps_left']:3}/{r['wrong']}")
        return
    if args.people_command == "track":
        todo = []
        for gid in ids(args.game):
            if people.raw_path(store, store.load_game(gid), args.model).exists():
                todo.append(gid)
            else:
                print(f"{gid}: no {args.model} boxes yet, skipped")
        for gid, n, k in people.track_games(store, todo, args.model, p, workers=args.workers):
            print(f"{gid}: {n} boxes in {k} tracklets", flush=True)
    elif args.people_command == "eval":
        print("model      tracker    callouts  found  coverage (median / mean)  run s  tracks/min  people/frame")
        out = {}
        for model in args.models:
            r = people.evaluate(store, model, p, players={args.player} if args.player else None, saved=args.saved,
                                log=log)
            out[model] = r
            s = r["summary"]
            if not s["callouts"]:
                print(f"{model:10} no callouts inside detected ranges")
                continue
            print(f"{model:10} {'saved' if args.saved else args.tracker:10} {s['callouts']:8}  {s['found']:5.0%}  {s['coverage_median']:8.0%} / "
                  f"{s['coverage_mean']:4.0%}         {s['run_s_median']:5}  {s['tracks_per_min']:10}  "
                  f"{s['people_per_frame']:12}")
        if args.json:
            from pathlib import Path

            Path(args.json).write_text(json.dumps(out, indent=2))


def cmd_gametime(store: Store, args: argparse.Namespace) -> None:
    from soccer_clips import gametime

    def fmt(t):
        return "-" if t is None else f"{int(t // 60)}:{int(t % 60):02d}"

    log = lambda m: print(m, flush=True)  # noqa: E731
    ids = [g.id for g in store.list_games()] if args.game == "all" else [args.game]
    for gid in ids:
        game = store.load_game(gid)
        if args.gametime_command == "suggest":
            sg = gametime.suggest(store, game)
            print(f"{gid}: 1st half {fmt(sg.start_1h_s)}-{fmt(sg.end_1h_s)}, 2nd half {fmt(sg.start_2h_s)}-"
                  f"{fmt(sg.end_2h_s)}" + (f"  ({'; '.join(sg.notes)})" if sg.notes else ""))
            if args.apply:
                with store.edit_game(gid) as g:
                    for key, val in (("kickoff_1h_s", sg.start_1h_s), ("end_1h_s", sg.end_1h_s),
                                     ("kickoff_2h_s", sg.start_2h_s), ("end_2h_s", sg.end_2h_s)):
                        if val is not None:
                            setattr(g, key, val)
        elif args.gametime_command == "cut":
            if game.cut:
                print(f"{gid}: already cut")
                continue
            t = gametime.periods(game)
            if not args.yes:
                raise SystemExit(f"{gid}: would keep 1st half {fmt(t[0][0])}-{fmt(t[0][1])} and 2nd half "
                                 f"{fmt(t[1][0])}-{fmt(t[1][1])}; add --yes to cut")
            print(f"{gid}: {gametime.cut_game(store, gid, log=log)}")
        elif args.gametime_command == "restore":
            gametime.restore(store, gid, log=log)


def cmd_reid(store: Store, args: argparse.Namespace) -> None:
    from soccer_clips.track import people, reid

    log = lambda m: print(m, flush=True)  # noqa: E731
    if args.reid_command == "crops":
        for gid in args.games or [g.id for g in store.list_games()]:
            game = store.load_game(gid)
            if people.tracks(store, game) is None:
                print(f"{gid}: no player tracks yet, skipped")
                continue
            print(f"{gid}: {reid.extract_crops(store, game, workers=args.workers, log=log)} crops", flush=True)
    elif args.reid_command == "sigs":
        for gid in args.games or [g.id for g in store.list_games()]:
            game = store.load_game(gid)
            if not reid.crops_path(store, game).exists():
                print(f"{gid}: no crops yet, skipped")
                continue
            print(f"{gid}: {reid.compute_signatures(store, game, device=args.device, log=log)} signatures", flush=True)
    elif args.reid_command == "auto-eval":
        ids = args.games or [g.id for g in store.list_games() if reid.sigs_path(store, g).exists()]
        r = reid.evaluate_auto(store, ids, log=log)
        print("group   min  margin | right wrong missed | precision coverage | claimed min | on others' labels s")
        for grp, per in r.items():
            for (mn, mg), x in per.items():
                print(f"{grp:7} {mn:4} {mg:6} | {x['right']:5} {x['wrong']:5} {x['missed']:6} | {x['precision']!s:9}"
                      f" {x['coverage']:8} | {x['claimed_s'] / 60:11.1f} | {x['false_s']:.0f}")
    elif args.reid_command == "experiment":
        r = reid.run_experiment(Path(args.data), Path(args.stock), Path(args.out), n_folds=args.folds,
                                steps=args.steps, device=args.device, log=log)
        print(json.dumps(r["summary"], indent=1))


def cmd_gpu(store: Store, args: argparse.Namespace) -> None:
    from soccer_clips import gpu

    if args.gpu_command == "run":
        import shutil

        from soccer_clips.track import ball_labels
        from soccer_clips.track.benchmark import models_dir

        if args.job == "ball-games":
            if args.mode == "callouts":
                raise SystemExit("ball-games takes --mode clips or all")
            ids = args.games or [g.id for g in store.list_games()]
            job = gpu.ball_games_job(store, ids, mode=args.mode, log=lambda m: print(m, flush=True))
        elif args.job == "reid-train":
            job = gpu.reid_train_job(store, args.games, steps=args.steps)
        elif args.job == "reid-sigs":
            job = gpu.reid_sigs_job(store, args.games)
        elif args.job == "people-games":
            ids = args.games or [g.id for g in store.list_games()]
            job = gpu.people_games_job(store, ids, mode=args.mode, models=args.models, workers=args.workers)
        else:
            job = gpu.ball_train_job(store, stock=args.stock, train_bases=args.train, epochs=args.epochs,
                                     batch=args.batch)
        runner = gpu.Runner(store, job, gpu_types=args.gpu or gpu.GPU_TYPES, max_hours=args.max_hours,
                            log=lambda m: print(m, flush=True), countries=args.countries,
                            use_volume=not args.no_volume, fallback=not args.no_fallback)
        print(f"run folder: {runner.dir}", flush=True)
        if args.job == "reid-train":  # what it learns from, for the roster page's run history
            shutil.copy(gpu.data_root(store) / "reid" / "index.json", runner.dir / "bundle_index.json")
        try:
            results = runner.run()
        finally:  # install whatever came back, even from a run that failed part-way
            if args.job == "ball-games" and (runner.dir / "results").exists():
                gpu.install_ball_tracks(store, runner.dir / "results", log=print)
            if args.job == "people-games" and (runner.dir / "results").exists():
                gpu.install_people_raw(store, runner.dir / "results", log=print)
            if args.job == "reid-sigs" and (runner.dir / "results").exists():
                gpu.install_tracks_files(store, runner.dir / "results", "reid_sigs.npz", log=print)
        for w in sorted((results / "models").glob("ball_*.pt")):
            shutil.copy(w, models_dir(store) / w.name)
            print(f"model: {models_dir(store) / w.name}")
        if args.job == "reid-train":
            from soccer_clips.track import reid

            if (results / "out" / reid.TUNED).exists():
                shutil.copy(results / "out" / reid.TUNED, models_dir(store) / reid.TUNED)
                print(f"model: {models_dir(store) / reid.TUNED}")
            f = results / "out" / "reid_results.json"
            if f.exists():
                print(json.dumps(json.loads(f.read_text())["summary"], indent=1))
        for name in ("stock", "finetuned"):
            f = results / "out" / f"{name}.json"
            if f.exists():
                print(f"\n{name}:\n" + ball_labels.score_table(json.loads(f.read_text())))
    elif args.gpu_command == "volume":
        vol = gpu.load_volume(store)
        if args.volume_command == "create":
            vol = gpu.create_volume(store, args.dc, args.size)
            print(f"created volume {vol['id']}: {vol['size']} GB in {vol['dataCenterId']}")
        elif args.volume_command == "fill":
            files = gpu.s3_sync(store, gpu.volume_fill_job(store).volume_files, log=lambda m: print(m, flush=True))
            print(f"volume has all {len(files)} file(s)")
        elif args.volume_command == "delete":
            if not args.yes:
                raise SystemExit("this deletes the volume and everything on it; add --yes to go ahead")
            vol = gpu.delete_volume(store)
            print(f"deleted volume {vol['id']} ({vol['dataCenterId']})")
        elif vol:
            live = next((v for v in gpu.api("GET", "/networkvolumes") or [] if v.get("id") == vol["id"]), None)
            print(f"volume {vol['id']}: {vol['size']} GB in {vol['dataCenterId']}"
                  + ("" if live else "  (NOT FOUND on RunPod)"))
        else:
            print("no volume set up (soccer-clips gpu volume create)")
    elif args.gpu_command == "retrain-reid":
        from soccer_clips import retrain

        retrain.run(store, log=lambda m: print(m, flush=True), log_name=args.log_name)
    elif args.gpu_command in ("retrain-ball", "retrack-balls"):
        from soccer_clips import ball_retrain

        step = ball_retrain.run_train if args.gpu_command == "retrain-ball" else ball_retrain.run_retrack
        step(store, log=lambda m: print(m, flush=True), log_name=args.log_name)
    elif args.gpu_command == "sweep":
        from soccer_clips import retrain

        print(f"{len(retrain.sweep(store))} overdue pod(s) deleted")
    elif args.gpu_command == "cleanup":
        pods = gpu.list_pods()
        for p in pods:
            gpu.delete_pod(p["id"])
            print(f"deleted {p['id']} ({p.get('name')})")
        print(f"{len(pods)} pod(s) deleted")
    else:
        pods = gpu.list_pods()
        for p in pods:
            print(f"{p['id']}  {p.get('name')}  {p.get('desiredStatus')}  ${p.get('costPerHr')}/h")
        print(f"{len(pods)} soccer-clips pod(s) running or stopped")


def people_defaults():
    from soccer_clips.track.people import Params

    return Params()


def main() -> None:
    parser = argparse.ArgumentParser(prog="soccer-clips")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="register new videos found in SOCCER_FULL_DIR/<year>/")
    sub.add_parser("list", help="list registered games")
    serve = sub.add_parser("serve", help="run the web server (scans on startup)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--no-reload", action="store_true", help="don't restart when the code changes")
    bench = sub.add_parser("bench", help="tracking benchmark clips (list them with no subcommand)")
    bench_sub = bench.add_subparsers(dest="bench_command")
    add = bench_sub.add_parser("add", help="add a tagged clip to the set (same name replaces it)")
    add.add_argument("game_id")
    add.add_argument("clip_id")
    add.add_argument("name", help="file name, e.g. fc1-corner")
    add.add_argument("--why", help="what it tests (default: the clip's note)")
    cut = bench_sub.add_parser("cut", help="cut the set's mp4s (only new or changed ones)")
    cut.add_argument("--force", action="store_true", help="recut everything")
    run = bench_sub.add_parser("run", help="run detection + tracking variants on the set")
    run.add_argument("variants", nargs="*", help="default: all (n640 s640 n1280 s1280 s1280-centre s1920 m1280-centre m1920 ft26x-centre)")
    draw = bench_sub.add_parser("draw", help="draw a variant's boxes onto the clips for checking by eye")
    draw.add_argument("variant")
    draw.add_argument("clips", nargs="*", help="default: all")
    draw.add_argument("--ball-conf", type=float, default=0.1, help="weakest ball detection to draw")
    draw.add_argument("--results", default="results", help="results folder in bench/ (e.g. results-l4)")
    ball = sub.add_parser("ball", help="ball labels: progress with no subcommand")
    ball_sub = ball.add_subparsers(dest="ball_command")
    sample = ball_sub.add_parser("sample", help="add frames to label (bench: spread over the bench clips)")
    sample.add_argument("set", choices=["bench", "train"])
    sample.add_argument("-n", type=int, help="how many (default 150 bench, 200 train)")
    sample.add_argument("--seed", type=int)
    guess = ball_sub.add_parser("guess", help="pre-compute the model's ball guesses for a set (not bench)")
    guess.add_argument("set", choices=["train"])
    others = ball_sub.add_parser("others", help="other real balls (spares etc.): counts with no subcommand")
    others_sub = others.add_subparsers(dest="others_command")
    find = others_sub.add_parser("find", help="find candidate other balls in labelled frames with a model")
    find.add_argument("--model", default="ball_yolo26x", help="name in models/ (default ball_yolo26x)")
    find.add_argument("--imgsz", type=int, default=1920, help="1920 = whole frame at native resolution")
    trk = ball_sub.add_parser("track", help="game-ball path through the bench clips, scored against bench labels")
    trk.add_argument("variants", nargs="+", help="benchmark variants with per-frame detections, e.g. m1280-centre")
    trk.add_argument("--results", default="results", help="results folder in bench/ (e.g. results-l4)")
    trk.add_argument("--draw", action="store_true", help="also write review videos (<clip>.track.mp4)")
    tg = ball_sub.add_parser("track-game", help="game-ball path for a game's tagged clips (or whole game)")
    tg.add_argument("game", help="game id, or 'all'")
    tg.add_argument("--mode", choices=["clips", "all"], default="clips", help="tagged clips (+2s) or the whole game")
    tg.add_argument("--model", default="ball_yolo26x", help="ball model in models/")
    tg.add_argument("--retrack", action="store_true", help="redo the path from saved detections (no detecting)")
    tg.add_argument("--workers", type=int, default=1, help="detect chunks of long ranges in parallel processes")
    fo = ball_sub.add_parser("fixed-objects", help="find things that never move (flags, chairs, bags) and block "
                             "their detections from being the ball (~10 min a game, CPU), then redo the path")
    fo.add_argument("game", help="game id, or 'all'")
    ev = ball_sub.add_parser("eval-games", help="score whole-game ball paths at the labelled frames in the games")
    ev.add_argument("--fixes", action="store_true", help="with hand fixes (circular where fixed frames are labels)")
    ball_sub.add_parser("export", help="turn train-set labels into a YOLO dataset (labels/dataset/)")
    train = ball_sub.add_parser("train", help="fine-tune YOLO bases on the exported dataset (GPU)")
    train.add_argument("bases", nargs="+", help="e.g. yolo11m yolo26l; weights go to models/ball_<base>.pt")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch", type=int, default=8)
    ev = ball_sub.add_parser("eval", help="score models on the bench frame images")
    ev.add_argument("models", nargs="+", help="names in models/ (yolo11m, ball_yolo11m, ...) or .pt paths")
    ev.add_argument("--full", action="store_true", help="whole frame instead of the centre crop")
    ev.add_argument("--imgsz", type=int, default=1280)
    ev.add_argument("--json", help="also write the scores to this file")
    ev.add_argument("--frames-file", help="score on these labelled frames instead: JSON [[game_id, frame], ...]")
    score = ball_sub.add_parser("score", help="score benchmark ball detections against the bench labels")
    score.add_argument("--results", default="results", help="results folder in bench/ (e.g. results-l4)")
    pp = sub.add_parser("people", help="person boxes and tracklets per game (Phase 3.2)")
    pp_sub = pp.add_subparsers(dest="people_command", required=True)
    det = pp_sub.add_parser("detect", help="detect people in a game's ranges (GPU; resumable)")
    det.add_argument("game", help="game id, or 'all'")
    det.add_argument("--mode", choices=["callouts", "clips", "all"], default="callouts",
                     help="clips with a callout, every tagged clip, or the whole game (each +2s either side)")
    det.add_argument("--model", default="yolo11m", help="stock detector, e.g. yolo11m, yolo26l")
    det.add_argument("--workers", type=int, default=1, help="detect chunks of long ranges in parallel processes")
    kt = pp_sub.add_parser("kit", help="kit colour of every saved box, for telling teams apart (CPU, ~20 min a game)")
    kt.add_argument("game", help="game id, or 'all'")
    kt.add_argument("--model", default="yolo11m", help="whose saved boxes")
    kt.add_argument("--workers", type=int, default=4, help="processes (one per CPU core)")
    tm = pp_sub.add_parser("teams", help="redo the kit clusters of saved tracklets (seconds; no tracking)")
    tm.add_argument("game", help="game id, or 'all'")
    tm.add_argument("--model", default="yolo11m", help="whose saved tracklets")
    for name, help_ in [("eval-labels", "score tracklets against your player labels (L clicks) and callouts"),
                        ("track", "turn saved boxes into tracklets (CPU), saved as tracks/people_<model>.npz"),
                        ("eval", "score tracklets at the callouts (how much of each clip her track covers)")]:
        sp = pp_sub.add_parser(name, help=help_)
        if name == "track":
            sp.add_argument("game", help="game id, or 'all'")
            sp.add_argument("--model", default="yolo11m", help="whose saved boxes")
            sp.add_argument("--workers", type=int, default=1, help="games tracked at once (one per CPU core)")
        elif name == "eval-labels":
            sp.add_argument("--model", default="yolo11m", help="whose saved boxes")
            sp.add_argument("--player", type=int, default=38)
            sp.add_argument("--saved", action="store_true", help="score the saved tracklets instead of tracking again")
            sp.add_argument("--games", nargs="+", help="only these game ids")
        else:
            sp.add_argument("models", nargs="*", default=["yolo11m"], help="whose saved boxes (default yolo11m)")
            sp.add_argument("--player", type=int, help="only callouts on clips tagged with this number")
            sp.add_argument("--json", help="also write per-callout results to this file")
            sp.add_argument("--saved", action="store_true",
                            help="score the saved tracklets (people track) instead of tracking again")
        sp.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort")
        sp.add_argument("--buffer", type=int, default=30, help="frames a lost track is kept to be found again")
        dp = people_defaults()
        sp.add_argument("--kit-gate", type=float, default=dp.kit_gate,
                        help=f"never match across kits further apart than this (0-1; 0 = off; default {dp.kit_gate})")
        sp.add_argument("--split-kit", type=float, default=dp.split_kit,
                        help=f"cut tracklets where the kit changes by more than this (0 = off; default {dp.split_kit})")
        sp.add_argument("--link-gap", type=float, default=dp.link_gap_s,
                        help=f"join tracklets across gaps up to this many seconds (0 = off; default {dp.link_gap_s})")
        sp.add_argument("--link-kit", type=float, default=dp.link_kit, help="...if their kits are at most this far apart")
        sp.add_argument("--link-heights", type=float, default=1.0, help="...and the start is this close (box heights)")
    gt = sub.add_parser("gametime", help="when each half is played; cut videos down to game time")
    gt_sub = gt.add_subparsers(dest="gametime_command", required=True)
    sg = gt_sub.add_parser("suggest", help="suggest half start/end times from the player tracks")
    sg.add_argument("game", help="game id, or 'all'")
    sg.add_argument("--apply", action="store_true", help="store the suggestions (times not found are left alone)")
    ct = gt_sub.add_parser("cut", help="cut the video to game time and move the game's data onto it (backed up)")
    ct.add_argument("game", help="game id, or 'all'")
    ct.add_argument("--yes", action="store_true", help="go ahead (without it, just says what would be kept)")
    rs = gt_sub.add_parser("restore", help="undo a cut from its backup")
    rs.add_argument("game")
    rp = sub.add_parser("reid", help="player appearance signatures (Phase 4)")
    rp_sub = rp.add_subparsers(dest="reid_command", required=True)
    rc = rp_sub.add_parser("crops", help="cut player crops out of each tracked game (for training and scoring)")
    rc.add_argument("--games", nargs="+", help="game ids (default: all)")
    rc.add_argument("--workers", type=int, default=4)
    rs = rp_sub.add_parser("sigs", help="appearance signatures of our team's crops (tuned model if there is one)")
    rs.add_argument("--games", nargs="+", help="game ids (default: all)")
    rs.add_argument("--device", default="cpu", help="cpu or cuda (about 17 crops/s on this CPU)")
    ra = rp_sub.add_parser("auto-eval", help="score automatic labels: hide half of each player's labels, auto-label, compare")
    ra.add_argument("--games", nargs="+", help="game ids (default: all with signatures)")
    rx = rp_sub.add_parser("experiment", help="cross-validated fine-tuning and scoring on a crops bundle (GPU pod)")
    rx.add_argument("--data", required=True, help="folder from reid.write_bundle")
    rx.add_argument("--stock", required=True, help="stock OSNet-AIN weights")
    rx.add_argument("--out", required=True)
    rx.add_argument("--folds", type=int, default=3)
    rx.add_argument("--steps", type=int, default=3000)
    rx.add_argument("--device", default="cuda")
    g = sub.add_parser("gpu", help="cloud GPU jobs on RunPod (lists our pods with no subcommand)")
    g_sub = g.add_subparsers(dest="gpu_command")
    run_gpu = g_sub.add_parser("run", help="start a pod, run the job, copy results back, delete the pod")
    run_gpu.add_argument("job", choices=["ball-train", "ball-games", "people-games", "reid-train", "reid-sigs"],
                         help="ball-train: fine-tune the ball model; ball-games: game-ball paths for games; "
                         "people-games: person boxes for games")
    run_gpu.add_argument("--mode", choices=["callouts", "clips", "all"], default="clips",
                         help="ball-games / people-games: tagged clips only (ball-games uploads just those "
                         "stretches) or whole games; people-games also takes callouts (clips with a callout)")
    run_gpu.add_argument("--games", nargs="+", help="ball-games / people-games: game ids (default: all games)")
    run_gpu.add_argument("--models", nargs="+", default=["yolo11m"], help="people-games: detectors, one pass each")
    run_gpu.add_argument("--workers", type=int, default=4, help="people-games: detection processes per game")
    run_gpu.add_argument("--stock", nargs="*", default=[], help="stock bases to score first (the best other one is also fine-tuned)")
    run_gpu.add_argument("--train", nargs="+", default=["yolo26x"], help="bases to fine-tune")
    run_gpu.add_argument("--epochs", type=int, default=60)
    run_gpu.add_argument("--steps", type=int, default=3000, help="reid-train: training steps per model")
    run_gpu.add_argument("--batch", type=int, default=4,
                         help="images per step; YOLO26x at 1280 runs out of memory at 8 on a 24 GB card")
    run_gpu.add_argument("--gpu", nargs="+", help="RunPod GPU type ids, in order of preference")
    run_gpu.add_argument("--countries", nargs="*", help="country codes for the pod's data centre (default: US; "
                         "give none to allow anywhere)")
    run_gpu.add_argument("--no-volume", action="store_true", help="upload everything instead of using the volume")
    run_gpu.add_argument("--no-fallback", action="store_true",
                         help="with a volume: wait for its data centre instead of running elsewhere")
    run_gpu.add_argument("--max-hours", type=float, default=4.0, help="hard limit; the pod also deletes itself after this")
    g_sub.add_parser("cleanup", help="delete every soccer-clips pod")
    rr = g_sub.add_parser("retrain-reid", help="fine-tune the recognition model, keep it if it scores as well, re-sign "
                          "every game (what the roster page's Retrain button starts)")
    rr.add_argument("--log-name", help=argparse.SUPPRESS)
    for name, about in (("retrain-ball", "fine-tune the ball model, keep it if it scores as well (the ball labels page's Retrain)"),
                        ("retrack-balls", "every game's ball track again with the ball model in use, hand fixes kept")):
        g_sub.add_parser(name, help=about).add_argument("--log-name", help=argparse.SUPPRESS)
    g_sub.add_parser("sweep", help="delete soccer-clips pods past their deadline")
    vol = g_sub.add_parser("volume", help="network volume for game videos and models (status with no subcommand)")
    vol_sub = vol.add_subparsers(dest="volume_command")
    vc = vol_sub.add_parser("create", help="create the network volume (billed monthly until deleted)")
    vc.add_argument("--dc", default="US-KS-2", help="data centre (default US-KS-2)")
    vc.add_argument("--size", type=int, default=50, help="GB (can grow later, never shrink)")
    vol_sub.add_parser("fill", help="upload game videos and the ball model to the volume (S3 API, no pod)")
    vd = vol_sub.add_parser("delete", help="delete the volume and everything on it")
    vd.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    store = Store(get_settings())
    {"scan": cmd_scan, "list": cmd_list, "serve": cmd_serve, "bench": cmd_bench, "ball": cmd_ball, "people": cmd_people, "gametime": cmd_gametime,
     "reid": cmd_reid, "gpu": cmd_gpu}[args.command](store, args)


if __name__ == "__main__":
    main()
