import argparse
import json
import logging

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
    import uvicorn

    from soccer_clips.server.app import create_app

    s = store.settings
    uvicorn.run(create_app(store), host=args.host or s.host, port=args.port or s.port)


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
        for name in args.models:
            weights = Path(name) if Path(name).exists() else models_dir(store) / f"{name}.pt"
            label = name + ("" if args.full else "-centre")
            scores[label] = ball_train.evaluate(store, weights, crop=not args.full, imgsz=args.imgsz)
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


def cmd_gpu(store: Store, args: argparse.Namespace) -> None:
    from soccer_clips import gpu

    if args.gpu_command == "run":
        import shutil

        from soccer_clips.track import ball_labels
        from soccer_clips.track.benchmark import models_dir

        if args.job == "ball-games":
            ids = args.games or [g.id for g in store.list_games()]
            job = gpu.ball_games_job(store, ids, mode=args.mode, log=lambda m: print(m, flush=True))
        else:
            job = gpu.ball_train_job(store, stock=args.stock, train_bases=args.train, epochs=args.epochs,
                                     batch=args.batch)
        runner = gpu.Runner(store, job, gpu_types=args.gpu or gpu.GPU_TYPES, max_hours=args.max_hours,
                            log=lambda m: print(m, flush=True), countries=args.countries,
                            use_volume=not args.no_volume, fallback=not args.no_fallback)
        print(f"run folder: {runner.dir}", flush=True)
        try:
            results = runner.run()
        finally:  # install whatever came back, even from a run that failed part-way
            if args.job == "ball-games" and (runner.dir / "results").exists():
                gpu.install_ball_tracks(store, runner.dir / "results", log=print)
        for w in sorted((results / "models").glob("ball_*.pt")):
            shutil.copy(w, models_dir(store) / w.name)
            print(f"model: {models_dir(store) / w.name}")
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="soccer-clips")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="register new videos found in SOCCER_FULL_DIR/<year>/")
    sub.add_parser("list", help="list registered games")
    serve = sub.add_parser("serve", help="run the web server (scans on startup)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
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
    score = ball_sub.add_parser("score", help="score benchmark ball detections against the bench labels")
    score.add_argument("--results", default="results", help="results folder in bench/ (e.g. results-l4)")
    g = sub.add_parser("gpu", help="cloud GPU jobs on RunPod (lists our pods with no subcommand)")
    g_sub = g.add_subparsers(dest="gpu_command")
    run_gpu = g_sub.add_parser("run", help="start a pod, run the job, copy results back, delete the pod")
    run_gpu.add_argument("job", choices=["ball-train", "ball-games"],
                         help="ball-train: fine-tune the ball model; ball-games: game-ball paths for games")
    run_gpu.add_argument("--mode", choices=["clips", "all"], default="clips",
                         help="ball-games: tagged clips only (uploads just those stretches) or whole games")
    run_gpu.add_argument("--games", nargs="+", help="ball-games: game ids (default: all games)")
    run_gpu.add_argument("--stock", nargs="*", default=[], help="stock bases to score first (the best other one is also fine-tuned)")
    run_gpu.add_argument("--train", nargs="+", default=["yolo26x"], help="bases to fine-tune")
    run_gpu.add_argument("--epochs", type=int, default=60)
    run_gpu.add_argument("--batch", type=int, default=4,
                         help="images per step; YOLO26x at 1280 runs out of memory at 8 on a 24 GB card")
    run_gpu.add_argument("--gpu", nargs="+", help="RunPod GPU type ids, in order of preference")
    run_gpu.add_argument("--countries", nargs="*", help="country codes for the pod's data centre (default: US; "
                         "give none to allow anywhere)")
    run_gpu.add_argument("--no-volume", action="store_true", help="upload everything instead of using the volume")
    run_gpu.add_argument("--no-fallback", action="store_true",
                         help="with a volume: wait for its data centre instead of running elsewhere")
    run_gpu.add_argument("--max-hours", type=float, default=4.0, help="hard limit; the pod also stops itself after this")
    g_sub.add_parser("cleanup", help="delete every soccer-clips pod")
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
    {"scan": cmd_scan, "list": cmd_list, "serve": cmd_serve, "bench": cmd_bench, "ball": cmd_ball, "gpu": cmd_gpu}[args.command](store, args)


if __name__ == "__main__":
    main()
