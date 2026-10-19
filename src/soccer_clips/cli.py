import argparse
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
            print(ball_labels.score_table({label: scores[label]}).split("\n", 1)[1])
    elif args.ball_command == "score":
        from soccer_clips.track.bench import bench_dir

        scores = ball_labels.score(store, bench_dir(store.settings) / args.results)
        print(ball_labels.score_table(scores))
    else:
        for s in ball_labels.summary(store):
            print(f"{s['name']:8} {s['labelled']}/{s['total']} labelled")


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
    run.add_argument("variants", nargs="*", help="default: all (n640 s640 n1280 s1280 s1280-centre s1920 m1280-centre m1920)")
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
    ball_sub.add_parser("export", help="turn train-set labels into a YOLO dataset (labels/dataset/)")
    train = ball_sub.add_parser("train", help="fine-tune YOLO bases on the exported dataset (GPU)")
    train.add_argument("bases", nargs="+", help="e.g. yolo11m yolo26l; weights go to models/ball_<base>.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=8)
    ev = ball_sub.add_parser("eval", help="score models on the bench frame images")
    ev.add_argument("models", nargs="+", help="names in models/ (yolo11m, ball_yolo11m, ...) or .pt paths")
    ev.add_argument("--full", action="store_true", help="whole frame instead of the centre crop")
    ev.add_argument("--imgsz", type=int, default=1280)
    score = ball_sub.add_parser("score", help="score benchmark ball detections against the bench labels")
    score.add_argument("--results", default="results", help="results folder in bench/ (e.g. results-l4)")
    args = parser.parse_args()

    store = Store(get_settings())
    {"scan": cmd_scan, "list": cmd_list, "serve": cmd_serve, "bench": cmd_bench, "ball": cmd_ball}[args.command](store, args)


if __name__ == "__main__":
    main()
