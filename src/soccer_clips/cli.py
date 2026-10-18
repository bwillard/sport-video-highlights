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
            print(benchmark.draw(store, args.variant, name, ball_conf=args.ball_conf))
    else:
        for c in bench.load_set(store.settings).clips:
            who = f"  callout #{c.callout_player}" if c.callout_player is not None else ("  callout" if c.callout else "")
            print(f"{c.name:24} {c.end_s - c.start_s:5.1f}s  {c.game_id}{who}  {c.why}")


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
    run.add_argument("variants", nargs="*", help="default: all (n640 s640 n1280 s1280 s1280-centre)")
    draw = bench_sub.add_parser("draw", help="draw a variant's boxes onto the clips for checking by eye")
    draw.add_argument("variant")
    draw.add_argument("clips", nargs="*", help="default: all")
    draw.add_argument("--ball-conf", type=float, default=0.1, help="weakest ball detection to draw")
    args = parser.parse_args()

    store = Store(get_settings())
    {"scan": cmd_scan, "list": cmd_list, "serve": cmd_serve, "bench": cmd_bench}[args.command](store, args)


if __name__ == "__main__":
    main()
