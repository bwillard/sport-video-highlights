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


def main() -> None:
    parser = argparse.ArgumentParser(prog="soccer-clips")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="register new videos found in SOCCER_FULL_DIR/<year>/")
    sub.add_parser("list", help="list registered games")
    serve = sub.add_parser("serve", help="run the web server (scans on startup)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    args = parser.parse_args()

    store = Store(get_settings())
    {"scan": cmd_scan, "list": cmd_list, "serve": cmd_serve}[args.command](store, args)


if __name__ == "__main__":
    main()
