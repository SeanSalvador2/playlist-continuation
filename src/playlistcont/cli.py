"""playlistcont command-line interface.

    playlistcont dashboard [--host 127.0.0.1] [--port 8000] [--no-browser]

Launches the **Taste Atlas** local web app: a seeded synthetic MPD is generated
and the interpretable Taste Engine (+ item-CF, popularity) is fitted in memory,
then a FastAPI server serves the recommender API and the built React dashboard
from one port.  No data downloads, fully self-contained.
"""
from __future__ import annotations

import sys
import webbrowser


def dashboard(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="playlistcont dashboard",
        description="Launch the Taste Atlas interactive dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window automatically.")
    args = parser.parse_args(argv)

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("The dashboard needs the web extra:  pip install 'playlistcont[app]'",
              file=sys.stderr)
        return 1

    # the FastAPI app package lives at the repo root (alongside src/), not inside
    # the installed package — make sure it is importable regardless of cwd.
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from app.backend.server import create_app

    print("Building the synthetic MPD and fitting models (a few seconds)...")
    app = create_app(warm=True)
    url = f"http://{args.host}:{args.port}"
    print(f"Taste Atlas is live at {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "dashboard":
        return dashboard(rest)
    print(f"Unknown command: {cmd!r}. Try 'playlistcont dashboard'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
