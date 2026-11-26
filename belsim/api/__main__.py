"""`uv run python -m belsim.api [--host 127.0.0.1] [--port 8000] [--no-reload] [--public]`

Restarts itself when model code changes, so a running dashboard never mixes old code with new
parameter files. `--public` (or `BELSIM_PUBLIC=1`) is the mode for a server open to anyone:
nothing written to disk, bounded requests, security headers, no restart on change. See docs/DEPLOY.md.
"""
import argparse
import os

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="belsim API + dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-reload", action="store_true", help="do not restart when Python files change")
    ap.add_argument("--public", action="store_true",
                    help="public mode: read-only, bounded, security headers, no auto-reload (see docs/DEPLOY.md)")
    a = ap.parse_args()
    if a.public:
        os.environ["BELSIM_PUBLIC"] = "1"       # read by belsim.api.public when the app is created
    is_public = os.environ.get("BELSIM_PUBLIC", "").strip().lower() in ("1", "true", "yes", "on")
    reload = not a.no_reload and not is_public
    print(f"belsim dashboard: http://{a.host}:{a.port}/" + (" (public mode)" if is_public else ""))
    uvicorn.run("belsim.api.app:app", host=a.host, port=a.port, reload=reload,
                reload_dirs=["belsim"] if reload else None, log_level="info",
                server_header=not is_public)


if __name__ == "__main__":
    main()
