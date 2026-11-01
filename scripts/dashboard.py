"""Dashboard: the simulator (set up an episode, then replay it) and the results browser.

    .venv/bin/python scripts/dashboard.py [--results] [--results-dir results] [--port 8050] [--no-browser]

`--results` opens the results browser (every run under `results/`) instead of the simulator page.
"""
from __future__ import annotations

import argparse
import webbrowser

from acmad.dashboard.server import serve
from acmad.experiments.config import REPO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--results", action="store_true", help="open the results browser")
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    a = ap.parse_args()
    httpd = serve(a.host, a.port, a.results_dir)
    url = f"http://{a.host}:{a.port}/" + ("results" if a.results else "")
    print(f"dashboard on {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
