"""Local HTTP server for the dashboard (stdlib only).

Simulator: the page `/`, `/api/options`, `/api/run`. Results browser: the page `/results`,
`/api/results` (every run), `/api/results/<experiment>/<run>` (one run) and
`/api/decisions/<experiment>/<run>?method=&scenario=&seed=` (one cell's decisions).
"""
from __future__ import annotations

import json
import traceback
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from acmad.dashboard.episode import options, run_episode
from acmad.dashboard.decisions import cell_decisions
from acmad.dashboard.results import list_runs, resolve, run_detail
from acmad.experiments.config import REPO

PAGE = Path(__file__).with_name("index.html")
RESULTS_PAGE = Path(__file__).with_name("results.html")


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, results_root: Path, **kw) -> None:
        self.results_root = results_root
        super().__init__(*args, **kw)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=_default).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/options":
            self._json(200, options())
        elif self.path in ("/results", "/results/"):
            self._send(200, RESULTS_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/results":
            self._json(200, list_runs(self.results_root))
        elif self.path.startswith("/api/decisions/"):
            u = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                d = resolve(self.results_root.resolve(), unquote(u.path.removeprefix("/api/decisions/")))
                self._json(200, cell_decisions(d, q["method"], q["scenario"], int(q["seed"])))
            except (FileNotFoundError, KeyError, ValueError) as err:
                self._json(404, {"error": f"{type(err).__name__}: {err}"})
        elif self.path.startswith("/api/results/"):
            try:
                self._json(200, run_detail(self.results_root, unquote(self.path.removeprefix("/api/results/"))))
            except FileNotFoundError as err:
                self._json(404, {"error": str(err)})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        try:
            params = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._json(200, run_episode(params))
        except Exception as err:   # shown in the page
            self._json(400, {"error": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc()})

    def log_message(self, fmt: str, *args) -> None:
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def _default(x):
    if hasattr(x, "item"):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def serve(host: str = "127.0.0.1", port: int = 8050, results_root: str | Path = REPO / "results") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), partial(Handler, results_root=Path(results_root)))
