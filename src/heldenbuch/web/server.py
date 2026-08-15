"""The local control panel's HTTP server.

Standard library only, deliberately: the rest of the project has three
dependencies, and a single-user localhost control panel does not justify
pulling in a web framework. It binds to 127.0.0.1 and has no authentication,
so do not expose it beyond your own machine.

The browser polls `/api/jobs/<id>?since=N` for new log lines rather than using
server-sent events -- streaming responses out of `http.server` is fiddly, and
polling a local socket costs nothing.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import yaml

from .. import __version__
from ..backends import BACKEND_NAMES, REQUIRED_KEY
from ..book.library import Library
from ..config import load_dotenv, load_spec
from ..pipeline import RunLayout, load_pages
from ..prompts import scene_prompt, sheet_prompt
from ..types import STRATEGIES, BenchmarkSpec
from . import thumbs
from .benchjobs import BenchmarkJobs
from .bookapi import BookApi
from .bookjobs import BookJobs
from .jobs import JobManager

STATIC_DIR = Path(__file__).parent / "static"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- helpers


def _backend_status() -> list[dict[str, Any]]:
    from ..backends import comfy_available

    load_dotenv()
    rows = []
    for name in BACKEND_NAMES:
        key = REQUIRED_KEY[name]
        if name == "comfy":  # no key; a running local server is what counts
            ready = comfy_available()
        else:
            ready = key is None or bool(os.environ.get(key, "").strip())
        rows.append({"name": name, "key": key, "ready": ready})
    return rows


def _packages() -> dict[str, bool]:
    installed = {}
    for module in ("google.genai", "torch", "transformers"):
        try:
            __import__(module)
            installed[module] = True
        except ImportError:
            installed[module] = False
    return installed


def _run_summary(root: Path) -> dict[str, Any]:
    """Cheap metadata for the run list, read from the JSON the run already wrote."""
    info: dict[str, Any] = {"name": root.name, "mtime": root.stat().st_mtime}
    summary_path = root / "summary.json"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = payload.get("summary", [])
            info["rows"] = len(rows)
            info["backends"] = sorted({r["backend"] for r in rows})
            info["best"] = rows[0] if rows else None
        except (json.JSONDecodeError, KeyError):
            pass
    try:
        info["pages"] = len(json.loads((root / "pages.json").read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        info["pages"] = 0
    info["has_report"] = (root / "report.html").is_file()
    return info


# --------------------------------------------------------------------------- app


class Api:
    """Everything the browser can ask for. Each method returns a JSON-able object."""

    def __init__(self, spec_path: Path, runs_dir: Path, library_dir: Path) -> None:
        self.spec_path = spec_path
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # One queue for both halves of the app: they share the same image APIs
        # and the same rate limits.
        self.library = Library(library_dir)
        self.jobs = JobManager()
        self.jobs.register(BenchmarkJobs(spec_path, runs_dir).workers())
        self.jobs.register(BookJobs(self.library).workers())
        self.book = BookApi(self.library, self.jobs)

    # -- status ------------------------------------------------------------

    def status(self, _query, _body) -> dict[str, Any]:
        spec_error = None
        spec_info: dict[str, Any] = {}
        try:
            spec = load_spec(self.spec_path)
            spec_info = {
                "run_name": spec.run_name,
                "character": spec.character.name,
                "style": spec.style.name,
                "scenes": len(spec.scenes),
                "backends": spec.experiment.backends,
                "strategies": spec.experiment.strategies,
                "judge": {"provider": spec.judge.provider, "model": spec.judge.model},
            }
        except Exception as exc:
            spec_error = str(exc)

        active = self.jobs.active()
        return {
            "version": __version__,
            "spec_path": str(self.spec_path),
            "runs_dir": str(self.runs_dir),
            "backends": _backend_status(),
            "all_strategies": list(STRATEGIES),
            "packages": _packages(),
            "spec": spec_info,
            "spec_error": spec_error,
            "active_job": active.id if active else None,
            "jobs": self.jobs.recent(),
        }

    # -- spec --------------------------------------------------------------

    def spec_get(self, _query, _body) -> dict[str, Any]:
        return {"yaml": self.spec_path.read_text(encoding="utf-8")}

    def spec_put(self, _query, body) -> dict[str, Any]:
        text = (body or {}).get("yaml", "")
        if not isinstance(text, str) or not text.strip():
            raise ApiError("no YAML supplied")
        self._validate(text)
        if not (body or {}).get("validate_only"):
            self.spec_path.write_text(text, encoding="utf-8")
        return {"ok": True, "saved": not (body or {}).get("validate_only")}

    @staticmethod
    def _validate(text: str) -> BenchmarkSpec:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ApiError(f"YAML syntax error: {exc}") from exc
        if not isinstance(raw, dict):
            raise ApiError("the spec must be a YAML mapping")
        try:
            return BenchmarkSpec.from_dict(raw)
        except ValueError as exc:
            raise ApiError(str(exc)) from exc

    def prompts(self, query, _body) -> dict[str, Any]:
        spec = load_spec(self.spec_path)
        index = int((query.get("scene") or ["0"])[0])
        scene = spec.scenes[max(0, min(index, len(spec.scenes) - 1))]
        blocks = [{"label": "character sheet", "text": sheet_prompt(spec)}]
        for strategy in STRATEGIES:
            blocks.append(
                {
                    "label": strategy,
                    "text": scene_prompt(spec, scene, strategy, has_previous=False),
                }
            )
        blocks.append(
            {
                "label": "sheet_plus_prev (with a previous page)",
                "text": scene_prompt(spec, scene, "sheet_plus_prev", has_previous=True),
            }
        )
        return {
            "scene": {"id": scene.id, "action": scene.action, "stress": scene.stress},
            "scenes": [{"id": s.id, "stress": s.stress} for s in spec.scenes],
            "blocks": blocks,
        }

    def estimate(self, query, _body) -> dict[str, Any]:
        """How many images a given selection would generate, before spending money."""
        spec = load_spec(self.spec_path)
        backends = query.get("backends") or spec.experiment.backends
        strategies = query.get("strategies") or spec.experiment.strategies
        limit = int((query.get("limit") or [0])[0] or 0)
        scenes = min(limit, len(spec.scenes)) if limit else len(spec.scenes)
        paid = [b for b in backends if b != "stub"]
        pages = len(backends) * len(strategies) * scenes
        return {
            "scenes": scenes,
            "pages": pages,
            "sheets": len(backends),
            "total_images": pages + len(backends),
            "paid_images": len(paid) * len(strategies) * scenes + len(paid),
            "judge_calls": pages,
        }

    # -- jobs --------------------------------------------------------------

    def job_start(self, _query, body) -> dict[str, Any]:
        body = body or {}
        action = body.get("action", "run")
        params = {k: v for k, v in body.items() if k != "action"}
        try:
            job = self.jobs.start(action, params)
        except (ValueError, RuntimeError) as exc:
            raise ApiError(str(exc), status=409) from exc
        return job.public()

    def job_get(self, job_id: str, query, _body) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise ApiError(f"no job {job_id}", status=404)
        return job.public(since=int((query.get("since") or ["0"])[0]))

    def job_cancel(self, job_id: str, _query, _body) -> dict[str, Any]:
        return {"cancelled": self.jobs.cancel(job_id)}

    def job_retry(self, job_id: str, _query, _body) -> dict[str, Any]:
        try:
            return self.jobs.retry(job_id).public()
        except ValueError as exc:
            raise ApiError(str(exc), status=404) from exc

    # -- runs --------------------------------------------------------------

    def runs(self, _query, _body) -> dict[str, Any]:
        found = [p for p in self.runs_dir.glob("*") if (p / "pages.json").is_file()]
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return {"runs": [_run_summary(p) for p in found]}

    def run_detail(self, name: str, _query, _body) -> dict[str, Any]:
        root = self._safe_run(name)
        layout = RunLayout(root)
        records = [r.to_dict() for r in load_pages(layout)]
        summary: dict[str, Any] = {}
        summary_path = root / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        contacts = sorted(p.name for p in root.glob("contact_*.png"))
        sheets = {
            p.parent.name: f"{p.parent.name}/sheet.png"
            for p in root.glob("*/sheet.png")
        }
        return {
            "name": root.name,
            "pages": records,
            "summary": summary.get("summary", []),
            "hardest_scenes": summary.get("hardest_scenes", []),
            "contact_sheets": contacts,
            "sheets": sheets,
            "has_report": (root / "report.html").is_file(),
        }

    def run_delete(self, name: str, _query, _body) -> dict[str, Any]:
        import shutil

        root = self._safe_run(name)
        shutil.rmtree(root)
        return {"deleted": root.name}

    def _safe_run(self, name: str) -> Path:
        root = (self.runs_dir / unquote(name)).resolve()
        if not root.is_relative_to(self.runs_dir.resolve()) or not root.is_dir():
            raise ApiError(f"no run named {name!r}", status=404)
        return root

    def file_path(self, relative: str) -> tuple[Path, str]:
        """Locate an image or the report inside the runs directory."""
        target = (self.runs_dir / unquote(relative)).resolve()
        if not target.is_relative_to(self.runs_dir.resolve()) or not target.is_file():
            raise ApiError("not found", status=404)
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return target, mime


# --------------------------------------------------------------------------- routing

Handler = Callable[..., Any]


class RequestHandler(BaseHTTPRequestHandler):
    api: Api  # injected by serve()
    server_version = f"Heldenbuch/{__version__}"
    protocol_version = "HTTP/1.1"

    # Routes with a single capture group are passed that group as the first arg.
    # (method, pattern, attribute path on the Api object)
    ROUTES: list[tuple[str, str, str]] = [
        # the book app
        ("GET", r"^/api/book/status$", "book.status"),
        ("GET", r"^/api/book/heroes$", "book.heroes"),
        ("GET", r"^/api/book/heroes/([^/]+)$", "book.hero"),
        ("PUT", r"^/api/book/heroes/([^/]+)$", "book.hero_update"),
        ("DELETE", r"^/api/book/heroes/([^/]+)$", "book.hero_delete"),
        ("GET", r"^/api/book/styles$", "book.styles"),
        ("DELETE", r"^/api/book/styles/([^/]+)$", "book.style_delete"),
        ("GET", r"^/api/book/books$", "book.books"),
        ("POST", r"^/api/book/books/([^/]+)/backup$", "book.book_backup"),
        ("GET", r"^/api/book/books/([^/]+)/preflight$", "book.book_preflight"),
        ("GET", r"^/api/book/books/([^/]+)/page-preview$", "book.book_page_preview"),
        ("GET", r"^/api/book/backups$", "book.backups"),
        ("POST", r"^/api/book/restore$", "book.book_restore"),
        ("POST", r"^/api/book/books/([^/]+)/open-folder$", "book.book_open_folder"),
        ("GET", r"^/api/book/books/([^/]+)$", "book.book"),
        ("PUT", r"^/api/book/books/([^/]+)$", "book.book_update"),
        ("DELETE", r"^/api/book/books/([^/]+)$", "book.book_delete"),
        # shared job queue
        ("POST", r"^/api/jobs$", "job_start"),
        ("GET", r"^/api/jobs/([^/]+)$", "job_get"),
        ("POST", r"^/api/jobs/([^/]+)/cancel$", "job_cancel"),
        ("POST", r"^/api/jobs/([^/]+)/retry$", "job_retry"),
        # the benchmark panel
        ("GET", r"^/api/status$", "status"),
        ("GET", r"^/api/spec$", "spec_get"),
        ("PUT", r"^/api/spec$", "spec_put"),
        ("GET", r"^/api/prompts$", "prompts"),
        ("GET", r"^/api/estimate$", "estimate"),
        ("GET", r"^/api/runs$", "runs"),
        ("GET", r"^/api/runs/([^/]+)$", "run_detail"),
        ("DELETE", r"^/api/runs/([^/]+)$", "run_delete"),
    ]

    def log_message(self, *_args) -> None:  # keep the console clean
        pass

    # -- plumbing ----------------------------------------------------------

    def _send(self, status: int, body: bytes, mime: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_file(self, target: Path, mime: str) -> None:
        """Serve a disk file, answering 304 when the browser already has it.

        The ETag comes from mtime and size, not a hash of the contents:
        pages are multi-megabyte, every write in this project replaces the
        file (fresh mtime), and hashing per request meant a full read even
        for a 304. Caching at all matters -- without it a page grid
        re-downloaded tens of megabytes on every render and repainted grey
        at exactly the moment a long render finished.
        """
        try:
            st = target.stat()
        except OSError:
            self._send_json({"error": "not found"}, status=404)
            return
        tag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if self.headers.get("If-None-Match") == tag:
            self.send_response(304)
            self.send_header("ETag", tag)
            self.send_header("Cache-Control", "private, max-age=60")
            self.end_headers()
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", tag)
        self.send_header("Cache-Control", "private, max-age=60")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _content_length(self) -> int:
        try:
            return max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            return 0

    def _drain_body(self) -> None:
        remaining = self._content_length()
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)

    def _read_body(self) -> Any:
        length = self._content_length()
        if not length:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError(f"invalid JSON body: {exc}") from exc

    # -- verbs -------------------------------------------------------------

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:
        # Answered explicitly, with no CORS headers, so a cross-origin
        # preflight is refused by design rather than by the default 501.
        self._send_json({"error": "method not allowed"}, status=405)

    #: Hosts this server will answer to. Anything else is a browser that
    #: resolved some other name to this address -- the DNS-rebinding shape,
    #: where a remote page becomes same-origin with a localhost service and can
    #: then read every response. Checking the header costs nothing and is the
    #: only defence, since the request otherwise looks entirely normal.
    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1", "")

    def _origin_allowed(self) -> bool:
        """A state-changing request must not come from another site.

        A cross-origin page can send a form-style POST without a preflight, so
        the side effect would happen even though it cannot read the reply --
        which for this app means spending money. Requiring a JSON content type
        and an absent or matching Origin closes that: neither can be set
        cross-origin without a preflight, and `do_OPTIONS` refuses those.
        """
        origin = self.headers.get("Origin")
        if origin:
            host = urlparse(origin).hostname or ""
            if host.lower() not in ("127.0.0.1", "localhost", "::1"):
                return False
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        return ctype in ("application/json", "")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if self.headers.get("Transfer-Encoding"):
            self._send_json({"error": "chunked bodies are not accepted"}, status=501)
            return
        if not self._host_allowed():
            self._send_json({"error": "host not allowed"}, status=421)
            return
        if method in ("POST", "PUT", "DELETE") and not self._origin_allowed():
            self._send_json({"error": "cross-origin request refused"}, status=403)
            return

        try:
            for route_method, pattern, name in self.ROUTES:
                if route_method != method:
                    continue
                match = re.match(pattern, path)
                if not match:
                    continue
                handler = self.api
                for part in name.split("."):
                    handler = getattr(handler, part)
                body = self._read_body() if method in ("POST", "PUT") else None
                result = handler(*match.groups(), query, body)
                self._send_json(result)
                return

            if method == "GET":
                self._serve_asset(path, query)
                return
            # Nothing matched. The body still has to come off the socket --
            # this connection is keep-alive, so bytes left unread would be
            # parsed as the start of the next request.
            self._drain_body()
            self._send_json({"error": "not found"}, status=404)

        except ApiError as exc:
            self._send_json({"error": str(exc)}, status=exc.status)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except ValueError as exc:
            # Rejected input, not a server fault -- a malformed id or an
            # unparseable number reaches here.
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    #: URL path -> static file. Everything else is looked up by name.
    PAGES = {"/": "app.html", "": "app.html", "/benchmark": "index.html"}

    def _serve_asset(self, path: str, query: dict[str, list[str]]) -> None:
        if path.startswith("/files/"):  # benchmark runs
            target, mime = self.api.file_path(path[len("/files/"):])
            self._send_file(target, mime)
            return
        if path.startswith("/library/"):  # heroes, styles, books
            target, mime = self.api.book.file_path(path[len("/library/"):])
            if query.get("thumb") and thumbs.thumbable(target):
                small = thumbs.thumbnail(target, self.api.library.root)
                if small is not target:
                    target, mime = small, "image/jpeg"
            self._send_file(target, mime)
            return

        name = self.PAGES.get(path) or path.lstrip("/")
        target = (STATIC_DIR / name).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        mime = mimetypes.guess_type(target.name)[0] or "text/plain"
        self._send(200, target.read_bytes(), f"{mime}; charset=utf-8")


def serve(
    spec_path: Path,
    runs_dir: Path,
    library_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    api = Api(spec_path, runs_dir, library_dir)
    handler = type("BoundHandler", (RequestHandler,), {"api": api})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{httpd.server_address[1]}"

    print(f"Heldenbuch on {url}")
    print(f"  Bücher:    {library_dir}")
    print(f"  Benchmark: {url}/benchmark")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: __import__("webbrowser").open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
