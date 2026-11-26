"""Public mode (`python -m belsim.api --public` or `BELSIM_PUBLIC=1`): what changes when the
dashboard is served to anyone rather than to its owner on localhost.

- nothing is written to disk on request (saving a scenario on the server is refused; the
  dashboard keeps scenarios in the browser);
- the expensive endpoints are bounded: horizon, scenarios per batch, measures per scenario,
  request body size, a per-address rate limit and a small number of simulations at a time;
- errors carry no file path and no exception text;
- requests that change something must come from the same origin; no CORS header is ever sent;
- security headers and cache headers are added;
- the interactive API documentation is not served.

Everything here is in-process and best effort. It does not replace a reverse proxy (TLS,
connection limits, time-outs) and it is no defence against a distributed flood: see docs/DEPLOY.md.
Every limit can be changed with an environment variable (`Limits.from_env`).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from ..core.dims import BASE_YEAR, END_YEAR

TRUE = ("1", "true", "yes", "on")


def is_public() -> bool:
    return os.environ.get("BELSIM_PUBLIC", "").strip().lower() in TRUE


def read_only() -> bool:
    """No scenario is saved or deleted on the server (public mode, or the older
    `BELSIM_READ_ONLY=1` on its own)."""
    return is_public() or os.environ.get("BELSIM_READ_ONLY", "").strip().lower() in TRUE


def _int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, "") or default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Limits:
    max_end: int = END_YEAR          # last simulated year
    max_batch: int = 8               # scenarios in one /api/run_batch
    max_measures: int = 80           # measures in one scenario
    max_changes: int = 400           # parameter changes in one scenario
    max_body: int = 256 * 1024       # bytes of a request body
    rate: int = 240                  # requests to /api per address per minute
    rate_heavy: int = 40             # of which requests that run simulations
    concurrency: int = 2             # simulations requests served at the same time
    queue: int = 8                   # ... and waiting; beyond that the answer is 503
    trust_proxy: bool = False        # read the address from X-Forwarded-For (rightmost entry)
    origins: tuple[str, ...] = ()    # extra allowed origins (host[:port]) for POST / DELETE

    @classmethod
    def from_env(cls) -> "Limits":
        d = cls()
        return cls(
            max_end=_int("BELSIM_MAX_END", d.max_end, BASE_YEAR + 1, END_YEAR),
            max_batch=_int("BELSIM_MAX_BATCH", d.max_batch, 1, 12),
            max_measures=_int("BELSIM_MAX_MEASURES", d.max_measures, 1, 1000),
            max_changes=_int("BELSIM_MAX_CHANGES", d.max_changes, 1, 10000),
            max_body=_int("BELSIM_MAX_BODY", d.max_body, 1024, 16 * 1024 * 1024),
            rate=_int("BELSIM_RATE", d.rate, 1, 100000),
            rate_heavy=_int("BELSIM_RATE_HEAVY", d.rate_heavy, 1, 100000),
            concurrency=_int("BELSIM_CONCURRENCY", d.concurrency, 1, 64),
            queue=_int("BELSIM_QUEUE", d.queue, 0, 1024),
            trust_proxy=os.environ.get("BELSIM_TRUST_PROXY", "").strip().lower() in TRUE,
            origins=tuple(o.strip().lower() for o in os.environ.get("BELSIM_ALLOWED_ORIGINS", "").split(",") if o.strip()),
        )


def limits() -> Limits:
    return Limits.from_env()


# Content-Security-Policy for a front end without CDN, inline script or eval. Inline styles are
# allowed: the charts position their marks with style attributes.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; "
       "base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    "content-security-policy": CSP,
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
    "x-frame-options": "DENY",
    "cross-origin-opener-policy": "same-origin",
    "permissions-policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
}

# GET answers that depend only on the parameter files and the language in the URL.
CACHEABLE_API = ("/api/meta", "/api/baseline", "/api/params", "/api/measures", "/api/version",
                 "/api/payslip/options")
HEAVY_GET = ("/api/measures/effects", "/api/parties/", "/api/arrays", "/api/pyramid")
HEAVY_POST = ("/api/run", "/api/run_batch", "/api/run_band", "/api/distribution", "/api/validate")


def cache_control(path: str, method: str, status: int) -> str:
    if status != 200 or method not in ("GET", "HEAD") or path == "/healthz":
        return "no-store"
    if path.startswith("/vendor/") or path in ("/og.png", "/favicon.svg"):
        return "public, max-age=604800"
    if path.startswith("/api/"):
        return "public, max-age=300" if path in CACHEABLE_API else "no-store"
    return "public, max-age=300"


def is_heavy(path: str, method: str) -> bool:
    if method == "POST":
        return path in HEAVY_POST
    return path.startswith(HEAVY_GET) and path != "/api/measures"


class _Buckets:
    """Token buckets per address: `rate` requests a minute, refilled continuously."""

    def __init__(self, rate: int, max_keys: int = 20000):
        self.rate, self.max_keys, self.data = rate, max_keys, {}

    def take(self, key: str, now: float | None = None) -> float:
        """0 when the request may pass, else the seconds to wait."""
        now = time.monotonic() if now is None else now
        tokens, last = self.data.get(key, (float(self.rate), now))
        tokens = min(float(self.rate), tokens + (now - last) * self.rate / 60.0)
        if tokens < 1.0:
            self.data[key] = (tokens, now)
            return (1.0 - tokens) * 60.0 / self.rate
        if len(self.data) >= self.max_keys and key not in self.data:
            for k in sorted(self.data, key=lambda k: self.data[k][1])[: self.max_keys // 2]:
                del self.data[k]            # forget the addresses seen longest ago
        self.data[key] = (tokens - 1.0, now)
        return 0.0


def _host_of(origin: str) -> str:
    try:
        return (urlsplit(origin).netloc or "").lower()
    except ValueError:
        return ""


class PublicGuard:
    """ASGI middleware of the public mode (pure ASGI: it must see the body before FastAPI)."""

    def __init__(self, app, lim: Limits | None = None):
        self.app = app
        self.lim = lim or limits()
        self.all = _Buckets(self.lim.rate)
        self.heavy = _Buckets(self.lim.rate_heavy)
        self.sem: asyncio.Semaphore | None = None
        self.waiting = 0

    # ------------------------------------------------------------ helpers
    def client(self, scope, headers: dict[bytes, bytes]) -> str:
        if self.lim.trust_proxy:
            fwd = headers.get(b"x-forwarded-for", b"").decode("latin-1")
            last = fwd.split(",")[-1].strip()
            if last:
                return last
        return (scope.get("client") or ("unknown", 0))[0]

    @staticmethod
    async def refuse(send, status: int, message: str, extra: dict[str, str] | None = None) -> None:
        body = json.dumps({"error": message}).encode()
        head = {"content-type": "application/json", "content-length": str(len(body)),
                "cache-control": "no-store", **SECURITY_HEADERS, **(extra or {})}
        await send({"type": "http.response.start", "status": status,
                    "headers": [(k.encode(), v.encode()) for k, v in head.items()]})
        await send({"type": "http.response.body", "body": body})

    def same_origin(self, headers: dict[bytes, bytes]) -> bool:
        origin = headers.get(b"origin", b"").decode("latin-1").strip()
        if not origin:
            return True                 # not a browser cross-site request (curl, same-origin GET)
        host = _host_of(origin)
        ours = {headers.get(b"host", b"").decode("latin-1").strip().lower(), *self.lim.origins}
        if self.lim.trust_proxy:
            ours.add(headers.get(b"x-forwarded-host", b"").decode("latin-1").strip().lower())
        return bool(host) and host in ours

    # ------------------------------------------------------------ ASGI
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope.get("path", ""), scope.get("method", "GET").upper()
        headers = {k.lower(): v for k, v in scope.get("headers") or []}

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                out = [(k, v) for k, v in message.get("headers") or []
                       if k.lower() not in (b"cache-control", b"server")]
                out.append((b"cache-control", cache_control(path, method, message["status"]).encode()))
                out.append((b"vary", b"Accept-Language, Accept-Encoding"))
                out += [(k.encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]
                message = {**message, "headers": out}
            await send(message)

        if method not in ("GET", "HEAD", "POST", "DELETE", "OPTIONS"):
            return await self.refuse(send, 405, "method not allowed")
        if method in ("POST", "DELETE") and not self.same_origin(headers):
            return await self.refuse(send, 403, "cross-origin requests are not accepted")
        if not path.startswith("/api/"):
            return await self.app(scope, receive, send_with_headers)

        heavy = is_heavy(path, method)
        who = self.client(scope, headers)
        wait = self.all.take(who) or (self.heavy.take(who) if heavy else 0.0)
        if wait:
            return await self.refuse(send, 429, "too many requests from this address; try again shortly",
                                     {"retry-after": str(max(1, int(wait + 0.999)))})

        if method == "POST":            # read the body here, never more than the limit
            try:
                declared = int(headers.get(b"content-length", b"0") or 0)
            except ValueError:
                declared = 0
            if declared > self.lim.max_body:
                return await self.refuse(send, 413, f"request body larger than {self.lim.max_body} bytes")
            chunks, size, more = [], 0, True
            while more:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return
                chunk = msg.get("body", b"")
                size += len(chunk)
                if size > self.lim.max_body:
                    return await self.refuse(send, 413, f"request body larger than {self.lim.max_body} bytes")
                chunks.append(chunk)
                more = msg.get("more_body", False)
            body, sent = b"".join(chunks), False

            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()
            inner_receive = replay
        else:
            inner_receive = receive

        if not heavy:
            return await self.app(scope, inner_receive, send_with_headers)
        if self.sem is None:
            self.sem = asyncio.Semaphore(self.lim.concurrency)
        if self.sem.locked() and self.waiting >= self.lim.queue:
            return await self.refuse(send, 503, "the server is busy; try again in a few seconds", {"retry-after": "5"})
        self.waiting += 1
        try:
            await self.sem.acquire()
        finally:
            self.waiting -= 1
        try:
            await self.app(scope, inner_receive, send_with_headers)
        finally:
            self.sem.release()


def check_scenario_size(scen: dict, lim: Limits) -> str | None:
    """A message when a scenario is larger than the public limits, else None."""
    measures = scen.get("measures") or []
    if len(measures) > lim.max_measures:
        return f"a scenario may hold at most {lim.max_measures} measures on this server"
    if sum(len(m.get("changes") or []) for m in measures) > lim.max_changes:
        return f"a scenario may hold at most {lim.max_changes} parameter changes on this server"
    return None
