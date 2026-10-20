"""Who may talk to the engine's HTTP API.

The API can read the user's video files and library, so only the desktop app may use it:

- It listens on 127.0.0.1 only, never on a network interface (see ``server.py``).
- Every request needs ``Authorization: Bearer <token>``. The token is random per start
  and only the process that started the engine knows it (it is handed over through an
  environment variable or the first line the engine prints, never on the command line,
  where other programs could read it).
- The ``Host`` header must be a loopback name. Otherwise a web page could reach the API
  through DNS rebinding (a site name that suddenly points to 127.0.0.1).
- Browsers may only call it from the desktop app's own origins (CORS); preflight
  requests carry no token and are answered by the CORS layer alone.

Tokens are never logged.
"""

from __future__ import annotations

import os
import re
import secrets
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

TOKEN_ENV = "CLIPSDEX_API_TOKEN"
MIN_TOKEN_LENGTH = 32
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
# The desktop app's web view (Tauri v2 on Windows, macOS/Linux) and its dev server.
DEFAULT_ORIGINS = (
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
)


EXTRA_ORIGINS_ENV = "CLIPSDEX_API_EXTRA_ORIGINS"  # e.g. the dev server: http://localhost:1420
LOOPBACK_ORIGIN = re.compile(r"https?://(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?")


class LaunchError(ValueError):
    """What the parent process passed in (token, origins) is unusable. Message is user-facing."""


def allowed_origins() -> tuple[str, ...]:
    """The app's origins, plus extra ones from the environment (development: the Vite
    server). Extra origins must be loopback addresses; anything else is refused."""
    extra = [o.strip() for o in os.environ.get(EXTRA_ORIGINS_ENV, "").split(",") if o.strip()]
    for origin in extra:
        if not LOOPBACK_ORIGIN.fullmatch(origin):
            raise LaunchError(f"{EXTRA_ORIGINS_ENV} may only list local addresses, not {origin}.")
    return (*DEFAULT_ORIGINS, *extra)


def new_token() -> str:
    return secrets.token_urlsafe(32)  # 256 bits


def token_from_env() -> str | None:
    """The token the desktop app chose, if it passed one; it must be long enough."""
    token = os.environ.get(TOKEN_ENV)
    if token is None:
        return None
    if len(token) < MIN_TOKEN_LENGTH or not token.isascii() or not token.isprintable():
        raise LaunchError(
            f"{TOKEN_ENV} must be at least {MIN_TOKEN_LENGTH} printable ASCII characters."
        )
    return token


def host_name(host_header: str) -> str:
    """'127.0.0.1:5123' -> '127.0.0.1', '[::1]:5123' -> '[::1]'."""
    if host_header.startswith("["):
        return host_header.split("]", 1)[0] + "]"
    return host_header.rsplit(":", 1)[0] if ":" in host_header else host_header


def _deny(status: int, message: str) -> Response:
    return JSONResponse({"error": message}, status_code=status)


class GuardMiddleware(BaseHTTPMiddleware):
    """Rejects requests with a non-loopback Host or without the right bearer token."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        super().__init__(app)
        self._expected = f"Bearer {token}".encode()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if host_name(request.headers.get("host", "")).lower() not in LOOPBACK_HOSTS:
            return _deny(400, "This API only answers on the local machine.")
        given = request.headers.get("authorization", "").encode()
        if not secrets.compare_digest(given, self._expected):
            return _deny(401, "Missing or wrong token.")
        return await call_next(request)
