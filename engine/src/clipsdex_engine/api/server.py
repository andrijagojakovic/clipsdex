"""Start the API on 127.0.0.1 and tell the parent process where it is.

The socket is bound here (not by Uvicorn) so the port can be 0, meaning "any free port":
the actual port is read back from the socket. Once the server accepts connections, one
JSON line goes to stdout:

    {"event": "ready", "port": 53124, "token": "..."}

``token`` is only included when the engine made it up itself; when the parent passed one
in ``CLIPSDEX_API_TOKEN``, it already knows it and it is never printed.

With ``exit_with_stdin`` the engine stops when its stdin closes. The desktop app keeps a
pipe to it open, so if the app exits or crashes, the pipe closes and the engine does not
stay behind in the background.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from typing import Any, TextIO

import uvicorn

from clipsdex_engine.api.app import create_app
from clipsdex_engine.api.security import allowed_origins, new_token, token_from_env
from clipsdex_engine.config import DataPaths

HOST = "127.0.0.1"  # never configurable: the API must not be reachable from the network


def bind(port: int = 0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform == "win32":
        # Without this, another program could bind the same port and steal requests.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]
    sock.bind((HOST, port))
    return sock


class _Server(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, on_ready: Any) -> None:
        super().__init__(config)
        self._on_ready = on_ready

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        if self.started:
            self._on_ready()


def stop_when_closed(stream: TextIO, server: uvicorn.Server) -> threading.Thread:
    """Ask ``server`` to stop once ``stream`` reaches end of file (the parent is gone)."""

    def watch() -> None:
        try:
            while stream.read(4096):
                pass  # the parent never writes; anything it sends is ignored
        except (OSError, ValueError):
            pass
        server.should_exit = True

    thread = threading.Thread(target=watch, name="clipsdex-parent-watch", daemon=True)
    thread.start()
    return thread


def serve(
    paths: DataPaths,
    port: int = 0,
    out: TextIO = sys.stdout,
    exit_with_stdin: bool = False,
) -> None:
    """Run until stopped (Ctrl+C, the parent ends the process, or stdin closes)."""
    given = token_from_env()
    token = given or new_token()
    origins = allowed_origins()
    sock = bind(port)
    actual_port = sock.getsockname()[1]
    app = create_app(paths, token, allowed_origins=origins)

    def ready() -> None:
        line: dict[str, Any] = {"event": "ready", "port": actual_port}
        if given is None:
            line["token"] = token
        out.write(json.dumps(line) + "\n")
        out.flush()

    config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on")
    server = _Server(config, ready)
    if exit_with_stdin:
        stop_when_closed(sys.stdin, server)
    server.run(sockets=[sock])
