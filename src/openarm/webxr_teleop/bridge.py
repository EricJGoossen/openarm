"""Drop-in replacement for xrobotoolkit_sdk, backed by a WebXR page running
in the Quest's own browser instead of a native APK.

Exposes the same function names your teleop script already calls
(init, close, get_left_controller_pose, get_right_controller_pose,
get_left_grip, get_right_grip, get_left_trigger, get_right_trigger) so
switching pose sources is a one-line import change, same as when we
switched from oculus_reader to xrobotoolkit_sdk earlier.

UNVERIFIED end-to-end -- this has not yet been run against a real headset
connection. Built from the documented WebXR Input API and XRRigidTransform
format, following the same [x, y, z, qx, qy, qz, qw] pose convention used
elsewhere in this project.

Requires: pip install websockets

Usage from your teleop script:
    import webxr_pose_bridge as xrt
    xrt.init(cert_file="cert.pem", key_file="key.pem")
    xrt.wait_for_data(timeout=5.0)
    ...
    xrt.get_state("right")
    ...
    xrt.close()

On the Quest 3S: open the Quest Browser and navigate to the URL init()
prints (https://<detected-lan-ip>:8443/), accept the self-signed cert
warning, tap "Enter AR".

Any other browser (on a different machine, same network) that opens the
same URL becomes a passive diagnostics viewer -- it can't start an AR
session itself, but it receives the same live pose/grip/trigger state
broadcast from this bridge and renders the same hand-tracking display.
Useful precisely because the headset's own page goes dark once it enters
the AR session.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socket
import ssl
import threading
import time
from pathlib import Path

import websockets

_DEFAULT_ZERO_ENTRY = {"pose7": [0.0] * 7, "trigger": 0.0, "grip": 0.0}

# How stale a side's data can be before is_tracked() reports it as untracked.
DEFAULT_MAX_AGE_S = 0.5


def _get_local_ip() -> str:
    """Best-effort discovery of this machine's LAN IP, so the startup
    message can print a URL that's ready to paste into the Quest Browser
    instead of a placeholder.

    Opens a UDP socket toward a public address without sending any
    packets -- just to see which local interface/IP the OS would route
    through -- so this doesn't depend on hostname/DNS being configured,
    which is often unreliable on headless machines.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "<could not detect LAN IP -- check `ip addr` and substitute manually>"
    finally:
        s.close()


class _SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._left = dict(_DEFAULT_ZERO_ENTRY)
        self._right = dict(_DEFAULT_ZERO_ENTRY)
        self._left_updated_at: float | None = None
        self._right_updated_at: float | None = None

    def update(self, payload: dict) -> None:
        now = time.monotonic()
        with self._lock:
            if payload.get("left") is not None:
                if self._left_updated_at is None:
                    print("WebXR bridge: first pose data received from LEFT controller")
                self._left = payload["left"]
                self._left_updated_at = now
            if payload.get("right") is not None:
                if self._right_updated_at is None:
                    print("WebXR bridge: first pose data received from RIGHT controller")
                self._right = payload["right"]
                self._right_updated_at = now

    def get(self, side: str) -> dict:
        with self._lock:
            if side == "left":
                return dict(self._left)
            return dict(self._right)

    def last_updated(self, side: str) -> float | None:
        with self._lock:
            return self._left_updated_at if side == "left" else self._right_updated_at

    def any_data_received(self) -> bool:
        with self._lock:
            return self._left_updated_at is not None or self._right_updated_at is not None


_state = _SharedState()
_ws_loop: asyncio.AbstractEventLoop | None = None
_ws_thread: threading.Thread | None = None
_http_server: http.server.ThreadingHTTPServer | None = None
_http_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None


# Connected WebSocket clients (the headset, plus any passive diagnostic
# viewers on other machines). Only ever touched from within the asyncio
# event loop thread (inside _ws_handler / _broadcast_state), so no lock
# is needed -- asyncio coroutines on one loop don't run concurrently.
_clients: set = set()


async def _ws_handler(websocket) -> None:
    remote = websocket.remote_address
    print(f"WebXR bridge: client connected from {remote}")
    _clients.add(websocket)
    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            _state.update(payload)
            await _broadcast_state()
    finally:
        _clients.discard(websocket)
        print(f"WebXR bridge: client disconnected ({remote})")


async def _broadcast_state() -> None:
    """Push the current left/right state to every connected client.

    This is what lets a second browser on another machine watch live
    telemetry -- the headset's own page goes dark the moment it enters
    the immersive AR session (WebXR replaces the 2D page), so the one
    device actually driving the data can no longer display it. Any other
    browser pointed at the same URL receives these broadcasts and renders
    the same hand-card UI as a passive viewer.
    """
    message = json.dumps({"left": _state.get("left"), "right": _state.get("right")})
    for client in list(_clients):
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass


def _run_ws_server(host: str, port: int, ssl_context: ssl.SSLContext) -> None:
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)

    async def _serve():
        async with websockets.serve(_ws_handler, host, port, ssl=ssl_context):
            await asyncio.Future()  # run forever, until loop is stopped/closed

    try:
        _ws_loop.run_until_complete(_serve())
    except asyncio.CancelledError:
        pass


def _run_http_server(host: str, port: int, ssl_context: ssl.SSLContext, directory: str) -> None:
    global _http_server

    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=directory, **kwargs
    )
    _http_server = http.server.ThreadingHTTPServer((host, port), handler)
    _http_server.socket = ssl_context.wrap_socket(_http_server.socket, server_side=True)
    _http_server.serve_forever()


def init(
    host: str = "0.0.0.0",
    ws_port: int = 8765,
    http_port: int = 8443,
    cert_file: str = "cert.pem",
    key_file: str = "key.pem",
    static_dir: str | None = None,
) -> None:
    """Start the HTTPS static server (serves index.html) and the WSS pose
    data server (receives controller pose/button JSON from the browser).

    static_dir defaults to the directory this file lives in, i.e. wherever
    index.html was placed alongside it.
    """
    global _ws_thread, _http_thread, _stop_event

    if static_dir is None:
        static_dir = str(Path(__file__).parent / "static")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    _stop_event = threading.Event()

    _ws_thread = threading.Thread(
        target=_run_ws_server, args=(host, ws_port, ssl_context), daemon=True
    )
    _ws_thread.start()

    # Separate SSLContext instance for the HTTP server -- SSLContext objects
    # aren't safely shareable across the wrap_socket call pattern used here
    # and the websockets library's own ssl handling.
    http_ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    http_ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    _http_thread = threading.Thread(
        target=_run_http_server, args=(host, http_port, http_ssl_context, static_dir), daemon=True
    )
    _http_thread.start()

    local_ip = _get_local_ip()
    print(f"WebXR bridge: open https://{local_ip}:{http_port}/ in the Quest Browser")
    print(f"WebXR bridge: pose data channel listening on wss://0.0.0.0:{ws_port}/")


def close() -> None:
    global _ws_loop, _http_server
    if _http_server is not None:
        _http_server.shutdown()
        _http_server = None
    if _ws_loop is not None:
        _ws_loop.call_soon_threadsafe(_ws_loop.stop)
        _ws_loop = None


def wait_for_data(timeout: float = 5.0, poll_interval: float = 0.05) -> bool:
    """Block until at least one real payload has arrived from the browser,
    or timeout elapses. Returns whether data arrived in time.

    Replaces a fixed warmup read count with an actual condition -- callers
    should check the return value and decide whether to proceed or bail.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _state.any_data_received():
            return True
        time.sleep(poll_interval)
    return _state.any_data_received()


def get_state(side: str) -> dict:
    """Return the full {pose7, trigger, grip} dict for one side in a
    single lock acquisition, instead of three separate getter calls.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return _state.get(side)


def is_tracked(side: str, max_age: float = DEFAULT_MAX_AGE_S) -> bool:
    """Whether `side` has received data within the last `max_age` seconds.

    There is no per-hand tracked/lost signal in the WebXR payload itself
    (unlike oculus_reader's poses dict) -- this approximates it from
    message recency instead.
    """
    updated_at = _state.last_updated(side)
    if updated_at is None:
        return False
    return (time.monotonic() - updated_at) <= max_age


def get_left_controller_pose() -> list[float]:
    return _state.get("left")["pose7"]


def get_right_controller_pose() -> list[float]:
    return _state.get("right")["pose7"]


def get_left_grip() -> float:
    return _state.get("left")["grip"]


def get_right_grip() -> float:
    return _state.get("right")["grip"]


def get_left_trigger() -> float:
    return _state.get("left")["trigger"]


def get_right_trigger() -> float:
    return _state.get("right")["trigger"]