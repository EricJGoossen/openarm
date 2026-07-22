"""Configuration for WebXR-driven teleop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk upward from `start` until a directory containing .git is
    found. Falls back to `start` itself if no repo root is found (e.g.
    if this package is ever installed outside a git checkout), so a
    missing sentinel doesn't break import.

    This makes cert-path defaults independent of the current working
    directory a script happens to be launched from -- `uv run` and
    plain `python` invoked from different directories were resolving
    the old relative "certs/cert.pem" default inconsistently.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.resolve()


def _default_cert_file() -> Path:
    return _find_repo_root(Path(__file__)) / "certs" / "cert.pem"


def _default_key_file() -> Path:
    return _find_repo_root(Path(__file__)) / "certs" / "key.pem"

@dataclass
class ClutchConfig:
    """Engage/disengage and gripper-toggle thresholds for one hand.

    grip_engage_threshold: measured grip range under a real squeeze is
    ~0.04-0.09, so this sits just below that band.

    trigger_press/release_threshold: hysteresis band, so jitter near a
    single midpoint doesn't register as press-release-press for one
    real squeeze.

    max_step_m: per-tick Cartesian step cap on the EE target.
    """

    grip_engage_threshold: float = 0.05
    trigger_press_threshold: float = 0.7
    trigger_release_threshold: float = 0.3
    position_scale: float = 1.0
    max_step_m: float = 0.05


@dataclass
class TimingConfig:
    """Control loop and render timing."""

    target_hz: float = 250.0
    rendered_fps: float = 30.0
    wait_for_data_timeout: float = 5.0  # seconds to wait for the first pose

    @property
    def period(self) -> float:
        return 1.0 / self.target_hz


@dataclass
class WebXRBridgeConfig:
    """Connection settings for the WebXR pose bridge server.

    cert_file/key_file default to <repo_root>/certs/{cert,key}.pem --
    gitignored and generated per-machine (see openarm/README.md for the
    openssl command). Resolved from this file's location, not the
    current working directory, so it doesn't matter where the demo
    script is launched from.
    """

    host: str = "0.0.0.0"
    ws_port: int = 8765
    http_port: int = 8443
    cert_file: Path = field(default_factory=_default_cert_file)
    key_file: Path = field(default_factory=_default_key_file)

@dataclass
class ControlConfig:
    """Teleop-only control limits. These override the arm's ArmConfig
    defults for teleop."""
    max_cartesian_speed: float | None = 1.0
    max_joint_step: float | None = None
    safety_mode: str = "allow"


@dataclass
class TeleopConfig:
    """Full teleop configuration."""

    left_clutch: ClutchConfig = field(default_factory=ClutchConfig)
    right_clutch: ClutchConfig = field(default_factory=ClutchConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    bridge: WebXRBridgeConfig = field(default_factory=WebXRBridgeConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    debug: bool = False

    @classmethod
    def default(cls) -> "TeleopConfig":
        """Default configuration for the WebXR sim demo."""
        return cls()
