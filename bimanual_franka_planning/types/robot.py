from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


@dataclass
class CameraConfig:
    link_name: str
    width: int
    height: int
    fov: float
    near: float
    far: float


@dataclass
class RobotConfig:
    urdf_path: str
    joint_names: list[str]
    camera: CameraConfig
    # Per-joint kinematic limits used by time-optimal trajectory parameterization.
    # Shape (len(joint_names),), strictly positive. ``None`` means "not supplied
    # by this robot" — callers must pass explicit limits instead.
    max_velocity: Optional["np.ndarray"] = field(default=None)
    max_acceleration: Optional["np.ndarray"] = field(default=None)


@dataclass(frozen=True)
class ChainConfig:
    """Configuration for a kinematic chain used by TRAC-IK."""

    base_link: str
    ee_link: str
    num_joints: int
    urdf_path: str

    def with_urdf_path(self, urdf_path: str) -> "ChainConfig":
        """Return a copy with a different URDF path."""
        return replace(self, urdf_path=urdf_path)
