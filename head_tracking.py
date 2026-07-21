from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class HeadTrackingConfig:
    stale_timeout_sec: float = 0.5
    joint_limit_margin_rad: float = np.deg2rad(1.0)
    max_velocity_rad_s: float = np.deg2rad(90.0)
    max_acceleration_rad_s2: float = np.deg2rad(180.0)
    yaw_scale: float = 1.0
    pitch_scale: float = 1.0


def extract_yaw_pitch(rotation: np.ndarray) -> np.ndarray:
    """Return robot-frame yaw/pitch while intentionally discarding roll.

    The robot head points along +X, pans about +Z, and tilts about +Y. Using
    the rotated forward vector makes the result independent of HMD roll.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")

    forward = rotation[:, 0]
    horizontal = float(np.hypot(forward[0], forward[1]))
    yaw = float(np.arctan2(forward[1], forward[0]))
    pitch = float(np.arctan2(-forward[2], horizontal))
    return np.array([yaw, pitch], dtype=float)


class HeadTracker:
    """Stateful relative HMD-to-RBY1 two-joint head mapper."""

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        config: HeadTrackingConfig,
    ):
        lower = np.asarray(lower_limits, dtype=float)
        upper = np.asarray(upper_limits, dtype=float)
        if lower.shape != (2,) or upper.shape != (2,):
            raise ValueError("RBY1 head tracking requires exactly two joint limits")

        margin = float(config.joint_limit_margin_rad)
        self.lower_limits = lower + margin
        self.upper_limits = upper - margin
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("head joint limit margin leaves no valid motion range")

        self.config = config
        self._reference_rotation: Optional[np.ndarray] = None
        self._reference_joints: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._velocity = np.zeros(2, dtype=float)
        self._last_update_at: Optional[float] = None
        self._last_packet_at: Optional[float] = None
        self._recenter_pending = True
        self._stale = True

    def request_recenter(self) -> None:
        self._recenter_pending = True

    @property
    def target(self) -> Optional[np.ndarray]:
        return None if self._target is None else self._target.copy()

    @property
    def calibrated(self) -> bool:
        return self._reference_rotation is not None and not self._recenter_pending

    @property
    def stale(self) -> bool:
        return self._stale

    def _calibrate(self, hmd_rotation: np.ndarray, actual_joints: np.ndarray, now: float) -> np.ndarray:
        self._reference_rotation = np.asarray(hmd_rotation, dtype=float).copy()
        self._reference_joints = np.clip(actual_joints, self.lower_limits, self.upper_limits)
        self._target = self._reference_joints.copy()
        self._velocity.fill(0.0)
        self._last_update_at = now
        self._recenter_pending = False
        self._stale = False
        return self._target.copy()

    def update(
        self,
        *,
        hmd_rotation: Optional[np.ndarray],
        actual_joints: np.ndarray,
        packet_received_at: Optional[float],
        now: float,
    ) -> np.ndarray:
        actual = np.asarray(actual_joints, dtype=float)
        if actual.shape != (2,):
            raise ValueError(f"actual_joints must have shape (2,), got {actual.shape}")

        if self._target is None:
            self._target = np.clip(actual, self.lower_limits, self.upper_limits)

        packet_is_fresh = (
            hmd_rotation is not None
            and packet_received_at is not None
            and 0.0 <= now - packet_received_at <= self.config.stale_timeout_sec
        )
        if not packet_is_fresh:
            self._stale = True
            self._velocity.fill(0.0)
            self._last_update_at = now
            return self._target.copy()

        recovering_from_stale = self._stale
        if self._recenter_pending or recovering_from_stale or self._reference_rotation is None:
            target = self._calibrate(np.asarray(hmd_rotation, dtype=float), actual, now)
            self._last_packet_at = packet_received_at
            return target

        relative_rotation = self._reference_rotation.T @ np.asarray(hmd_rotation, dtype=float)
        yaw_pitch = extract_yaw_pitch(relative_rotation)
        yaw_pitch *= np.array([self.config.yaw_scale, self.config.pitch_scale], dtype=float)
        desired = np.clip(
            self._reference_joints + yaw_pitch,
            self.lower_limits,
            self.upper_limits,
        )

        dt = max(1e-6, now - (self._last_update_at if self._last_update_at is not None else now))
        error = desired - self._target
        desired_velocity = np.clip(
            error / dt,
            -self.config.max_velocity_rad_s,
            self.config.max_velocity_rad_s,
        )
        max_velocity_change = self.config.max_acceleration_rad_s2 * dt
        self._velocity += np.clip(
            desired_velocity - self._velocity,
            -max_velocity_change,
            max_velocity_change,
        )
        self._velocity = np.clip(
            self._velocity,
            -self.config.max_velocity_rad_s,
            self.config.max_velocity_rad_s,
        )

        step = self._velocity * dt
        reached = np.abs(step) >= np.abs(error)
        self._target += step
        self._target[reached] = desired[reached]
        self._velocity[reached] = 0.0
        self._target = np.clip(self._target, self.lower_limits, self.upper_limits)
        self._last_update_at = now
        self._last_packet_at = packet_received_at
        self._stale = False
        return self._target.copy()

    def status_snapshot(self, now: Optional[float] = None) -> dict[str, object]:
        packet_age = None
        if now is not None and self._last_packet_at is not None:
            packet_age = max(0.0, now - self._last_packet_at)
        return {
            "calibrated": self.calibrated,
            "stale": self.stale,
            "recenter_pending": self._recenter_pending,
            "target": None if self._target is None else self._target.tolist(),
            "packet_age_sec": packet_age,
        }
