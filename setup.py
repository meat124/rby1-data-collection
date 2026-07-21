import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
import threading
import os
import yaml
from vr_control_state import VRControlState


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
try:
    with open(_CONFIG_PATH, encoding='utf-8') as _f:
        _config = yaml.safe_load(_f) or {}
except Exception:
    _config = {}

class Settings:
    dt: float = 0.1
    hand_offset: float = np.array([0.0, 0.0, 0.0])

    T_hand_offset = np.identity(4)
    T_hand_offset[0:3, 3] = hand_offset

    vr_control_local_port: int = 5005
    vr_control_meta_quest_port: int = 6000

    mobile_linear_acceleration_gain: float = -0.15
    mobile_angular_acceleration_gain: float = 0.15
    mobile_linear_damping_gain: float = 0.3
    mobile_angular_damping_gain: float = 0.3

    rec_fps: int = _config.get('rec_fps', 15)

    head_tracking: dict[str, Any] = {
        "stale_timeout_sec": 0.5,
        "joint_limit_margin_deg": 1.0,
        "max_velocity_deg_s": 90.0,
        "max_acceleration_deg_s2": 180.0,
        "yaw_scale": 1.0,
        "pitch_scale": 1.0,
        **(_config.get("head_tracking", {}) or {}),
    }
    quest_video: dict[str, Any] = {
        "enabled": True,
        "camera": "head",
        "bind_host": "0.0.0.0",
        "advertise_host": None,
        "port": 8080,
        "width": 640,
        "height": 480,
        "fps": 30,
        "codec": "H264",
        "stale_frame_timeout_sec": 0.5,
        **(_config.get("quest_video", {}) or {}),
    }

    MAX_POINT: int = None

    # Initial poses in degrees
    torso_init_pose: tuple = (0.0, 20.0, -40.0, 35.0, 0.0, 0.0)  #0 20 -40 20 0 0 -> 0 20 -40 40 0 0
    right_arm_init_pose: tuple = (-24.0, -60.0, 10.0, -120.0, -60.0, 85.0, 0.0)  #0 -15 0 -120 0 70 0 -> -24 -20 10 -120 -20 85 0
    left_arm_init_pose: tuple = (-24.0, 60.0, -10.0, -120.0, 60.0, 85.0, 0.0)    #0 15 0 -120 0 70 0 -> -24 20 -10 -120 20 85 0
    torso_head_init_pose: tuple = (0.0, 40.0)
    bimanual_head_init_pose: tuple = (0.0, 40.0)
    
    shoulder_pitch_angle: float = 70.0
    shoulder_roll_angle: float = 30.0
    elbow_angle: float = -100.0
    wrist_angle: float = -70.0

    right_arm_midpoint1 = np.deg2rad([shoulder_pitch_angle, -shoulder_roll_angle, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])
    left_arm_midpoint1 = np.deg2rad([shoulder_pitch_angle, shoulder_roll_angle, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])

    right_arm_midpoint2 = np.deg2rad([0.0, -15.0, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])
    left_arm_midpoint2 = np.deg2rad([0.0, 15.0, 0.0, elbow_angle, 0.0, wrist_angle, 0.0])

    body_init_pose: float = np.deg2rad(torso_init_pose + right_arm_init_pose + left_arm_init_pose)


@dataclass(slots=True)
class RecordingPreflightState:
    robot_connected: bool = False
    control_manager_ready: bool = False
    robot_initialized: bool = False
    cameras_configured: bool = False
    cameras_ready: bool = False
    gripper_ready: bool = False
    storage_ready: bool = False
    idle_ready: bool = False
    ready: bool = False
    preview_available: bool = False
    last_checked_at: float = 0.0
    camera_message: str = "Waiting for camera checks."
    camera_diagnostics: Optional[dict[str, Any]] = None
    message: str = "Waiting for backend startup."


@dataclass(slots=True)
class RecordingSessionState:
    status: str = "idle"
    message: str = "Idle"
    current_episode_path: Optional[str] = None
    last_saved_path: Optional[str] = None
    last_discarded_path: Optional[str] = None
    recording_started_at: Optional[float] = None
    last_transition_at: float = 0.0
    pending_action: Optional[str] = None
    last_error: Optional[str] = None

class SystemContext:
    robot_model: Optional[Any] = None
    vr_state: VRControlState = VRControlState()
    # H5 writer and recording stop-event stored here so other threads/handlers can access them
    h5_writer: Optional[Any] = None
    rec_stop_event: Optional[threading.Event] = None
    cleanup_functions: list = []  # Functions to call on cleanup
    
    # Camera object for stopping frames when recording stops
    realsense: Optional[object] = None
    camera_warmup_done: bool = False
    camera_context: Optional[dict[str, Any]] = None
    head_tracker: Optional[Any] = None
    head_tracking_status: dict[str, Any] = {}
    quest_video_server: Optional[Any] = None
    quest_video_status: dict[str, Any] = {}

    recording_state: RecordingSessionState = RecordingSessionState()
    preflight_state: RecordingPreflightState = RecordingPreflightState()
    recording_manager: Optional[Any] = None

    sensor_lock: threading.Lock = threading.Lock()
    ft_sensor_right_force: np.ndarray = np.zeros(3, dtype=float)
    ft_sensor_right_torque: np.ndarray = np.zeros(3, dtype=float)
    ft_sensor_left_force: np.ndarray = np.zeros(3, dtype=float)
    ft_sensor_left_torque: np.ndarray = np.zeros(3, dtype=float)
    gripper_state: Optional[np.ndarray] = None
    gripper_target: Optional[np.ndarray] = None
    storage_root: Optional[str] = None
    
    # Lock for thread-safe recording state transitions
    # (protects demo_recording_status and start_new_recording_requested shared between main loop and VR button handler)
    recording_lock: threading.Lock = threading.Lock()
    # Demo recording state management
    demo_recording_status: str = "idle"  # "idle", "recording", "stopping"
    start_new_recording_requested: bool = False  # Flag to request starting a new recording in main loop