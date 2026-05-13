import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from h5_writer import H5Writer
from setup import RecordingPreflightState, RecordingSessionState, SystemContext
from utils import get_next_h5_path


logger = logging.getLogger(__name__)


class RecordingManager:
    def __init__(
        self,
        root_path: str,
        logger_factory: Callable[[Any, H5Writer, Any], Any],
        flush_every: int = 60,
        flush_secs: float = 1.0,
    ):
        self.root_path = root_path
        self.logger_factory = logger_factory
        self.flush_every = flush_every
        self.flush_secs = flush_secs

        os.makedirs(self.root_path, exist_ok=True)
        SystemContext.storage_root = self.root_path
        SystemContext.recording_state = RecordingSessionState(last_transition_at=time.time())
        SystemContext.preflight_state = RecordingPreflightState()

    def _legacy_status(self, status: str) -> str:
        if status == "recording":
            return "recording"
        if status == "saving":
            return "stopping"
        return "idle"

    def _update_state(self, **kwargs):
        with SystemContext.recording_lock:
            state = SystemContext.recording_state
            for key, value in kwargs.items():
                setattr(state, key, value)
            state.last_transition_at = time.time()
            SystemContext.demo_recording_status = self._legacy_status(state.status)

    def _camera_name(self, serial: str) -> str:
        camera_context = getattr(SystemContext, "camera_context", None) or {}
        serial_to_name = camera_context.get("serial_to_cam_name", {}) or {}
        return str(serial_to_name.get(serial, serial))

    def _collect_camera_diagnostics(self) -> dict[str, Any]:
        camera_context = getattr(SystemContext, "camera_context", None) or {}
        configured_serials = list(camera_context.get("camera_serials", []) or [])
        primary_serial = camera_context.get("primary_serial")
        primary_name = camera_context.get("primary_cam_name")

        configured_names = [self._camera_name(serial) for serial in configured_serials]
        diagnostics = {
            "configured_names": configured_names,
            "started_names": [],
            "live_names": [],
            "synced_names": [],
            "missing_started_names": configured_names.copy(),
            "missing_live_names": configured_names.copy(),
            "missing_synced_names": configured_names.copy(),
            "preview_source_name": str(primary_name or self._camera_name(primary_serial)) if (primary_name or primary_serial) else None,
            "preview_available": False,
            "all_started": False,
            "all_live": False,
            "all_synced": False,
            "message": "Camera pipelines are not running yet.",
        }

        if not configured_serials:
            diagnostics.update(
                {
                    "all_started": True,
                    "all_live": True,
                    "all_synced": True,
                    "message": "No cameras configured in config.yaml.",
                }
            )
            return diagnostics

        realsense = SystemContext.realsense
        if realsense is None or not getattr(realsense, "running", False):
            return diagnostics

        raw = {}
        if hasattr(realsense, "get_camera_diagnostics"):
            try:
                raw = realsense.get_camera_diagnostics()
            except Exception:
                raw = {}

        started_serials = list(raw.get("started_serials", []))
        latest_serials = list(raw.get("latest_frame_serials", []))
        synced_serials = list(raw.get("synced_frame_serials", []))

        started_names = [self._camera_name(serial) for serial in started_serials]
        live_names = [self._camera_name(serial) for serial in latest_serials]
        synced_names = [self._camera_name(serial) for serial in synced_serials]
        missing_started_names = [self._camera_name(serial) for serial in raw.get("missing_started_serials", [])]
        missing_live_names = [self._camera_name(serial) for serial in raw.get("missing_latest_serials", [])]
        missing_synced_names = [self._camera_name(serial) for serial in raw.get("missing_synced_serials", [])]

        preview_source_serial = primary_serial if primary_serial in latest_serials else None

        if raw.get("all_have_latest", False):
            camera_message = "All configured cameras are streaming live frames."
        elif live_names:
            if primary_serial is not None and primary_serial not in latest_serials:
                camera_message = (
                    f"Preview camera {self._camera_name(primary_serial)} is not live. "
                    f"Live frames available from {', '.join(live_names)}. "
                    f"Missing: {', '.join(missing_live_names)}"
                )
            else:
                camera_message = (
                    f"Live frames available from {', '.join(live_names)}. "
                    f"Missing: {', '.join(missing_live_names)}"
                )
        elif started_names:
            camera_message = (
                f"Cameras started but no frames received yet: {', '.join(started_names)}"
            )
        else:
            camera_message = "No configured camera stream has started yet."

        diagnostics.update(
            {
                "started_names": started_names,
                "live_names": live_names,
                "synced_names": synced_names,
                "missing_started_names": missing_started_names,
                "missing_live_names": missing_live_names,
                "missing_synced_names": missing_synced_names,
                "preview_source_name": str(primary_name or self._camera_name(primary_serial)) if (primary_name or primary_serial) else None,
                "preview_available": preview_source_serial is not None,
                "all_started": bool(raw.get("all_started", False)),
                "all_live": bool(raw.get("all_have_latest", False)),
                "all_synced": bool(raw.get("all_have_synced", False)),
                "message": camera_message,
            }
        )
        return diagnostics

    def request_start_recording(self) -> bool:
        with SystemContext.recording_lock:
            state = SystemContext.recording_state
            if state.status in {"recording", "saving"} or state.pending_action == "start":
                return False
            if not SystemContext.preflight_state.ready:
                state.message = f"Preflight not ready: {SystemContext.preflight_state.message}"
                state.last_error = SystemContext.preflight_state.message
                return False
            state.pending_action = "start"
            state.message = "Start requested"
            SystemContext.start_new_recording_requested = True
            return True

    def request_stop_and_save(self) -> bool:
        with SystemContext.recording_lock:
            state = SystemContext.recording_state
            if state.status != "recording" or state.pending_action == "stop":
                return False
            state.pending_action = "stop"
            state.message = "Stop requested"
            SystemContext.start_new_recording_requested = False
            return True

    def request_toggle_recording(self) -> bool:
        with SystemContext.recording_lock:
            status = SystemContext.recording_state.status
        if status == "recording":
            return self.request_stop_and_save()
        return self.request_start_recording()

    def request_discard_last_demo(self) -> bool:
        with SystemContext.recording_lock:
            state = SystemContext.recording_state
            if state.status in {"recording", "saving"} or not state.last_saved_path:
                return False
            state.pending_action = "discard"
            state.message = "Discard requested"
            return True

    def process_pending_actions(self, gripper: Any, robot: Any) -> bool:
        with SystemContext.recording_lock:
            action = SystemContext.recording_state.pending_action
            SystemContext.recording_state.pending_action = None

        if action == "start":
            self._start_recording(gripper, robot)
            return True
        if action == "stop":
            self._stop_recording()
            return True
        if action == "discard":
            self._discard_last_demo()
            return True
        return False

    def _start_recording(self, gripper: Any, robot: Any):
        output_path = get_next_h5_path(self.root_path)
        writer = H5Writer(path=output_path, flush_every=self.flush_every, flush_secs=self.flush_secs).start()
        stop_event = self.logger_factory(gripper, writer, robot)

        SystemContext.h5_writer = writer
        SystemContext.rec_stop_event = stop_event
        self._update_state(
            status="recording",
            message=f"Recording to {output_path}",
            current_episode_path=output_path,
            recording_started_at=time.time(),
            last_error=None,
        )
        logger.info("Recording started: %s", output_path)

    def _stop_recording(self):
        writer = SystemContext.h5_writer
        stop_event = SystemContext.rec_stop_event
        current_path = SystemContext.recording_state.current_episode_path

        self._update_state(status="saving", message="Saving current recording")

        if stop_event is not None:
            stop_event.set()
        if writer is not None:
            writer.stop()

        SystemContext.h5_writer = None
        SystemContext.rec_stop_event = None
        self._update_state(
            status="saved",
            message=f"Saved recording to {current_path}",
            current_episode_path=None,
            last_saved_path=current_path,
            recording_started_at=None,
        )
        logger.info("Recording saved: %s", current_path)

    def _discard_last_demo(self):
        saved_path = SystemContext.recording_state.last_saved_path
        if not saved_path:
            return

        saved_file = Path(saved_path)
        if saved_file.parent.is_dir() and saved_file.parent.name.startswith("episode_"):
            shutil.rmtree(saved_file.parent, ignore_errors=True)
        elif saved_file.exists():
            saved_file.unlink(missing_ok=True)

        self._update_state(
            status="discarded",
            message=f"Discarded {saved_path}",
            last_saved_path=None,
            last_discarded_path=saved_path,
        )
        logger.info("Discarded saved demo: %s", saved_path)

    def shutdown(self):
        if SystemContext.recording_state.status == "recording":
            self._stop_recording()

    def refresh_preflight(self, robot: Any = None, gripper: Any = None):
        root_path = Path(self.root_path)
        root_path.mkdir(parents=True, exist_ok=True)

        robot_connected = False
        control_manager_ready = False
        if robot is not None:
            try:
                robot_connected = bool(robot.is_connected())
            except Exception:
                robot_connected = True
            try:
                control_manager_state = robot.get_control_manager_state()
                manager_state = getattr(getattr(control_manager_state, "state", None), "name", "")
                control_state = getattr(getattr(control_manager_state, "control_state", None), "name", "")
                control_manager_ready = (
                    manager_state == "Enabled"
                    and control_state in {"Idle", "Executing", "Switching"}
                )
            except Exception:
                try:
                    control_manager_ready = bool(robot.wait_for_control_ready(0))
                except Exception:
                    control_manager_ready = False

        camera_diagnostics = self._collect_camera_diagnostics()
        cameras_configured = bool(camera_diagnostics["configured_names"])
        cameras_ready = bool(camera_diagnostics["all_live"])
        gripper_ready = gripper is None or hasattr(gripper, "get_target")
        storage_ready = os.access(root_path, os.W_OK)
        robot_initialized = bool(SystemContext.vr_state.is_initialized)

        with SystemContext.recording_lock:
            idle_ready = SystemContext.recording_state.status not in {"recording", "saving"}

        missing = []
        if not robot_connected:
            missing.append("robot connection")
        if not control_manager_ready:
            missing.append("control manager")
        if not robot_initialized:
            missing.append("ready pose initialization")
        if cameras_configured and not cameras_ready:
            missing.append(camera_diagnostics["message"])
        if not gripper_ready:
            missing.append("gripper")
        if not storage_ready:
            missing.append("storage")
        if not idle_ready:
            missing.append("recording stop/save")

        ready = not missing
        SystemContext.preflight_state = RecordingPreflightState(
            robot_connected=robot_connected,
            control_manager_ready=control_manager_ready,
            robot_initialized=robot_initialized,
            cameras_configured=cameras_configured,
            cameras_ready=cameras_ready,
            gripper_ready=gripper_ready,
            storage_ready=storage_ready,
            idle_ready=idle_ready,
            ready=ready,
            preview_available=bool(camera_diagnostics["preview_available"]),
            last_checked_at=time.time(),
            camera_message=str(camera_diagnostics["message"]),
            camera_diagnostics=camera_diagnostics,
            message="Ready to record." if ready else f"Waiting on: {'; '.join(missing)}",
        )

    def get_status_snapshot(self) -> dict[str, Any]:
        with SystemContext.recording_lock:
            recording = SystemContext.recording_state
            preflight = SystemContext.preflight_state
            elapsed_seconds = None
            if recording.recording_started_at is not None:
                elapsed_seconds = max(0.0, time.time() - recording.recording_started_at)
            return {
                "status": recording.status,
                "message": recording.message,
                "current_episode_path": recording.current_episode_path,
                "last_saved_path": recording.last_saved_path,
                "last_discarded_path": recording.last_discarded_path,
                "pending_action": recording.pending_action,
                "last_error": recording.last_error,
                "elapsed_seconds": elapsed_seconds,
                "preview_available": preflight.preview_available,
                "camera_diagnostics": dict(preflight.camera_diagnostics or {}),
                "preflight": {
                    "ready": preflight.ready,
                    "message": preflight.message,
                    "robot_connected": preflight.robot_connected,
                    "control_manager_ready": preflight.control_manager_ready,
                    "robot_initialized": preflight.robot_initialized,
                    "cameras_configured": preflight.cameras_configured,
                    "cameras_ready": preflight.cameras_ready,
                    "gripper_ready": preflight.gripper_ready,
                    "storage_ready": preflight.storage_ready,
                    "idle_ready": preflight.idle_ready,
                    "preview_available": preflight.preview_available,
                    "camera_message": preflight.camera_message,
                    "last_checked_at": preflight.last_checked_at,
                },
            }