import html
from typing import Any

import zmq


class VRDashboardClient:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556", timeout_ms: int = 2000):
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

        try:
            socket.connect(self.endpoint)
            socket.send_json(payload)
            return socket.recv_json()
        finally:
            socket.close()

    def get_status(self, include_preview: bool = False) -> dict[str, Any]:
        return self._request({"command": "get_status", "include_preview": include_preview})

    def start_recording(self) -> dict[str, Any]:
        return self._request({"command": "start_recording"})

    def stop_recording(self) -> dict[str, Any]:
        return self._request({"command": "stop_recording"})

    def toggle_recording(self) -> dict[str, Any]:
        return self._request({"command": "toggle_recording"})

    def discard_last_demo(self) -> dict[str, Any]:
        return self._request({"command": "discard_last_demo"})

    def init_torso_mode(self) -> dict[str, Any]:
        return self._request({"command": "init_torso_mode"})

    def init_bimanual_mode(self) -> dict[str, Any]:
        return self._request({"command": "init_bimanual_mode"})

    def check_cameras(self) -> dict[str, Any]:
        return self._request({"command": "check_cameras", "include_preview": True})


STATUS_COLORS = {
    "idle": "#556b75",
    "recording": "#c44536",
    "saving": "#d38b18",
    "saved": "#2d8659",
    "discarded": "#6c5ce7",
    "error": "#9b2226",
}


def render_status_badge(status: str, message: str) -> str:
    color = STATUS_COLORS.get(status, "#556b75")
    safe_status = html.escape(status.upper())
    safe_message = html.escape(message or "")
    return (
        f"<div style='padding:10px 14px;border-radius:12px;background:{color};color:white;'>"
        f"<div style='font-size:12px;opacity:0.8;'>STATUS</div>"
        f"<div style='font-size:22px;font-weight:700;'>{safe_status}</div>"
        f"<div style='font-size:13px;margin-top:4px;'>{safe_message}</div>"
        "</div>"
    )


def render_preflight_html(preflight: dict[str, Any]) -> str:
    rows = []
    labels = {
        "robot_connected": "Robot",
        "control_manager_ready": "Control manager",
        "robot_initialized": "Ready pose",
        "cameras_configured": "Camera config",
        "cameras_ready": "Camera stream",
        "gripper_ready": "Gripper",
        "storage_ready": "Storage",
        "idle_ready": "Recording idle",
    }
    for key, label in labels.items():
        ok = bool(preflight.get(key, False))
        color = "#1b7f4a" if ok else "#a63d40"
        text = "READY" if ok else "WAIT"
        rows.append(
            f"<tr><td style='padding:4px 8px'>{html.escape(label)}</td>"
            f"<td style='padding:4px 8px;color:{color};font-weight:600'>{text}</td></tr>"
        )

    message = html.escape(preflight.get("message", ""))
    return (
        "<div style='padding:10px 14px;border:1px solid #d9e2ec;border-radius:12px;background:#f8fbfc;'>"
        "<div style='font-size:12px;color:#486581;margin-bottom:6px;'>PREFLIGHT</div>"
        f"<div style='font-size:13px;margin-bottom:8px;color:#102a43;'>{message}</div>"
        "<table style='border-collapse:collapse;font-size:13px;'>"
        f"{''.join(rows)}"
        "</table>"
        f"<div style='font-size:12px;color:#486581;margin-top:8px;'>{html.escape(preflight.get('camera_message', ''))}</div>"
        "</div>"
    )


def render_camera_html(camera_diagnostics: dict[str, Any]) -> str:
    if not camera_diagnostics:
        return (
            "<div style='padding:10px 14px;border:1px solid #d9e2ec;border-radius:12px;background:#ffffff;'>"
            "<div style='font-size:12px;color:#486581;margin-bottom:6px;'>CAMERAS</div>"
            "<div style='font-size:13px;color:#102a43;'>Camera diagnostics unavailable.</div>"
            "</div>"
        )

    def _fmt(names: Any) -> str:
        if not names:
            return "-"
        return ", ".join(html.escape(str(name)) for name in names)

    preview_ok = bool(camera_diagnostics.get("preview_available", False))
    preview_color = "#1b7f4a" if preview_ok else "#a63d40"
    preview_text = "AVAILABLE" if preview_ok else "UNAVAILABLE"

    return (
        "<div style='padding:10px 14px;border:1px solid #d9e2ec;border-radius:12px;background:#ffffff;'>"
        "<div style='font-size:12px;color:#486581;margin-bottom:6px;'>CAMERAS</div>"
        f"<div style='font-size:13px;margin-bottom:8px;color:#102a43;'>{html.escape(camera_diagnostics.get('message', ''))}</div>"
        f"<div style='font-size:13px;'><b>Configured:</b> {_fmt(camera_diagnostics.get('configured_names'))}</div>"
        f"<div style='font-size:13px;'><b>Started:</b> {_fmt(camera_diagnostics.get('started_names'))}</div>"
        f"<div style='font-size:13px;'><b>Live frames:</b> {_fmt(camera_diagnostics.get('live_names'))}</div>"
        f"<div style='font-size:13px;'><b>Synced frames:</b> {_fmt(camera_diagnostics.get('synced_names'))}</div>"
        f"<div style='font-size:13px;'><b>Missing live:</b> {_fmt(camera_diagnostics.get('missing_live_names'))}</div>"
        f"<div style='font-size:13px;'><b>Preview source:</b> {html.escape(str(camera_diagnostics.get('preview_source_name') or '-'))}</div>"
        f"<div style='font-size:13px;color:{preview_color};font-weight:600;'><b>Preview:</b> {preview_text}</div>"
        "</div>"
    )


def render_paths_html(snapshot: dict[str, Any]) -> str:
    current_path = html.escape(snapshot.get("current_episode_path") or "-")
    saved_path = html.escape(snapshot.get("last_saved_path") or "-")
    discarded_path = html.escape(snapshot.get("last_discarded_path") or "-")
    elapsed = snapshot.get("elapsed_seconds")
    elapsed_text = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "-"
    return (
        "<div style='padding:10px 14px;border:1px solid #d9e2ec;border-radius:12px;background:#ffffff;'>"
        "<div style='font-size:12px;color:#486581;margin-bottom:6px;'>SESSION</div>"
        f"<div style='font-size:13px;'><b>Current:</b> {current_path}</div>"
        f"<div style='font-size:13px;'><b>Last saved:</b> {saved_path}</div>"
        f"<div style='font-size:13px;'><b>Last discarded:</b> {discarded_path}</div>"
        f"<div style='font-size:13px;'><b>Elapsed:</b> {elapsed_text}</div>"
        "</div>"
    )