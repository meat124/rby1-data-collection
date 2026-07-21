from __future__ import annotations

import asyncio
from fractions import Fraction
import importlib
import logging
import threading
import time
from typing import Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


def webrtc_dependencies_available() -> bool:
    return all(
        importlib.util.find_spec(module_name) is not None
        for module_name in ("aiohttp", "aiortc", "av")
    )


class HeadCameraVideoTrack:
    """aiortc-compatible video source backed by MultiRealsense latest frames."""

    kind = "video"

    def __init__(
        self,
        *,
        media_stream_track_cls,
        video_frame_cls,
        realsense: Any,
        camera_serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        stale_timeout_sec: float,
    ):
        self._video_frame_cls = video_frame_cls
        self.realsense = realsense
        self.camera_serial = camera_serial
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.stale_timeout_sec = float(stale_timeout_sec)
        self._started_at = time.monotonic()
        self._next_frame_at = self._started_at
        self._pts = 0
        self._last_sensor_timestamp: Optional[float] = None
        self._last_new_frame_at: Optional[float] = None
        self._last_rgb: Optional[np.ndarray] = None

        owner = self

        class _Track(media_stream_track_cls):
            kind = "video"

            async def recv(self):
                return await owner._recv()

        self._track = _Track()

    @property
    def track(self):
        return self._track

    @property
    def last_frame_age_sec(self) -> Optional[float]:
        if self._last_new_frame_at is None:
            return None
        return max(0.0, time.monotonic() - self._last_new_frame_at)

    def _latest_rgb(self) -> Optional[np.ndarray]:
        if self.realsense is None or self.camera_serial is None:
            return None

        try:
            frames = self.realsense.get_latest_frames()
            frame = frames.get(self.camera_serial)
            if frame is None:
                return None

            sensor_timestamp = float(frame.t)
            if sensor_timestamp != self._last_sensor_timestamp:
                self._last_sensor_timestamp = sensor_timestamp
                self._last_new_frame_at = time.monotonic()
                self._last_rgb = np.asarray(frame.color, dtype=np.uint8)
        except Exception as exc:
            logger.warning("[quest-video] Failed to read head camera frame: %s", exc)
            return None

        if self._last_new_frame_at is None:
            return None
        if time.monotonic() - self._last_new_frame_at > self.stale_timeout_sec:
            return None
        return self._last_rgb

    async def _recv(self):
        period = 1.0 / self.fps
        now = time.monotonic()
        if self._next_frame_at > now:
            await asyncio.sleep(self._next_frame_at - now)
        else:
            self._next_frame_at = now
        self._next_frame_at += period

        rgb = self._latest_rgb()
        if rgb is None:
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        frame = self._video_frame_cls.from_ndarray(rgb, format="rgb24")
        if frame.width != self.width or frame.height != self.height:
            frame = frame.reformat(width=self.width, height=self.height, format="yuv420p")
        frame.pts = self._pts
        frame.time_base = Fraction(1, self.fps)
        self._pts += 1
        return frame


class QuestVideoServer:
    """Single-Quest WebRTC signaling server with an H.264 head-camera track."""

    def __init__(
        self,
        *,
        realsense: Any,
        camera_name: str,
        camera_serial: Optional[str],
        bind_host: str,
        port: int,
        width: int,
        height: int,
        fps: int,
        codec: str,
        stale_timeout_sec: float,
    ):
        self.realsense = realsense
        self.camera_name = camera_name
        self.camera_serial = camera_serial
        self.bind_host = bind_host
        self.port = int(port)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.codec = codec.upper()
        self.stale_timeout_sec = float(stale_timeout_sec)

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner = None
        self._peer = None
        self._track_source: Optional[HeadCameraVideoTrack] = None
        self._started = threading.Event()
        self._start_error: Optional[str] = None
        self._peer_state = "disconnected"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._start_error is None

    def start(self, timeout_sec: float = 5.0) -> bool:
        if self.running:
            return True
        if not webrtc_dependencies_available():
            self._start_error = (
                "Optional WebRTC dependencies are missing. "
                "Install requirements-webrtc.txt to enable Quest video."
            )
            logger.error("[quest-video] %s", self._start_error)
            return False

        self._started.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._run, name="quest_video_server", daemon=True)
        self._thread.start()
        self._started.wait(timeout_sec)
        if not self._started.is_set():
            self._start_error = "WebRTC server startup timed out"
        if self._start_error:
            logger.error("[quest-video] %s", self._start_error)
            return False
        return True

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_async())
        except Exception as exc:
            self._start_error = str(exc)
            logger.exception("[quest-video] Server startup failed")
            self._started.set()
            self._loop.close()
            self._loop = None
            return

        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._shutdown_async())
            self._loop.close()

    async def _start_async(self) -> None:
        aiohttp_web = importlib.import_module("aiohttp.web")
        app = aiohttp_web.Application()
        app.router.add_post("/offer", self._handle_offer)
        app.router.add_get("/health", self._handle_health)
        self._runner = aiohttp_web.AppRunner(app)
        await self._runner.setup()
        site = aiohttp_web.TCPSite(self._runner, self.bind_host, self.port)
        await site.start()
        logger.info(
            "[quest-video] Signaling server listening on http://%s:%d (camera=%s)",
            self.bind_host,
            self.port,
            self.camera_name,
        )

    async def _close_peer(self) -> None:
        peer = self._peer
        self._peer = None
        self._track_source = None
        if peer is not None:
            await peer.close()
        self._peer_state = "disconnected"

    async def _handle_offer(self, request):
        aiohttp_web = importlib.import_module("aiohttp.web")
        aiortc = importlib.import_module("aiortc")
        try:
            payload = await request.json()
            if payload.get("type") != "offer" or not isinstance(payload.get("sdp"), str):
                raise ValueError("Expected JSON object with type='offer' and string sdp")

            await self._close_peer()
            peer = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
            self._peer = peer
            self._peer_state = "connecting"

            av = importlib.import_module("av")
            media = importlib.import_module("aiortc.mediastreams")
            self._track_source = HeadCameraVideoTrack(
                media_stream_track_cls=media.MediaStreamTrack,
                video_frame_cls=av.VideoFrame,
                realsense=self.realsense,
                camera_serial=self.camera_serial,
                width=self.width,
                height=self.height,
                fps=self.fps,
                stale_timeout_sec=self.stale_timeout_sec,
            )
            sender = peer.addTrack(self._track_source.track)
            transceiver = next(
                item for item in peer.getTransceivers() if item.sender == sender
            )

            codec_mime = f"video/{self.codec}"
            capabilities = aiortc.RTCRtpSender.getCapabilities("video")
            preferred_codecs = [
                codec for codec in capabilities.codecs
                if codec.mimeType.lower() == codec_mime.lower()
            ]
            if preferred_codecs:
                transceiver.setCodecPreferences(preferred_codecs)
            else:
                raise RuntimeError(f"Requested codec is unavailable: {codec_mime}")

            @peer.on("connectionstatechange")
            async def on_connectionstatechange():
                self._peer_state = peer.connectionState
                if peer.connectionState in {"failed", "closed"} and self._peer is peer:
                    await self._close_peer()

            offer = aiortc.RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
            await peer.setRemoteDescription(offer)
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            self._peer_state = peer.connectionState
            return aiohttp_web.json_response(
                {"sdp": peer.localDescription.sdp, "type": peer.localDescription.type}
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("[quest-video] Offer rejected: %s", exc)
            return aiohttp_web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            logger.exception("[quest-video] Offer handling failed")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    async def _handle_health(self, request):
        aiohttp_web = importlib.import_module("aiohttp.web")
        return aiohttp_web.json_response(self.status_snapshot())

    async def _shutdown_async(self) -> None:
        await self._close_peer()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def status_snapshot(self) -> dict[str, object]:
        frame_age = None
        if self._track_source is not None:
            frame_age = self._track_source.last_frame_age_sec
        camera_running = bool(
            self.realsense is not None and getattr(self.realsense, "running", False)
        )
        return {
            "enabled": True,
            "running": self.running,
            "error": self._start_error,
            "peer_state": self._peer_state,
            "camera_name": self.camera_name,
            "camera_serial": self.camera_serial,
            "camera_running": camera_running,
            "last_frame_age_sec": frame_age,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def stop(self, timeout_sec: float = 5.0) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout_sec)
        self._thread = None
        self._loop = None
