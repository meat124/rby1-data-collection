import asyncio
from pathlib import Path
import socket
import unittest
from unittest import mock

import numpy as np

from head_tracking import HeadTracker, HeadTrackingConfig, extract_yaw_pitch
from quest_streaming import (
    HeadCameraVideoTrack,
    QuestVideoServer,
    webrtc_dependencies_available,
)


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class HeadTrackerTests(unittest.TestCase):
    def make_tracker(self):
        return HeadTracker(
            lower_limits=np.array([-0.523, -0.35]),
            upper_limits=np.array([0.523, 1.57]),
            config=HeadTrackingConfig(
                stale_timeout_sec=0.5,
                joint_limit_margin_rad=0.0,
                max_velocity_rad_s=100.0,
                max_acceleration_rad_s2=1000.0,
            ),
        )

    def test_extract_yaw_pitch_ignores_roll(self):
        yaw = 0.22
        pitch = -0.13
        rotation = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(0.8)
        np.testing.assert_allclose(extract_yaw_pitch(rotation), [yaw, pitch], atol=1e-8)

    def test_relative_hmd_rotation_moves_only_two_head_joints(self):
        tracker = self.make_tracker()
        actual = np.array([0.0, 0.7])
        np.testing.assert_allclose(
            tracker.update(
                hmd_rotation=np.eye(3),
                actual_joints=actual,
                packet_received_at=0.0,
                now=0.0,
            ),
            actual,
        )

        target = tracker.update(
            hmd_rotation=rotation_z(0.2) @ rotation_y(-0.1),
            actual_joints=actual,
            packet_received_at=0.1,
            now=0.1,
        )
        np.testing.assert_allclose(target, [0.2, 0.6], atol=1e-8)

    def test_targets_are_clamped_to_robot_limits(self):
        tracker = self.make_tracker()
        actual = np.array([0.4, 1.5])
        tracker.update(
            hmd_rotation=np.eye(3),
            actual_joints=actual,
            packet_received_at=0.0,
            now=0.0,
        )
        target = tracker.update(
            hmd_rotation=rotation_z(1.0) @ rotation_y(1.0),
            actual_joints=actual,
            packet_received_at=0.1,
            now=0.1,
        )
        np.testing.assert_allclose(target, [0.523, 1.57], atol=1e-8)

    def test_stale_packet_holds_and_new_packet_rebases_without_jump(self):
        tracker = self.make_tracker()
        actual = np.array([0.0, 0.7])
        tracker.update(
            hmd_rotation=np.eye(3),
            actual_joints=actual,
            packet_received_at=0.0,
            now=0.0,
        )
        moved = tracker.update(
            hmd_rotation=rotation_z(0.2),
            actual_joints=actual,
            packet_received_at=0.1,
            now=0.1,
        )
        held = tracker.update(
            hmd_rotation=rotation_z(0.2),
            actual_joints=moved,
            packet_received_at=0.1,
            now=1.0,
        )
        np.testing.assert_allclose(held, moved)
        self.assertTrue(tracker.stale)

        rebased = tracker.update(
            hmd_rotation=rotation_z(1.0),
            actual_joints=moved,
            packet_received_at=1.1,
            now=1.1,
        )
        np.testing.assert_allclose(rebased, moved)
        self.assertFalse(tracker.stale)

    def test_main_has_no_hmd_to_torso_follow_transform(self):
        source = (Path(__file__).parents[1] / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("torso_start_pose @ diff", source)
        self.assertNotIn("is_torso_following and torso_mode", source)
        self.assertIn("torso_T = SystemContext.vr_state.torso_locked_pose", source)
        self.assertIn("component_command.set_head_command(head_command)", source)


class _FakeMediaStreamTrack:
    pass


class _FakeVideoFrame:
    def __init__(self, array):
        self.array = array
        self.height, self.width = array.shape[:2]
        self.pts = None
        self.time_base = None

    @classmethod
    def from_ndarray(cls, array, format):
        assert format == "rgb24"
        return cls(array.copy())

    def reformat(self, *, width, height, format):
        self.width = width
        self.height = height
        return self


class _FakeRgbdFrame:
    def __init__(self):
        self.t = 1.0
        self.color = np.full((2, 3, 3), 17, dtype=np.uint8)


class _FakeRealsense:
    running = True

    def get_latest_frames(self):
        return {"head-serial": _FakeRgbdFrame()}


class QuestVideoTests(unittest.TestCase):
    def test_video_track_reads_latest_head_frame(self):
        source = HeadCameraVideoTrack(
            media_stream_track_cls=_FakeMediaStreamTrack,
            video_frame_cls=_FakeVideoFrame,
            realsense=_FakeRealsense(),
            camera_serial="head-serial",
            width=3,
            height=2,
            fps=30,
            stale_timeout_sec=0.5,
        )
        frame = asyncio.run(source.track.recv())
        self.assertEqual(frame.pts, 0)
        np.testing.assert_array_equal(frame.array, np.full((2, 3, 3), 17, dtype=np.uint8))

    @mock.patch("quest_streaming.importlib.util.find_spec", return_value=None)
    def test_missing_webrtc_dependencies_do_not_start_server(self, _find_spec):
        server = QuestVideoServer(
            realsense=None,
            camera_name="head",
            camera_serial=None,
            bind_host="127.0.0.1",
            port=8080,
            width=640,
            height=480,
            fps=30,
            codec="H264",
            stale_timeout_sec=0.5,
        )
        self.assertFalse(server.start())
        self.assertIn("missing", server.status_snapshot()["error"].lower())


@unittest.skipUnless(webrtc_dependencies_available(), "optional WebRTC dependencies are not installed")
class QuestVideoIntegrationTests(unittest.TestCase):
    def test_h264_offer_answer_delivers_a_video_frame(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        server = QuestVideoServer(
            realsense=_FakeRealsense(),
            camera_name="head",
            camera_serial="head-serial",
            bind_host="127.0.0.1",
            port=port,
            width=64,
            height=48,
            fps=30,
            codec="H264",
            stale_timeout_sec=0.5,
        )
        self.assertTrue(server.start(), server.status_snapshot())

        async def negotiate_and_receive():
            import aiohttp
            from aiortc import RTCPeerConnection, RTCSessionDescription

            peer = RTCPeerConnection()
            frame_future = asyncio.get_running_loop().create_future()

            @peer.on("track")
            def on_track(track):
                async def receive_one():
                    frame = await track.recv()
                    if not frame_future.done():
                        frame_future.set_result(frame)

                asyncio.create_task(receive_one())

            try:
                peer.addTransceiver("video", direction="recvonly")
                offer = await peer.createOffer()
                await peer.setLocalDescription(offer)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/offer",
                        json={
                            "type": peer.localDescription.type,
                            "sdp": peer.localDescription.sdp,
                        },
                    ) as response:
                        self.assertEqual(response.status, 200, await response.text())
                        answer = await response.json()
                await peer.setRemoteDescription(
                    RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
                )
                frame = await asyncio.wait_for(frame_future, timeout=10.0)
                self.assertEqual((frame.width, frame.height), (64, 48))
            finally:
                await peer.close()

        try:
            asyncio.run(negotiate_and_receive())
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
