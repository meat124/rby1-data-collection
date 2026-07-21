import argparse
import base64
import importlib
import logging
import zmq
import time
import threading
from local_sdk_bootstrap import ensure_local_rby1_sdk_path

ensure_local_rby1_sdk_path()

try:
    rby = importlib.import_module("rby1_sdk")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "rby1_sdk is required to run the Meta Quest RBY1 backend. "
        "Build or install the SDK bindings before starting main.py."
    ) from exc
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from remote_gripper import Gripper
import pickle
import os

from pcd_utils import rgbd_to_pointcloud, REALSENSE_D435_INTRINSICS, REALSENSE_D435_INTRINSICS_848x480

from utils import *
from helper import *

from camera import MultiRealsense
from head_tracking import HeadTracker, HeadTrackingConfig
from quest_streaming import QuestVideoServer
from recording_manager import RecordingManager
from setup import Settings, SystemContext
from robot_communicate import connect_rby1
from vr_communicate import setup_meta_quest_udp_communication, handle_vr_button_event


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)

T_conv = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])

def open_zmq_pub_socket(bind_address: str) -> zmq.Socket:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(bind_address)
    logging.info(f"ZMQ PUB server opened at {bind_address}")
    return socket


def open_zmq_rep_socket(bind_address: str) -> zmq.Socket:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(bind_address)
    logging.info(f"ZMQ REP server opened at {bind_address}")
    return socket

def pose_to_se3(position, rotation_quat):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3] = position
    return T


def publish_gv(bind_address: str):
    sock = open_zmq_pub_socket(bind_address)
    while True:
        try:
            sock.send(pickle.dumps(SystemContext.vr_state))
        except Exception as e:
            logging.warning(f"[publish_gv] Failed to publish state: {e}")
        time.sleep(0.1)


def load_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError:
        return None


def encode_primary_preview_jpeg(max_width: int = 480, quality: int = 80) -> str | None:
    realsense_obj = SystemContext.realsense
    realsense = realsense_obj if isinstance(realsense_obj, MultiRealsense) else None
    if realsense is None or not getattr(realsense, "running", False):
        return None

    frames = realsense.get_frames()
    if not frames and hasattr(realsense, "get_latest_frames"):
        frames = realsense.get_latest_frames()
    if not frames:
        return None

    camera_context = build_camera_recording_context(Settings.rec_fps)
    primary_serial = camera_context["primary_serial"]
    if primary_serial is None or primary_serial not in frames:
        return None

    frame = frames[primary_serial]

    rgb = getattr(frame, "color", None)
    if rgb is None:
        return None

    try:
        cv2 = load_cv2()
        if cv2 is None:
            return None

        height, width = rgb.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)))
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")
    except Exception as exc:
        logging.warning("[dashboard] Failed to encode preview frame: %s", exc)
        return None


def queue_dashboard_initialization(command: str) -> bool:
    if command == "init_torso_mode":
        SystemContext.vr_state.event_right_a_pressed = True
        return True
    if command == "init_bimanual_mode":
        SystemContext.vr_state.event_left_b_pressed = True
        return True
    return False


def build_dashboard_snapshot(
    recording_manager: RecordingManager,
    *,
    robot=None,
    gripper=None,
    include_preview: bool = False,
) -> dict:
    recording_manager.refresh_preflight(robot=robot, gripper=gripper)
    snapshot = recording_manager.get_status_snapshot()
    snapshot["head_tracking"] = dict(SystemContext.head_tracking_status)
    video_server = SystemContext.quest_video_server
    if video_server is not None:
        SystemContext.quest_video_status = video_server.status_snapshot()
    snapshot["quest_video"] = dict(SystemContext.quest_video_status)
    snapshot["ok"] = True
    if include_preview:
        snapshot["preview_jpeg_base64"] = encode_primary_preview_jpeg()
    return snapshot


def start_dashboard_command_server(bind_address: str, recording_manager: RecordingManager, *, robot=None, gripper=None):
    def _serve():
        socket = open_zmq_rep_socket(bind_address)
        while True:
            response = {"ok": False, "error": "uninitialized"}
            try:
                request = socket.recv_json()
                if not isinstance(request, dict):
                    raise ValueError("Dashboard request must be a JSON object")
                command = request.get("command", "get_status")
                if not isinstance(command, str):
                    raise ValueError("Dashboard command must be a string")
                include_preview = bool(request.get("include_preview", False))

                if command == "get_status":
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=include_preview,
                    )
                elif command == "start_recording":
                    recording_manager.refresh_preflight(robot=robot, gripper=gripper)
                    accepted = recording_manager.request_start_recording()
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=True,
                    )
                    response.update({"ok": accepted, "accepted": accepted})
                    if not accepted:
                        response["error"] = response.get("message") or "Start request rejected"
                elif command == "stop_recording":
                    accepted = recording_manager.request_stop_and_save()
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=True,
                    )
                    response.update({"ok": accepted, "accepted": accepted})
                    if not accepted:
                        response["error"] = response.get("message") or "Stop request rejected"
                elif command == "toggle_recording":
                    recording_manager.refresh_preflight(robot=robot, gripper=gripper)
                    accepted = recording_manager.request_toggle_recording()
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=True,
                    )
                    response.update({"ok": accepted, "accepted": accepted})
                    if not accepted:
                        response["error"] = response.get("message") or "Toggle request rejected"
                elif command == "discard_last_demo":
                    accepted = recording_manager.request_discard_last_demo()
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=True,
                    )
                    response.update({"ok": accepted, "accepted": accepted})
                    if not accepted:
                        response["error"] = response.get("message") or "Discard request rejected"
                elif command == "init_torso_mode":
                    accepted = queue_dashboard_initialization(command)
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=True,
                    )
                    response.update(
                        {
                            "ok": accepted,
                            "accepted": accepted,
                            "command_message": "Torso-mode ready pose requested. Robot motion can take several seconds.",
                        }
                    )
                elif command == "init_bimanual_mode":
                    accepted = queue_dashboard_initialization(command)
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=True,
                    )
                    response.update(
                        {
                            "ok": accepted,
                            "accepted": accepted,
                            "command_message": "Bimanual ready pose requested. Robot motion can take several seconds.",
                        }
                    )
                elif command == "check_cameras":
                    response = build_dashboard_snapshot(
                        recording_manager,
                        robot=robot,
                        gripper=gripper,
                        include_preview=True,
                    )
                    response.update(
                        {
                            "ok": True,
                            "accepted": True,
                            "command_message": "Camera diagnostics refreshed.",
                        }
                    )
                else:
                    response = {"ok": False, "error": f"Unknown command: {command}"}
            except Exception as exc:
                logging.exception("[dashboard] Command server error")
                response = {"ok": False, "error": str(exc)}

            try:
                socket.send_json(response)
            except Exception as exc:
                logging.warning("[dashboard] Failed to send command response: %s", exc)

    thread = threading.Thread(target=_serve, name="dashboard_command_server", daemon=True)
    thread.start()
    return thread

def get_config():
    return load_collection_config()


def build_camera_recording_context(fps: int, camera_names: set[str] | None = None) -> dict:
    config = get_config()
    all_cameras_config = config.get('cameras', {}) or {}
    cameras_config = get_enabled_camera_configs(config)
    if camera_names is not None:
        cameras_config = {
            name: spec for name, spec in cameras_config.items()
            if name in camera_names
        }
    if not all_cameras_config:
        logging.warning("No camera configuration found in config.yaml. Camera streams will not be processed.")
    elif not cameras_config:
        logging.warning(
            "Requested cameras are not enabled in config.yaml: %s",
            sorted(camera_names or []),
        )

    cam_names = list(cameras_config.keys())
    cam_name_to_serial = {cam_name: spec['serial'] for cam_name, spec in cameras_config.items()}
    camera_serials = [cam_name_to_serial[cam_name] for cam_name in cam_names]
    serial_to_cam_name = {v: k for k, v in cam_name_to_serial.items()}
    primary_cam_name = "head" if "head" in cam_name_to_serial else (cam_names[0] if cam_names else None)

    context = {
        "cam_names": cam_names,
        "cam_name_to_serial": cam_name_to_serial,
        "camera_serials": camera_serials,
        "serial_to_cam_name": serial_to_cam_name,
        "expected_serials": set(camera_serials),
        "primary_cam_name": primary_cam_name,
        "primary_serial": cam_name_to_serial.get(primary_cam_name) if primary_cam_name else None,
        "img_size": config.get('img_size', {'width': 640, 'height': 480}),
        "capture_fps": max(int(fps), int(Settings.quest_video.get("fps", 30))),
        "camera_warmup_secs": float(config.get('camera_warmup_secs', 2.0)),
        "camera_warmup_frames": int(config.get('camera_warmup_frames', max(15, int(fps)))),
    }
    SystemContext.camera_context = context
    return context


def ensure_realsense_started(camera_context: dict) -> MultiRealsense | None:
    camera_serials = camera_context["camera_serials"]
    if not camera_serials:
        return None

    realsense_obj = SystemContext.realsense
    realsense = realsense_obj if isinstance(realsense_obj, MultiRealsense) else None
    if realsense is not None and getattr(realsense, "running", False):
        return realsense

    img_size = camera_context["img_size"]
    realsense = MultiRealsense(
        camera_serials=camera_serials,
        width=img_size['width'],
        height=img_size['height'],
        fps=int(camera_context["capture_fps"]),
    )
    realsense.start()
    SystemContext.realsense = realsense
    SystemContext.camera_warmup_done = False
    logging.info("Started RealSense streams (warm-up will run once).")
    return realsense

# NOTE
def start_demo_logger(
    gripper: Gripper | None,
    h5_writer,
    robot,
    *,
    data_collection_enabled: bool,
    fps: int = 30,
) -> threading.Event:
    """
    Starts a daemon thread that prints a dict with:
      - robot_state.position
      - gripper.get_state()
    at ~fps Hz (default 30). Returns a stop_event you can set() to stop it.
    """
    print("start demo logger on\n")
    period = 1.0 / float(fps)
    stop_event = threading.Event()

    camera_context = build_camera_recording_context(fps=fps)
    cam_names = camera_context["cam_names"]
    cam_name_to_serial = camera_context["cam_name_to_serial"]
    serial_to_cam_name = camera_context["serial_to_cam_name"]
    expected_serials = camera_context["expected_serials"]
    primary_serial = camera_context["primary_serial"]
    camera_warmup_secs = camera_context["camera_warmup_secs"]
    camera_warmup_frames = camera_context["camera_warmup_frames"]

    realsense = ensure_realsense_started(camera_context)
    if realsense is not None and getattr(realsense, "running", False):
        logging.info("Reusing existing RealSense streams (warm-up already done).")

    def _loop():
        logging.info("loop started")
        next_t = time.perf_counter()
        record_start_t = time.perf_counter()
        last_logged_primary_ts = None
        last_camera_wait_warn_t = 0.0
        last_warmup_warn_t = 0.0
        warmup_unique_frames = 0
        last_warmup_primary_ts = None

        while not stop_event.is_set():
            # Thread-safe snapshot of latest camera frames (optional)
            frames = realsense.get_frames() if realsense is not None else {}

            if not frames:
                frames = {}

            # Use the first camera in cam_ids for primary tasks like PCD generation (if available).
            primary_cam_id = None
            primary_cam_data = None
            frame = None
            depth = None
            frame_stamp = None

            if frames and primary_serial is not None and primary_serial in frames:
                try:
                    primary_cam_id = primary_serial
                    primary_cam_data = frames[primary_cam_id]
                    frame = primary_cam_data.color
                    depth = primary_cam_data.depth
                    frame_stamp = primary_cam_data.t
                except Exception as e:
                    logging.warning(f"[demo_logger] Failed to get primary camera data: {e}")
                    frame = None
                    depth = None
                    frame_stamp = None

            # Strict synchronization rule for dataset recording:
            # record only when every configured camera exists and primary frame timestamp is fresh.
            data_collection_bool = data_collection_enabled
            if data_collection_bool and expected_serials:
                missing_serials = expected_serials - set(frames.keys())
                if missing_serials:
                    now_t = time.perf_counter()
                    if now_t - last_camera_wait_warn_t > 1.0:
                        missing_names = [serial_to_cam_name.get(s, s) for s in sorted(missing_serials)]
                        logging.warning(
                            f"[demo_logger] Missing synced frames from cameras={missing_names}. "
                            "Skip this timestep to keep robot/image counts aligned."
                        )
                        last_camera_wait_warn_t = now_t

                    next_t += period
                    sleep_for = next_t - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_t = time.perf_counter()
                    continue

                if frame_stamp is None:
                    next_t += period
                    sleep_for = next_t - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_t = time.perf_counter()
                    continue

                if last_logged_primary_ts is not None and frame_stamp <= (last_logged_primary_ts + 1e-6):
                    # No new camera frame yet; skip to avoid duplicated image samples.
                    next_t += period
                    sleep_for = next_t - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_t = time.perf_counter()
                    continue

                # Camera warm-up gate (process-level one-time):
                # drop initial frames while auto-exposure/auto-white-balance converges.
                if not SystemContext.camera_warmup_done:
                    if frame_stamp is not None and (
                        last_warmup_primary_ts is None or frame_stamp > (last_warmup_primary_ts + 1e-6)
                    ):
                        warmup_unique_frames += 1
                        last_warmup_primary_ts = frame_stamp

                    elapsed = time.perf_counter() - record_start_t
                    warmup_done = (
                        (elapsed >= camera_warmup_secs)
                        and (warmup_unique_frames >= camera_warmup_frames)
                    )

                    if not warmup_done:
                        now_t = time.perf_counter()
                        if now_t - last_warmup_warn_t > 1.0:
                            logging.info(
                                "[demo_logger] Camera warm-up in progress "
                                f"(elapsed={elapsed:.2f}s/{camera_warmup_secs:.2f}s, "
                                f"frames={warmup_unique_frames}/{camera_warmup_frames}). "
                                "Skip initial unstable frames."
                            )
                            last_warmup_warn_t = now_t

                        next_t += period
                        sleep_for = next_t - time.perf_counter()
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        else:
                            next_t = time.perf_counter()
                        continue

                    SystemContext.camera_warmup_done = True
                    logging.info("[demo_logger] Camera warm-up completed (process-level one-time).")

            # Robot joint positions - 현재 포지션 읽기
            robot_pos = None

            try:
                if SystemContext.vr_state.joint_positions is not None:
                    # robot_pos = np.array(SystemContext.robot_state.position, dtype=float)
                # elif SystemContext.vr_state.joint_positions.size > 0:
                    # fallback to the latest joints cached in vr_state
                    robot_pos = np.array(SystemContext.vr_state.joint_positions, dtype=float)
            except Exception as e:
                logging.warning(f"[demo_logger] Failed to read robot position: {e}")
            
            # Skip this step if robot position is unavailable
            if robot_pos is None:
                next_t += period
                sleep_for = next_t - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_t = time.perf_counter()
                continue

            # robot_target_joints initially equals current position;
            # update_previous_target() will overwrite step t-1 with robot_pos[t] on the next iteration.
            robot_target_joints = robot_pos.copy()

            # Gripper encoders (actual measured state)
            grip = None
            grip_target = None
            ft_sensor_right_force = None
            ft_sensor_right_torque = None
            ft_sensor_left_force = None
            ft_sensor_left_torque = None
            try:
                with SystemContext.sensor_lock:
                    if SystemContext.gripper_state is not None:
                        grip = np.array(SystemContext.gripper_state, dtype=float)
                    if SystemContext.gripper_target is not None:
                        grip_target = np.array(SystemContext.gripper_target, dtype=float)
                    ft_sensor_right_force = np.array(SystemContext.ft_sensor_right_force, dtype=float)
                    ft_sensor_right_torque = np.array(SystemContext.ft_sensor_right_torque, dtype=float)
                    ft_sensor_left_force = np.array(SystemContext.ft_sensor_left_force, dtype=float)
                    ft_sensor_left_torque = np.array(SystemContext.ft_sensor_left_torque, dtype=float)

                if gripper is not None and hasattr(gripper, "get_state"):
                    if grip is None:
                        grip = gripper.get_state()  # expected to return np.ndarray or None
                        if grip is not None:
                            with SystemContext.sensor_lock:
                                SystemContext.gripper_state = np.array(grip, dtype=float)
                    if grip_target is None:
                        grip_target = gripper.get_target().copy()
                        with SystemContext.sensor_lock:
                            SystemContext.gripper_target = np.array(grip_target, dtype=float)
            except Exception as e:
                logging.warning(f"[demo_logger] Failed to read gripper state: {e}")

            # base velocity
            base_state = None
            try:
                if SystemContext.vr_state.mobile_linear_velocity is not None:
                    linear_vel = np.array(SystemContext.vr_state.mobile_linear_velocity, dtype=float)
                    linear_vel = np.squeeze(linear_vel)
                    angular_vel = np.array(SystemContext.vr_state.mobile_angular_velocity, dtype=float)
                    base_state = np.concatenate((linear_vel, [angular_vel]), axis=0)
            except Exception as e:
                logging.warning(f"[demo_logger] Failed to read base velocity: {e}")

            '''
            x 버튼 한번 누르면 데이터 수집 시작
            x 버튼 한번 더 누르면 데이터 수집 종료 및 저장
            x 버튼 다시 누르면 새로운 데모 수집
            '''

            # logging.info(f"data_collection_bool activated: {data_collection_bool}")

            if data_collection_bool:
                # Generate PCD from RGB-D
                pcd_points = None
                pcd_colors = None
                if frame is not None and depth is not None:
                    try:
                        rgb_frame = frame
                        depth_frame = depth
                        print("camera time stamp : ", frame_stamp)
                        
                        # Determine intrinsics based on depth image size
                        H_d, W_d = depth_frame.shape
                        if W_d == 848 and H_d == 480:                      
                            intrinsics = REALSENSE_D435_INTRINSICS_848x480
                        else:
                            intrinsics = REALSENSE_D435_INTRINSICS
                        
                        # Resize RGB to match depth if needed
                        if rgb_frame.shape[:2] != depth_frame.shape:
                            try:
                                cv2 = load_cv2()
                                if cv2 is None:
                                    raise ModuleNotFoundError("cv2")
                                rgb_frame = cv2.resize(rgb_frame, (depth_frame.shape[1], depth_frame.shape[0]))
                            except ImportError:
                                logging.warning("[demo_logger] cv2 not available for RGB resize")
                        depth_frame_for_pcd = np.where(depth_frame > 3000, 0, depth_frame)  
                        # Convert to point cloud (all points, no downsampling)
                        pcd_points, pcd_colors = rgbd_to_pointcloud(
                            rgb_frame, depth_frame_for_pcd,
                            intrinsics['fx'], intrinsics['fy'],
                            intrinsics['cx'], intrinsics['cy'],
                            max_points=Settings.MAX_POINT
                        )
                        
                        # Debug: check if PCD was generated
                        if pcd_points is not None and len(pcd_points) > 0:
                            print(f"[demo_logger] Generated PCD: {len(pcd_points)} points")
                        else:
                            print(f"[demo_logger] WARNING: PCD generation returned empty result")
                    except Exception as e:
                        logging.warning(f"[demo_logger] Failed to generate PCD: {e}")
                
                
                # Align global sample timestamp to camera timestamp
                sample_ts = float(frame_stamp) if frame_stamp is not None else time.time()

                # Update previous step's target only when we are actually committing a sample.
                h5_writer.update_previous_target(robot_pos)

                print("demo saved\n")
                # print("robot_pose shape: ", robot_pos.shape)
                # print("robot_target_joints shape: ", robot_target_joints.shape)
                # print("gripper_state shape: ", grip.shape if grip is not None else None)
                # print("base_state shape: ", base_state.shape if base_state is not None else None)
                # print("head_rgb shape: ", headcam_sub.curr_frame.shape if headcam_sub.curr_frame is not None else None)
                

                # Create a dynamic dictionary to hold data for all cameras
                data_to_save = {
                    "ts": sample_ts,
                    "robot_position": robot_pos,
                    "robot_target_joints": robot_target_joints,
                    "gripper_state": grip,
                    "gripper_target": grip_target,
                    "base_state" : base_state,
                    "ft_sensor_right_force": ft_sensor_right_force,
                    "ft_sensor_right_torque": ft_sensor_right_torque,
                    "ft_sensor_left_force": ft_sensor_left_force,
                    "ft_sensor_left_torque": ft_sensor_left_torque,
                    "pcd_points": pcd_points,
                    "pcd_colors": pcd_colors
                }

                # Iterate through all available cameras and add their data to the dictionary
                for cam_name in cam_names:
                    cam_id = cam_name_to_serial[cam_name]
                    if cam_id not in frames:
                        # Should not happen due strict check above, but keep safe.
                        continue
                    logging.info(f"{cam_name} image is saving")
                    cam_data = frames[cam_id]
                    data_to_save[f"{cam_name}_rgb"] = cam_data.color
                    data_to_save[f"{cam_name}_rgb_ts"] = cam_data.t
                    if cam_data.depth is not None:
                        data_to_save[f"{cam_name}_depth"] = cam_data.depth
                        data_to_save[f"{cam_name}_depth_ts"] = cam_data.t
                    # logging.info(f"{cam_data} color frame is {cam_data.color}")
                h5_writer.put(data_to_save)
                if frame_stamp is not None:
                    last_logged_primary_ts = frame_stamp

            # pacing at ~30 FPS with drift correction
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # we overran; reset schedule to now to avoid accumulating lag
                next_t = time.perf_counter()

    t = threading.Thread(target=_loop, name="demo_logger", daemon=True)
    t.start()
    logging.info(f"Started demo logger at {fps} FPS")

    # This is a bit of a hack. The _loop will run until the main thread exits.
    # We return a stop_event that can be used to signal the loop to stop.
    # Camera stop is handled at process shutdown, so streams can be reused across demos.
    def cleanup():
        stop_event.set()

    SystemContext.cleanup_functions.append(cleanup)
    
    return stop_event

def main(args: argparse.Namespace):
    logging.info("=== VR Control System Starting ===")
    logging.info(f"Server Address       : {args.server}")
    logging.info(f"Local (UPC) IP       : {args.local_ip}:{Settings.vr_control_local_port}")
    logging.info(f"Meta Quest IP        : {args.meta_quest_ip}:{Settings.vr_control_meta_quest_port}")
    logging.info(f"Use Gripper          : {'No' if args.no_gripper else 'Yes'}")
    logging.info(f"RB-Y1 gRPC Address   : {args.rby1}")
    logging.info(f"RB-Y1 Model          : {args.rby1_model}")
    logging.info(f"Use Head             : {'No' if args.no_head else 'Yes'}")
    logging.info(f"Data Collection      : {args.data_collect}")
    logging.info(f"Quest Head-Cam Video : {args.quest_video}")
    logging.info(f"Dashboard Command API: {args.dashboard_cmd_server}")

    config = get_config()
    root_path = os.path.join(config['demo_root'], config['task_name'])
    if args.data_collect:
        os.makedirs(root_path, exist_ok=True)
    SystemContext.storage_root = root_path

    recording_manager = None
    if args.data_collect:
        logging.info("Data collection mode enabled - H5 file will be created when X button is pressed")

    def power_off_and_stop():
        video_server = SystemContext.quest_video_server
        if video_server is not None:
            video_server.stop()
        if recording_manager is not None:
            recording_manager.shutdown()
        if isinstance(SystemContext.realsense, MultiRealsense):
            try:
                SystemContext.realsense.stop()
            except Exception as e:
                logging.warning(f"Failed to stop camera during shutdown: {e}")
        robot.power_off(".*")
        for cleanup_func in SystemContext.cleanup_functions:
            cleanup_func()


    robot = connect_rby1(args.rby1, args.rby1_model, args.no_head)
    model = robot.model()
    setup_meta_quest_udp_communication(
        args.local_ip, 
        Settings.vr_control_local_port, 
        args.meta_quest_ip,
        Settings.vr_control_meta_quest_port, 
        power_off_and_stop,
        video_enabled=args.quest_video,
    )

    gripper = None
    gripper_last_state_poll = 0.0
    gripper_state_failures = 0
    gripper_last_recovery = 0.0
    if not args.no_gripper:
        for arm in ["left", "right"]:
            if not robot.set_tool_flange_output_voltage(arm, 12):
                logging.error(f"Failed to supply 12V to tool flange. ({arm})")
        time.sleep(0.5)
        gripper = Gripper()
        if not gripper.initialize(verbose=True):
            exit(1)
        gripper.homing()
        gripper.start()
        gripper.set_normalized_target(np.array([1.0, 1.0]))
        with SystemContext.sensor_lock:
            SystemContext.gripper_target = np.array([1.0, 1.0], dtype=float)
        try:
            initial_gripper_state = np.array(gripper.get_state(), dtype=float)
        except Exception as exc:
            initial_gripper_state = None
            logging.warning("[gripper] Failed to read initial state: %s", exc)
        if initial_gripper_state is not None:
            with SystemContext.sensor_lock:
                SystemContext.gripper_state = initial_gripper_state

    def refresh_gripper_feedback(force: bool = False):
        nonlocal gripper
        nonlocal gripper_last_state_poll
        nonlocal gripper_state_failures
        nonlocal gripper_last_recovery

        if gripper is None:
            return

        now = time.monotonic()
        if not force and (now - gripper_last_state_poll) < 0.2:
            return
        gripper_last_state_poll = now

        try:
            current_gripper_state = np.array(gripper.get_state(), dtype=float)
            with SystemContext.sensor_lock:
                SystemContext.gripper_state = current_gripper_state
            gripper_state_failures = 0
            return
        except Exception as exc:
            gripper_state_failures += 1
            if gripper_state_failures == 1 or gripper_state_failures % 5 == 0:
                logging.warning(
                    "[gripper] Failed to refresh state (%d): %s",
                    gripper_state_failures,
                    exc,
                )
            if gripper_state_failures < 3 or (now - gripper_last_recovery) < 2.0:
                return

        gripper_last_recovery = now
        with SystemContext.sensor_lock:
            cached_target = (
                np.array(SystemContext.gripper_target, dtype=float)
                if SystemContext.gripper_target is not None
                else None
            )
        if cached_target is None:
            try:
                cached_target = np.array(gripper.get_target(), dtype=float)
            except Exception:
                cached_target = np.array([1.0, 1.0], dtype=float)

        old_gripper = gripper
        try:
            recovered_gripper = Gripper()
            if not recovered_gripper.initialize(verbose=True):
                raise RuntimeError("remote gripper ping failed during recovery")
            recovered_gripper.start()
            recovered_gripper.set_normalized_target(cached_target, wait_for_reply=False)
            recovered_gripper_state = np.array(recovered_gripper.get_state(), dtype=float)

            gripper = recovered_gripper
            gripper_state_failures = 0
            with SystemContext.sensor_lock:
                SystemContext.gripper_target = np.array(cached_target, dtype=float)
                SystemContext.gripper_state = recovered_gripper_state
            logging.warning("[gripper] Reconnected remote gripper client after repeated state failures")
        except Exception as recovery_exc:
            logging.warning("[gripper] Recovery attempt failed: %s", recovery_exc)
        finally:
            if old_gripper is not gripper:
                try:
                    old_gripper.close()
                except Exception:
                    pass

    pub_thread = threading.Thread(target=publish_gv, args=(args.server,), daemon=True)
    pub_thread.start()

    # Initialize cleanup functions container before starting subsystems
    SystemContext.cleanup_functions = []

    camera_context = None
    realsense = None
    if args.data_collect or args.quest_video:
        requested_cameras = None
        if args.quest_video and not args.data_collect:
            requested_cameras = {str(Settings.quest_video.get("camera", "head"))}
        camera_context = build_camera_recording_context(
            Settings.rec_fps,
            camera_names=requested_cameras,
        )
        realsense = ensure_realsense_started(camera_context)

    if args.quest_video:
        video_config = Settings.quest_video
        camera_name = str(video_config.get("camera", "head"))
        camera_serial = None
        if camera_context is not None:
            camera_serial = camera_context["cam_name_to_serial"].get(camera_name)
        if camera_serial is None:
            logging.error(
                "[quest-video] Camera '%s' is not enabled in config.yaml; streaming black status frames",
                camera_name,
            )
        quest_video_server = QuestVideoServer(
            realsense=realsense,
            camera_name=camera_name,
            camera_serial=camera_serial,
            bind_host=str(video_config.get("bind_host", "0.0.0.0")),
            port=int(video_config.get("port", 8080)),
            width=int(video_config.get("width", 640)),
            height=int(video_config.get("height", 480)),
            fps=int(video_config.get("fps", 30)),
            codec=str(video_config.get("codec", "H264")),
            stale_timeout_sec=float(video_config.get("stale_frame_timeout_sec", 0.5)),
        )
        SystemContext.quest_video_server = quest_video_server
        quest_video_server.start()
        SystemContext.quest_video_status = quest_video_server.status_snapshot()
        SystemContext.cleanup_functions.append(quest_video_server.stop)
    else:
        SystemContext.quest_video_server = None
        SystemContext.quest_video_status = {"enabled": False, "running": False}

    if args.data_collect:
        recording_manager = RecordingManager(
            root_path=root_path,
            logger_factory=lambda gripper_handle, writer_handle, robot_handle: start_demo_logger(
                gripper_handle,
                writer_handle,
                robot_handle,
                data_collection_enabled=args.data_collect,
                fps=Settings.rec_fps,
            ),
        )
        SystemContext.recording_manager = recording_manager
        recording_manager.refresh_preflight(robot=robot, gripper=gripper)
        start_dashboard_command_server(
            args.dashboard_cmd_server,
            recording_manager,
            robot=robot,
            gripper=gripper,
        )
        logging.info("Data collection backend is ready and waiting for an explicit start request")
    else:
        SystemContext.recording_manager = None

    dyn_robot = robot.get_dynamics()
    if SystemContext.robot_model is None:
        raise RuntimeError("SystemContext.robot_model must be initialized before entering the control loop")
    dyn_state = dyn_robot.make_state(["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
                                     SystemContext.robot_model.robot_joint_names)
    base_link_idx, link_torso_5_idx, link_right_arm_6_idx, link_left_arm_6_idx = 0, 1, 2, 3

    head_tracker = None
    head_indices = np.asarray(model.head_idx, dtype=int)
    if not args.no_head:
        if len(head_indices) != 2:
            logging.warning("Head tracking disabled: expected 2 head joints, got %d", len(head_indices))
        else:
            head_config = Settings.head_tracking
            lower_limits = dyn_robot.get_limit_q_lower(dyn_state)[head_indices]
            upper_limits = dyn_robot.get_limit_q_upper(dyn_state)[head_indices]
            head_tracker = HeadTracker(
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                config=HeadTrackingConfig(
                    stale_timeout_sec=float(head_config["stale_timeout_sec"]),
                    joint_limit_margin_rad=np.deg2rad(float(head_config["joint_limit_margin_deg"])),
                    max_velocity_rad_s=np.deg2rad(float(head_config["max_velocity_deg_s"])),
                    max_acceleration_rad_s2=np.deg2rad(float(head_config["max_acceleration_deg_s2"])),
                    yaw_scale=float(head_config["yaw_scale"]),
                    pitch_scale=float(head_config["pitch_scale"]),
                ),
            )
            logging.info(
                "HMD head tracking enabled with limits lower=%s upper=%s",
                head_tracker.lower_limits,
                head_tracker.upper_limits,
            )
    SystemContext.head_tracker = head_tracker

    next_time = time.monotonic()
    stream = None
    torso_reset = False
    right_reset = False
    left_reset = False

    logging.info("Entering main control loop")

    while True:
        now = time.monotonic()
        if now < next_time:
            time.sleep(next_time - now)
        next_time += Settings.dt

        if recording_manager is not None:
            recording_manager.refresh_preflight(robot=robot, gripper=gripper)

        refresh_gripper_feedback()

        if SystemContext.vr_state.joint_positions.size == 0:
            continue
        stream_reset_requested, _torso_mode = handle_vr_button_event(robot, args.no_head)
        if recording_manager is not None:
            recording_manager.process_pending_actions(gripper=gripper, robot=robot)
        
        if stream_reset_requested:
            if stream is not None:
                stream.cancel()
                stream = None

        if not SystemContext.vr_state.is_initialized:
            continue

        if SystemContext.vr_state.is_stopped:
            if stream is not None:
                stream.cancel()
                stream = None
            SystemContext.vr_state.is_initialized = False
            continue

        # Gripper: update cached target and send (avoid remote get_normalized_target() RPC every tick)
        if "hands" in SystemContext.vr_state.controller_state:
            if "right" in SystemContext.vr_state.controller_state["hands"]:
                right_controller = SystemContext.vr_state.controller_state["hands"]["right"]
                if gripper is not None:
                    gripper_target = gripper.get_target()
                    gripper_target[0] = 1. -right_controller["buttons"]["trigger"]
                    gripper.set_normalized_target(gripper_target)
                    with SystemContext.sensor_lock:
                        SystemContext.gripper_target = np.array(gripper_target, dtype=float)

            if "left" in SystemContext.vr_state.controller_state["hands"]:
                left_controller = SystemContext.vr_state.controller_state["hands"]["left"]
                if gripper is not None:
                    gripper_target = gripper.get_target()
                    gripper_target[1] = 1. - left_controller["buttons"]["trigger"]
                    gripper.set_normalized_target(gripper_target)
                    with SystemContext.sensor_lock:
                        SystemContext.gripper_target = np.array(gripper_target, dtype=float)

        # logging.info(f"{SystemContext.vr_state.center_of_mass = }")

        dyn_state.set_q(SystemContext.vr_state.joint_positions.copy())
        dyn_robot.compute_forward_kinematics(dyn_state)

        SystemContext.vr_state.base_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx, link_torso_5_idx)
        SystemContext.vr_state.torso_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                     link_torso_5_idx)
        SystemContext.vr_state.right_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                        link_right_arm_6_idx) @ Settings.T_hand_offset
        SystemContext.vr_state.left_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                       link_left_arm_6_idx) @ Settings.T_hand_offset

        # Tracking
        if stream is None:
            if robot.wait_for_control_ready(0):
                stream = robot.create_command_stream()
                SystemContext.vr_state.mobile_linear_velocity = np.array([0.0, 0.0])
                SystemContext.vr_state.mobile_angular_velocity = 0.
                SystemContext.vr_state.is_torso_following = False
                SystemContext.vr_state.is_right_following = False
                SystemContext.vr_state.is_left_following = False
                SystemContext.vr_state.base_start_pose = SystemContext.vr_state.base_pose
                SystemContext.vr_state.torso_locked_pose = SystemContext.vr_state.torso_current_pose
                SystemContext.vr_state.right_hand_locked_pose = SystemContext.vr_state.right_ee_current_pose
                SystemContext.vr_state.left_hand_locked_pose = SystemContext.vr_state.left_ee_current_pose
                torso_reset = True

        if "hands" in SystemContext.vr_state.controller_state:
            if "right" in SystemContext.vr_state.controller_state["hands"]:
                right_controller = SystemContext.vr_state.controller_state["hands"]["right"]
                thumbstick_axis = right_controller["buttons"]["thumbstickAxis"]
                acc = np.array([thumbstick_axis[1], thumbstick_axis[0]])
                SystemContext.vr_state.mobile_linear_velocity += Settings.mobile_linear_acceleration_gain * acc
                # SystemContext.vr_state.mobile_angular_velocity += Settings.mobile_angular_acceleration_gain * \
                #                                                   thumbstick_axis[0]
                SystemContext.vr_state.right_controller_current_pose = T_conv.T @ pose_to_se3(
                    right_controller["position"],
                    right_controller["rotation"]) @ T_conv

                trigger_pressed = right_controller["buttons"]["grip"] > 0.8
                if SystemContext.vr_state.is_right_following and not trigger_pressed:
                    SystemContext.vr_state.is_right_following = False
                if not SystemContext.vr_state.is_right_following and trigger_pressed:
                    SystemContext.vr_state.right_controller_start_pose = SystemContext.vr_state.right_controller_current_pose
                    SystemContext.vr_state.right_ee_start_pose = SystemContext.vr_state.right_ee_current_pose
                    SystemContext.vr_state.is_right_following = True
                    right_reset = True
            else:
                SystemContext.vr_state.is_right_following = False

            if "left" in SystemContext.vr_state.controller_state["hands"]:
                left_controller = SystemContext.vr_state.controller_state["hands"]["left"]
                thumbstick_axis = left_controller["buttons"]["thumbstickAxis"]
                # SystemContext.vr_state.mobile_linear_velocity += Settings.mobile_linear_acceleration_gain * \
                #                                                  thumbstick_axis[1]
                SystemContext.vr_state.mobile_angular_velocity += Settings.mobile_angular_acceleration_gain * \
                                                                  thumbstick_axis[0]
                SystemContext.vr_state.left_controller_current_pose = T_conv.T @ pose_to_se3(
                    left_controller["position"],
                    left_controller["rotation"]) @ T_conv

                trigger_pressed = left_controller["buttons"]["grip"] > 0.8
                if SystemContext.vr_state.is_left_following and not trigger_pressed:
                    SystemContext.vr_state.is_left_following = False
                if not SystemContext.vr_state.is_left_following and trigger_pressed:
                    SystemContext.vr_state.left_controller_start_pose = SystemContext.vr_state.left_controller_current_pose
                    SystemContext.vr_state.left_ee_start_pose = SystemContext.vr_state.left_ee_current_pose
                    SystemContext.vr_state.is_left_following = True
                    left_reset = True
            else:
                SystemContext.vr_state.is_left_following = False

        hmd_rotation = None
        if "head" in SystemContext.vr_state.controller_state:
            head_controller = SystemContext.vr_state.controller_state["head"]
            SystemContext.vr_state.head_controller_current_pose = T_conv.T @ pose_to_se3(
                head_controller["position"],
                head_controller["rotation"]) @ T_conv
            hmd_rotation = SystemContext.vr_state.head_controller_current_pose[:3, :3]

        SystemContext.vr_state.is_torso_following = False
        head_target = None
        if head_tracker is not None:
            if SystemContext.vr_state.head_recenter_requested:
                head_tracker.request_recenter()
                SystemContext.vr_state.head_recenter_requested = False
            actual_head_joints = SystemContext.vr_state.joint_positions[head_indices]
            head_target = head_tracker.update(
                hmd_rotation=hmd_rotation,
                actual_joints=actual_head_joints,
                packet_received_at=SystemContext.vr_state.controller_state_received_at,
                now=now,
            )
            SystemContext.vr_state.head_target_joints = head_target.copy()
            SystemContext.head_tracking_status = head_tracker.status_snapshot(now)

        SystemContext.vr_state.mobile_linear_velocity -= Settings.mobile_linear_damping_gain * SystemContext.vr_state.mobile_linear_velocity
        SystemContext.vr_state.mobile_angular_velocity -= Settings.mobile_angular_damping_gain * SystemContext.vr_state.mobile_angular_velocity

        if stream:
            try:
                if SystemContext.vr_state.is_right_following:
                    diff = np.linalg.inv(
                        SystemContext.vr_state.right_controller_start_pose) @ SystemContext.vr_state.right_controller_current_pose

                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.right_ee_start_pose[:3, :3]
                    right_T = SystemContext.vr_state.right_ee_start_pose @ diff_global
                    SystemContext.vr_state.right_hand_locked_pose = right_T
                else:
                    right_T = SystemContext.vr_state.right_hand_locked_pose

                if SystemContext.vr_state.is_left_following:
                    diff = np.linalg.inv(
                        SystemContext.vr_state.left_controller_start_pose) @ SystemContext.vr_state.left_controller_current_pose

                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.left_ee_start_pose[:3, :3]
                    left_T = SystemContext.vr_state.left_ee_start_pose @ diff_global
                    SystemContext.vr_state.left_hand_locked_pose = left_T
                else:
                    left_T = SystemContext.vr_state.left_hand_locked_pose

                torso_T = SystemContext.vr_state.torso_locked_pose

                if args.whole_body:
                    ctrl_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                        .set_joint_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset | left_reset | torso_reset)
                    )
                    ctrl_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    ctrl_builder.add_target("base", "link_right_arm_6", right_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)
                    ctrl_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                else:
                    torso_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6)
                        .set_joint_torque_limit([500] * 6)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(torso_reset)
                    )
                    right_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset)
                    )
                    left_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(left_reset)
                    )
                    torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    right_builder.add_target("base", "link_right_arm_6",
                                             right_T @ np.linalg.inv(Settings.T_hand_offset),
                                             2, np.pi * 2, 20, np.pi * 80)
                    left_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                    ctrl_builder = (
                        rby.BodyComponentBasedCommandBuilder()
                        .set_torso_command(torso_builder)
                        .set_right_arm_command(right_builder)
                        .set_left_arm_command(left_builder)
                    )

                component_command = (
                    rby.ComponentBasedCommandBuilder()
                    .set_mobility_command(
                        rby.SE2VelocityCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_velocity(-SystemContext.vr_state.mobile_linear_velocity,
                                      -SystemContext.vr_state.mobile_angular_velocity)
                        .set_minimum_time(Settings.dt * 1.01)
                    )
                    .set_body_command(ctrl_builder)
                )
                if head_tracker is not None and head_target is not None:
                    head_command = (
                        rby.JointPositionCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_position(head_target)
                        .set_velocity_limit(np.full(2, head_tracker.config.max_velocity_rad_s))
                        .set_acceleration_limit(np.full(2, head_tracker.config.max_acceleration_rad_s2))
                        .set_minimum_time(Settings.dt * 1.01)
                    )
                    component_command.set_head_command(head_command)

                torso_reset = False
                right_reset = False
                left_reset = False

                stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        component_command
                    )
                )
            except Exception as e:
                logging.error(f"Stream error: {e}")
                stream = None
                exit(1)

        # ...


if __name__ == "__main__":
    config = get_config()
    user_pc_ip = config['user_pc_ip']
    metaquest_ip = config['metaquest_ip']
    local_ip = config['local_ip']

    parser = argparse.ArgumentParser(description="RB-Y1 VR Control Launcher")

    parser.add_argument(
        "-s", "--server", type=str, default="tcp://*:5555",
        help="ZMQ server address for the UPC (default: tcp://*:5555)"
    )
    parser.add_argument(
        "--local_ip", required=False, default=local_ip,type=str,
        help="Local Wi-Fi (or LAN) IP address of the UPC"
    )
    parser.add_argument(
        "--meta_quest_ip", required=False, default=metaquest_ip, type=str,
        help=f"Wi-Fi (or LAN) IP address of the Meta Quest (default: {metaquest_ip})"
    )
    parser.add_argument(
        "--no_gripper", action="store_true",
        help="Run without gripper support"
    )
    parser.add_argument(
        "--rby1", default=user_pc_ip, type=str,
        help=f"RPC address of the RB-Y1 robot (default: {user_pc_ip})"
    )
    parser.add_argument(
        "--rby1_model", default="a", type=str,
        help="Model type of the RB-Y1 robot (default: a)"
    )
    parser.add_argument(
        "--no_head", action="store_true",
        help="Run without controlling the head"
    )
    parser.add_argument(
        "--whole_body", action="store_true",
        help="Use a whole-body optimization formulation (single control for all joints)"
    )
    parser.add_argument(
        "--data_collect", action="store_true",
        help="Enable H5 Writer and saved demo (default: False)"
    )
    parser.add_argument(
        "--no_data_collect", action="store_false", dest="data_collect",
        help="Disable H5 Writer and saved demo (default: True)"
    )
    quest_video_group = parser.add_mutually_exclusive_group()
    quest_video_group.add_argument(
        "--quest_video", action="store_true", dest="quest_video",
        help="Enable Meta Quest head-camera WebRTC streaming"
    )
    quest_video_group.add_argument(
        "--no_quest_video", action="store_false", dest="quest_video",
        help="Disable Meta Quest head-camera WebRTC streaming"
    )
    parser.set_defaults(quest_video=bool(Settings.quest_video.get("enabled", True)))
    parser.add_argument(
        "--dashboard_cmd_server", type=str, default="tcp://127.0.0.1:5556",
        help="Local ZMQ REP endpoint used by the notebook dashboard (default: tcp://127.0.0.1:5556)"
    )

    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
    finally:
        manager = SystemContext.recording_manager
        if manager is not None:
            try:
                manager.shutdown()
            except Exception as exc:
                logging.warning("Failed to shut down recording manager: %s", exc)
        for cleanup_func in reversed(SystemContext.cleanup_functions):
            try:
                cleanup_func()
            except Exception as exc:
                logging.warning("Cleanup failed: %s", exc)
        if isinstance(SystemContext.realsense, MultiRealsense):
            try:
                SystemContext.realsense.stop()
            except Exception as exc:
                logging.warning("Failed to stop camera during final cleanup: %s", exc)
