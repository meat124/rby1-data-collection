"""Replay HDF5 teleop recordings to an RB-Y1 robot or simulator.

This script supports two modes:
 - Direct SDK commands (default): sends joint impedance / body cartesian commands via rby1_sdk.
 - ROS2 publish mode: when `--ros-topic` is provided, publishes sensor_msgs/JointState messages
   to the given topic and sleeps 1 second between frames (per your request).

Usage examples:
  # SDK mode
  python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051

  # ROS publish mode (1s sleep per frame)
  python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051 --ros-topic /rby1/joint_states

Note: ROS mode requires ROS2 and the `rclpy` Python package available in the environment.
"""

import argparse
import time
import logging
from pathlib import Path
import h5py
import numpy as np

import rby1_sdk as rby

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")


def clamp_and_fill(arr, last_valid):
    """Replace NaNs in arr with values from last_valid (if provided)."""
    if arr is None:
        return last_valid
    a = np.asarray(arr, dtype=float)
    if last_valid is None:
        if np.all(np.isnan(a)):
            return None
        nan_mask = np.isnan(a)
        if nan_mask.any():
            a[nan_mask] = 0.0
    else:
        nan_mask = np.isnan(a)
        if nan_mask.any():
            try:
                a[nan_mask] = last_valid[nan_mask]
            except Exception:
                a[nan_mask] = 0.0
    return a


def connect(address: str, model: str = "a"):
    logging.info(f"Connecting to RB-Y1 at {address} (model={model})")
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError("Failed to connect to RB-Y1")
    logging.info("Connected")
    try:
        robot.enable_control_manager(unlimited_mode_enabled=True)
    except Exception:
        logging.debug("enable_control_manager not available or failed")
    try:
        robot.start_state_update(lambda s: None, 30)
    except Exception:
        pass
    return robot


def replay_file(robot, h5path: Path, speed: float = 1.0, use_target_joints: bool = True,
                dt_min: float = 0.01, ros_topic: str = None, ros_node: Node = None,
                use_streams: bool = True):
    logging.info(f"Replaying {h5path} (speed={speed}, use_target_joints={use_target_joints}, ros_topic={ros_topic})")

    with h5py.File(h5path, "r") as f:
        grp = f.get("samples")
        if grp is None:
            raise ValueError("H5 file does not contain 'samples' group")

        times = grp["time"][()] if "time" in grp else None
        rp = grp["robot_position"][()] if "robot_position" in grp else None
        rtj = grp["robot_target_joints"][()] if "robot_target_joints" in grp else None
        gripper_state = grp["gripper_state"][()] if "gripper_state" in grp else None
        gripper_target = grp["gripper_target"][()] if "gripper_target" in grp else None

        n = int(times.shape[0]) if times is not None else (rp.shape[0] if rp is not None else (rtj.shape[0] if rtj is not None else 0))
        logging.info(f"Found {n} samples")

        last_cmd_pos = None
        last_grip = None

        # Create command stream (preferred) if available and requested
        stream = None
        if use_streams:
            try:
                stream = robot.create_command_stream()
            except Exception:
                stream = None

        # ROS publisher setup (if requested)
        ros_pub = None
        created_ros_node = False
        if ros_topic is not None:
            if not ROS_AVAILABLE:
                raise RuntimeError("ROS2 / rclpy not available in this Python environment")
            if ros_node is None:
                rclpy.init()
                ros_node = Node("rby1_replay_publisher")
                created_ros_node = True
            ros_pub = ros_node.create_publisher(JointState, ros_topic, 10)

        # helper to send a RobotCommandBuilder either via stream or sync; recreates stream if expired
        def send_command_with_retry(cmd_builder):
            nonlocal stream
            if stream is not None:
                try:
                    stream.send_command(cmd_builder)
                    return True
                except Exception as e:
                    msg = str(e).lower()
                    """Replay HDF5 teleop recordings to an RB-Y1 robot or simulator.

                    This script supports two modes:
                     - Direct SDK commands (default): sends joint impedance / body cartesian commands via rby1_sdk.
                     - ROS2 publish mode: when `--ros-topic` is provided, publishes sensor_msgs/JointState messages
                       to the given topic and sleeps 1 second between frames (per your request).

                    Usage examples:
                      # SDK mode
                      python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051

                      # ROS publish mode (1s sleep per frame)
                      python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051 --ros-topic /rby1/joint_states

                    Note: ROS mode requires ROS2 and the `rclpy` Python package available in the environment.
                    """

                    import argparse
                    import time
                    import logging
                    from pathlib import Path
                    import h5py
                    import numpy as np

                    import rby1_sdk as rby

                    try:
                        import rclpy
                        from rclpy.node import Node
                        from sensor_msgs.msg import JointState
                        ROS_AVAILABLE = True
                    except Exception:
                        ROS_AVAILABLE = False


                    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")


                    def clamp_and_fill(arr, last_valid):
                        """Replace NaNs in arr with values from last_valid (if provided)."""
                        if arr is None:
                            return last_valid
                        a = np.asarray(arr, dtype=float)
                        if last_valid is None:
                            if np.all(np.isnan(a)):
                                return None
                            nan_mask = np.isnan(a)
                            if nan_mask.any():
                                a[nan_mask] = 0.0
                        else:
                            nan_mask = np.isnan(a)
                            if nan_mask.any():
                                try:
                                    a[nan_mask] = last_valid[nan_mask]
                                except Exception:
                                    a[nan_mask] = 0.0
                        return a


                    def connect(address: str, model: str = "a"):
                        logging.info(f"Connecting to RB-Y1 at {address} (model={model})")
                        robot = rby.create_robot(address, model)
                        if not robot.connect():
                            raise RuntimeError("Failed to connect to RB-Y1")
                        logging.info("Connected")
                        try:
                            robot.enable_control_manager(unlimited_mode_enabled=True)
                        except Exception:
                            logging.debug("enable_control_manager not available or failed")
                        try:
                            robot.start_state_update(lambda s: None, 30)
                        except Exception:
                            pass
                        return robot


                    def replay_file(robot, h5path: Path, speed: float = 1.0, use_target_joints: bool = True,
                                    dt_min: float = 0.01, ros_topic: str = None, ros_node: Node = None,
                                    use_streams: bool = True):
                        logging.info(f"Replaying {h5path} (speed={speed}, use_target_joints={use_target_joints}, ros_topic={ros_topic})")

                        with h5py.File(h5path, "r") as f:
                            grp = f.get("samples")
                            if grp is None:
                                raise ValueError("H5 file does not contain 'samples' group")

                            times = grp["time"][()] if "time" in grp else None
                            rp = grp["robot_position"][()] if "robot_position" in grp else None
                            rtj = grp["robot_target_joints"][()] if "robot_target_joints" in grp else None
                            gripper_state = grp["gripper_state"][()] if "gripper_state" in grp else None
                            gripper_target = grp["gripper_target"][()] if "gripper_target" in grp else None

                            n = 0
                            if times is not None:
                                n = int(times.shape[0])
                            elif rp is not None:
                                n = int(rp.shape[0])
                            elif rtj is not None:
                                n = int(rtj.shape[0])

                            logging.info(f"Found {n} samples")

                            last_cmd_pos = None
                            last_grip = None

                            # Create command stream (preferred) if available and requested
                            stream = None
                            if use_streams:
                                try:
                                    stream = robot.create_command_stream()
                                except Exception:
                                    stream = None

                            # ROS publisher setup (if requested)
                            ros_pub = None
                            created_ros_node = False
                            if ros_topic is not None:
                                if not ROS_AVAILABLE:
                                    raise RuntimeError("ROS2 / rclpy not available in this Python environment")
                                if ros_node is None:
                                    rclpy.init()
                                    ros_node = Node("rby1_replay_publisher")
                                    created_ros_node = True
                                ros_pub = ros_node.create_publisher(JointState, ros_topic, 10)

                            # helper to send a RobotCommandBuilder either via stream or sync; recreates stream if expired
                            def send_command_with_retry(cmd_builder):
                                nonlocal stream
                                if use_streams and (stream is not None):
                                    try:
                                        stream.send_command(cmd_builder)
                                        return True
                                    except Exception as e:
                                        msg = str(e).lower()
                                        logging.warning(f"Stream send failed: {e}")
                                        if "expired" in msg or "stream is expired" in msg or "streamexpired" in msg:
                                            for attempt in range(3):
                                                try:
                                                    logging.info(f"Recreating expired command stream (attempt {attempt+1})")
                                                    try:
                                                        stream.cancel()
                                                    except Exception:
                                                        pass
                                                    time.sleep(0.05 * (attempt + 1))
                                                    try:
                                                        robot.wait_for_control_ready(100)
                                                    except Exception:
                                                        pass
                                                    new_stream = robot.create_command_stream()
                                                    if new_stream is not None:
                                                        stream = new_stream
                                                        stream.send_command(cmd_builder)
                                                        return True
                                                except Exception as e2:
                                                    logging.warning(f"Recreate/send attempt {attempt+1} failed: {e2}")
                                            logging.warning("All recreate attempts failed; falling back to synchronous send")
                                try:
                                    robot.send_command(cmd_builder).get()
                                    return True
                                except Exception as e:
                                    logging.warning(f"Synchronous send failed: {e}")
                                    return False

                            # iterate samples
                            for i in range(n):
                                t0 = float(times[i]) if times is not None else float(i) * dt_min

                                # Determine desired joint array (prefer targets)
                                desired = None
                                if use_target_joints and (rtj is not None):
                                    try:
                                        desired = rtj[i]
                                    except Exception:
                                        desired = rtj[()] if rtj is not None else None
                                if desired is None and (rp is not None):
                                    try:
                                        desired = rp[i]
                                    except Exception:
                                        desired = rp[()] if rp is not None else None

                                desired = clamp_and_fill(desired, last_cmd_pos)

                                # Gripper
                                grip_val = None
                                if gripper_target is not None:
                                    try:
                                        grip_val = gripper_target[i]
                                    except Exception:
                                        grip_val = gripper_target[()] if gripper_target is not None else None
                                elif gripper_state is not None:
                                    try:
                                        grip_val = gripper_state[i]
                                    except Exception:
                                        grip_val = gripper_state[()] if gripper_state is not None else None

                                # If requested, publish ROS JointState and sleep 1s per frame
                                if ros_topic is not None and ros_pub is not None:
                                    try:
                                        js = JointState()
                                        try:
                                            names = robot.model().robot_joint_names
                                            js.name = names
                                        except Exception:
                                            js.name = [f"joint_{k}" for k in range(len(desired))] if desired is not None else []

                                        js.position = ([] if desired is None else [float(x) for x in np.asarray(desired).reshape(-1).tolist()])
                                        js.header.stamp = ros_node.get_clock().now().to_msg()
                                        ros_pub.publish(js)
                                        logging.debug(f"Published JointState to {ros_topic} (sample {i})")
                                    except Exception as e:
                                        logging.warning(f"Failed to publish JointState for sample {i}: {e}")

                                    # per your request: 1 second sleep for each frame
                                    time.sleep(1.0)
                                    last_cmd_pos = np.array(desired, dtype=float) if desired is not None else last_cmd_pos
                                    continue

                                # SDK command mode: send joint impedance command OR cartesian body commands
                                if desired is not None:
                                    try:
                                        if use_target_joints and (rtj is not None):
                                            pos_list = [float(x) for x in np.asarray(desired).reshape(-1).tolist()]
                                            body_cmd = (
                                                rby.JointImpedanceControlCommandBuilder()
                                                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.1))
                                                .set_position(pos_list)
                                                .set_stiffness([80.0] * len(pos_list))
                                                .set_torque_limit([30.0] * len(pos_list))
                                                .set_minimum_time(max(dt_min, 0.01))
                                            )

                                            cbc = rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
                                            send_command_with_retry(rby.RobotCommandBuilder().set_command(cbc))
                                            last_cmd_pos = np.array(desired, dtype=float)
                                        else:
                                            # fallback: attempt to use cartesian targets if available
                                            if "robot_target_cartesian" in grp:
                                                try:
                                                    rtc_sample = grp["robot_target_cartesian"][i]
                                                except Exception:
                                                    rtc_sample = grp["robot_target_cartesian"][()]

                                                if rtc_sample is not None:
                                                    rtc_arr = np.asarray(rtc_sample).reshape(-1)
                                                    if rtc_arr.size >= 9:
                                                        def make_T(t_vec):
                                                            T = np.eye(4)
                                                            T[0:3, 3] = t_vec
                                                            return T

                                                        right_T = make_T(rtc_arr[0:3])
                                                        left_T = make_T(rtc_arr[3:6])
                                                        torso_T = make_T(rtc_arr[6:9])

                                                        torso_builder = (
                                                            rby.CartesianImpedanceControlCommandBuilder()
                                                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(dt_min * 10))
                                                            .set_minimum_time(max(dt_min, 0.01))
                                                            .set_joint_stiffness([400.0] * 6)
                                                            .set_joint_torque_limit([500.0] * 6)
                                                        )
                                                        right_builder = (
                                                            rby.CartesianImpedanceControlCommandBuilder()
                                                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(dt_min * 10))
                                                            .set_minimum_time(max(dt_min, 0.01))
                                                            .set_joint_stiffness([80.0] * 7)
                                                            .set_joint_torque_limit([30.0] * 7)
                                                        )
                                                        left_builder = (
                                                            rby.CartesianImpedanceControlCommandBuilder()
                                                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(dt_min * 10))
                                                            .set_minimum_time(max(dt_min, 0.01))
                                                            .set_joint_stiffness([80.0] * 7)
                                                            .set_joint_torque_limit([30.0] * 7)
                                                        )

                                                        torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                                                        right_builder.add_target("base", "link_right_arm_6", right_T, 2, np.pi * 2, 20, np.pi * 80)
                                                        left_builder.add_target("base", "link_left_arm_6", left_T, 2, np.pi * 2, 20, np.pi * 80)

                                                        ctrl_builder = (
                                                            rby.BodyComponentBasedCommandBuilder()
                                                            .set_torso_command(torso_builder)
                                                            .set_right_arm_command(right_builder)
                                                            .set_left_arm_command(left_builder)
                                                        )

                                                        stream_cmd = rby.ComponentBasedCommandBuilder().set_body_command(ctrl_builder)
                                                        send_command_with_retry(rby.RobotCommandBuilder().set_command(stream_cmd))
                                                        last_cmd_pos = np.array(desired, dtype=float)
                                                    else:
                                                        # not enough cartesian values, fall back to joint command
                                                        pos_list = [float(x) for x in np.asarray(desired).reshape(-1).tolist()]
                                                        body_cmd = (
                                                            rby.JointImpedanceControlCommandBuilder()
                                                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.1))
                                                            .set_position(pos_list)
                                                            .set_stiffness([80.0] * len(pos_list))
                                                            .set_torque_limit([30.0] * len(pos_list))
                                                            .set_minimum_time(max(dt_min, 0.01))
                                                        )
                                                        cbc = rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
                                                        send_command_with_retry(rby.RobotCommandBuilder().set_command(cbc))
                                                        last_cmd_pos = np.array(desired, dtype=float)
                                            else:
                                                # no cartesian data, fall back to joints
                                                pos_list = [float(x) for x in np.asarray(desired).reshape(-1).tolist()]
                                                body_cmd = (
                                                    rby.JointImpedanceControlCommandBuilder()
                                                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.1))
                                                    .set_position(pos_list)
                                                    .set_stiffness([80.0] * len(pos_list))
                                                    .set_torque_limit([30.0] * len(pos_list))
                                                    .set_minimum_time(max(dt_min, 0.01))
                                                )
                                                cbc = rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
                                                send_command_with_retry(rby.RobotCommandBuilder().set_command(cbc))
                                                last_cmd_pos = np.array(desired, dtype=float)
                                    except Exception as e:
                                        logging.warning(f"Failed to send joint/cartesian command for sample {i}: {e}")

                                # Gripper command handling (best-effort)
                                if grip_val is not None:
                                    try:
                                        if hasattr(grip_val, "tolist"):
                                            val = float(np.asarray(grip_val).reshape(-1)[0])
                                        else:
                                            val = float(grip_val)
                                        try:
                                            robot.set_tool_position("right", val)
                                        except Exception:
                                            pass
                                        last_grip = grip_val
                                    except Exception as e:
                                        logging.debug(f"Gripper command failed: {e}")

                                # compute sleep until next sample when using recorded timestamps
                                if i < n - 1 and times is not None:
                                    t_next = float(times[i + 1])
                                    sleep = max(0.0, (t_next - t0) / max(1e-6, speed))
                                    if sleep > 0:
                                        time.sleep(sleep)
                                else:
                                    time.sleep(dt_min)

                            # ensure final command is applied
                            time.sleep(0.1)
                            logging.info("Replay finished")

                            # cleanup temporary ROS node if we created one
                            if created_ros_node and ros_node is not None:
                                try:
                                    ros_node.destroy_node()
                                    rclpy.shutdown()
                                except Exception:
                                    pass


                    def parse_args():
                        p = argparse.ArgumentParser()
                        p.add_argument("--h5", required=True, help="Path to .h5 file (single)")
                        p.add_argument("--rby1", required=True, help="RB-Y1 gRPC address (e.g. 127.0.0.1:50051)")
                        p.add_argument("--model", default="a", help="Robot model (a or m)")
                        p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real time)")
                        p.add_argument("--no-targets", dest="use_targets", action="store_false", help="Do not use robot_target_joints, only robot_position")
                        p.add_argument("--dt-min", type=float, default=0.01, help="Minimum command time step (s)")
                        p.add_argument("--ros-topic", type=str, default=None, help="If provided, publish sensor_msgs/JointState to this ROS2 topic and sleep 1s per frame")
                        p.add_argument("--no-streams", dest="use_streams", action="store_false", help="Do not use command streams; send synchronously")
                        p.add_argument("--no-poweroff", dest="no_poweroff", action="store_true", help="Do not call robot.power_off() on exit")
                        return p.parse_args()


                    if __name__ == "__main__":
                        args = parse_args()
                        p = Path(args.h5)
                        if not p.exists():
                            raise SystemExit(f"H5 file not found: {p}")

                        robot = connect(args.rby1, args.model)
                        try:
                            # Let replay_file create a ROS node if needed
                            replay_file(robot, p, speed=args.speed, use_target_joints=args.use_targets, dt_min=args.dt_min,
                                        ros_topic=args.ros_topic, ros_node=None, use_streams=args.use_streams)
                        finally:
                            if not getattr(args, 'no_poweroff', False):
                                try:
                                    robot.power_off(".*")
                                except Exception:
                                    pass
import argparse
import time
import logging
from pathlib import Path
import h5py
import numpy as np
import rby1_sdk as rby


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")


def clamp_and_fill(arr, last_valid):
    """Replace NaNs in arr with values from last_valid (if provided)."""
    if arr is None:
        return last_valid
    a = np.asarray(arr, dtype=float)
    if last_valid is None:
        # if entire array is NaN, return None
        if np.all(np.isnan(a)):
            return None
        # replace NaNs with nearest non-nan in same array (forward fill/backfill)
        nan_mask = np.isnan(a)
        if nan_mask.any():
            # simple strategy: replace NaN with 0 where no last_known exists
            a[nan_mask] = 0.0
    else:
        nan_mask = np.isnan(a)
        if nan_mask.any():
            a[nan_mask] = last_valid[nan_mask]
    return a


def connect(address: str, model: str = "a"):
    logging.info(f"Connecting to RB-Y1 at {address} (model={model})")
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError("Failed to connect to RB-Y1")
    logging.info("Connected")
    # enable control manager like main does
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        logging.warning("Failed to enable control manager; continuing anyway")
    # start state update for internal caching (optional)
    try:
        robot.start_state_update(lambda s: None, 30)
    except Exception:
        pass
    return robot


def replay_file(robot, h5path: Path, speed: float = 1.0, use_target_joints: bool = True, dt_min: float = 0.01,
                ros_topic: str = None, ros_node: Node = None):
    logging.info(f"Replaying {h5path} (speed={speed}, use_target_joints={use_target_joints}, ros_topic={ros_topic})")
    with h5py.File(h5path, "r") as f:
        grp = f.get("samples")
        if grp is None:
            raise ValueError("H5 file does not contain 'samples' group")

        times = grp["time"][()] if "time" in grp else None
        rp = grp["robot_position"][()] if "robot_position" in grp else None
        rtj = grp["robot_target_joints"][()] if "robot_target_joints" in grp else None
        gripper_state = grp["gripper_state"][()] if "gripper_state" in grp else None
        gripper_target = grp["gripper_target"][()] if "gripper_target" in grp else None

        n = int(times.shape[0]) if times is not None else (rp.shape[0] if rp is not None else (rtj.shape[0] if rtj is not None else 0))
        logging.info(f"Found {n} samples")

        last_cmd_pos = None
        last_grip = None

        # Create command stream (preferred) if available
        try:
            stream = robot.create_command_stream()
        except Exception:
            stream = None

        # iterate samples
        for i in range(n):
            t0 = times[i] if times is not None else float(i) * dt_min
            # Determine desired joint array
            desired = None
            if use_target_joints and (rtj is not None):
                desired = rtj[i]
            if desired is None and (rp is not None):
                desired = rp[i]
            if use_target_joints and (rtj is not None):
                desired = rtj[i]
            if desired is None and (rp is not None):
                desired = rp[i]

            desired = clamp_and_fill(desired, last_cmd_pos)

            # Gripper
            grip_val = None
            if gripper_target is not None:
                grip_val = gripper_target[i]
            elif gripper_state is not None:
                grip_val = gripper_state[i]

            if desired is not None:
                # Send as JointImpedanceControlCommand (similar to main)
                try:
                    body_cmd = (
                        rby.JointImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.1))
                        .set_position([float(x) for x in desired.tolist()])
                        .set_stiffness([80.] * len(desired))
                        .set_torque_limit([30.] * len(desired))
                        .set_minimum_time(max(dt_min, 0.01))
                    )

                    cbc = rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)

                    if stream is not None:
                        stream.send_command(rby.RobotCommandBuilder().set_command(cbc))
                    else:
                        robot.send_command(rby.RobotCommandBuilder().set_command(cbc)).get()

                    last_cmd_pos = np.array(desired, dtype=float)
                except Exception as e:
                    logging.warning(f"Failed to send joint command for sample {i}: {e}")
            # If ros_topic is provided, publish JointState instead and sleep 1s per frame
            if ros_topic is not None and ros_node is not None:
                try:
                    pub = ros_node.get_publisher(JointState, ros_topic) if hasattr(ros_node, 'get_publisher') else None
                except Exception:
                    pub = None

                # create a local publisher if ros_node doesn't already hold one
                if pub is None:
                    pub = ros_node.create_publisher(JointState, ros_topic, 10)

                try:
                    js = JointState()
                    # use joint names from robot model if available, otherwise use numbered names
                    try:
                        names = robot.model().robot_joint_names
                        js.name = names
                    except Exception:
                        js.name = [f"joint_{k}" for k in range(len(desired))] if desired is not None else []

                    js.position = ([] if desired is None else [float(x) for x in desired.tolist()])
                    """Replay HDF5 teleop recordings to an RB-Y1 robot or simulator.

                    This script supports two modes:
                     - Direct SDK commands (default): sends joint impedance / body cartesian commands via rby1_sdk.
                     - ROS2 publish mode: when `--ros-topic` is provided, publishes sensor_msgs/JointState messages
                       to the given topic and sleeps 1 second between frames (per your request).

                    Usage examples:
                      # SDK mode
                      python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051

                      # ROS publish mode (1s sleep per frame)
                      python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051 --ros-topic /rby1/joint_states

                    Note: ROS mode requires ROS2 and the `rclpy` Python package available in the environment.
                    """

                    import argparse
                    import time
                    import logging
                    from pathlib import Path
                    import h5py
                    import numpy as np

                    import rby1_sdk as rby

                    try:
                        import rclpy
                        from rclpy.node import Node
                        from sensor_msgs.msg import JointState
                        ROS_AVAILABLE = True
                    except Exception:
                        ROS_AVAILABLE = False


                    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")


                    def clamp_and_fill(arr, last_valid):
                        """Replace NaNs in arr with values from last_valid (if provided)."""
                        if arr is None:
                            return last_valid
                        a = np.asarray(arr, dtype=float)
                        if last_valid is None:
                            if np.all(np.isnan(a)):
                                return None
                            nan_mask = np.isnan(a)
                            if nan_mask.any():
                                a[nan_mask] = 0.0
                        else:
                            nan_mask = np.isnan(a)
                            if nan_mask.any():
                                try:
                                    a[nan_mask] = last_valid[nan_mask]
                                except Exception:
                                    a[nan_mask] = 0.0
                        return a


                    def connect(address: str, model: str = "a"):
                        logging.info(f"Connecting to RB-Y1 at {address} (model={model})")
                        robot = rby.create_robot(address, model)
                        if not robot.connect():
                            raise RuntimeError("Failed to connect to RB-Y1")
                        logging.info("Connected")
                        try:
                            robot.enable_control_manager(unlimited_mode_enabled=True)
                        except Exception:
                            logging.debug("enable_control_manager not available or failed")
                        try:
                            robot.start_state_update(lambda s: None, 30)
                        except Exception:
                            pass
                        return robot


                    def replay_file(robot, h5path: Path, speed: float = 1.0, use_target_joints: bool = True,
                                    dt_min: float = 0.01, ros_topic: str = None, ros_node: Node = None,
                                    use_streams: bool = True):
                        logging.info(f"Replaying {h5path} (speed={speed}, use_target_joints={use_target_joints}, ros_topic={ros_topic})")

                        with h5py.File(h5path, "r") as f:
                            grp = f.get("samples")
                            if grp is None:
                                raise ValueError("H5 file does not contain 'samples' group")

                            times = grp["time"][()] if "time" in grp else None
                            rp = grp["robot_position"][()] if "robot_position" in grp else None
                            rtj = grp["robot_target_joints"][()] if "robot_target_joints" in grp else None
                            gripper_state = grp["gripper_state"][()] if "gripper_state" in grp else None
                            gripper_target = grp["gripper_target"][()] if "gripper_target" in grp else None

                            n = int(times.shape[0]) if times is not None else (rp.shape[0] if rp is not None else (rtj.shape[0] if rtj is not None else 0))
                            logging.info(f"Found {n} samples")

                            last_cmd_pos = None
                            last_grip = None

                            # Create command stream (preferred) if available and requested
                            stream = None
                            if use_streams:
                                try:
                                    stream = robot.create_command_stream()
                                except Exception:
                                    stream = None

                            # ROS publisher setup (if requested)
                            ros_pub = None
                            created_ros_node = False
                            if ros_topic is not None:
                                if not ROS_AVAILABLE:
                                    raise RuntimeError("ROS2 / rclpy not available in this Python environment")
                                if ros_node is None:
                                    rclpy.init()
                                    ros_node = Node("rby1_replay_publisher")
                                    created_ros_node = True
                                ros_pub = ros_node.create_publisher(JointState, ros_topic, 10)

                            # helper to send a RobotCommandBuilder either via stream or sync; recreates stream if expired
                            def send_command_with_retry(cmd_builder):
                                nonlocal stream
                                if stream is not None:
                                    try:
                                        stream.send_command(cmd_builder)
                                        return True
                                    except Exception as e:
                                        msg = str(e).lower()
                                        logging.warning(f"Stream send failed: {e}")
                                        if "expired" in msg or "stream is expired" in msg or "streamexpired" in msg:
                                            for attempt in range(3):
                                                try:
                                                    logging.info(f"Recreating expired command stream (attempt {attempt+1})")
                                                    try:
                                                        stream.cancel()
                                                    except Exception:
                                                        pass
                                                    time.sleep(0.05 * (attempt + 1))
                                                    try:
                                                        robot.wait_for_control_ready(100)
                                                    except Exception:
                                                        pass
                                                    new_stream = robot.create_command_stream()
                                                    if new_stream is not None:
                                                        stream = new_stream
                                                        stream.send_command(cmd_builder)
                                                        return True
                                                except Exception as e2:
                                                    logging.warning(f"Recreate/send attempt {attempt+1} failed: {e2}")
                                            logging.warning("All recreate attempts failed; falling back to synchronous send")
                                try:
                                    robot.send_command(cmd_builder).get()
                                    return True
                                except Exception as e:
                                    logging.warning(f"Synchronous send failed: {e}")
                                    return False

                            # iterate samples
                            for i in range(n):
                                t0 = times[i] if times is not None else float(i) * dt_min

                                # Determine desired joint array (prefer targets)
                                desired = None
                                if use_target_joints and (rtj is not None):
                                    desired = rtj[i]
                                if desired is None and (rp is not None):
                                    desired = rp[i]

                                desired = clamp_and_fill(desired, last_cmd_pos)

                                # Gripper
                                grip_val = None
                                if gripper_target is not None:
                                    grip_val = gripper_target[i]
                                elif gripper_state is not None:
                                    grip_val = gripper_state[i]

                                # If requested, publish ROS JointState and sleep 1s per frame
                                if ros_topic is not None and ros_pub is not None:
                                    try:
                                        js = JointState()
                                        try:
                                            names = robot.model().robot_joint_names
                                            js.name = names
                                        except Exception:
                                            js.name = [f"joint_{k}" for k in range(len(desired))] if desired is not None else []

                                        js.position = ([] if desired is None else [float(x) for x in desired.tolist()])
                                        """Replay HDF5 teleop recordings to an RB-Y1 robot or simulator.

                                        This script supports two modes:
                                         - Direct SDK commands (default): sends joint impedance / body cartesian commands via rby1_sdk.
                                         - ROS2 publish mode: when `--ros-topic` is provided, publishes sensor_msgs/JointState messages
                                           to the given topic and sleeps 1 second between frames (per your request).

                                        Usage examples:
                                          # SDK mode
                                          python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051

                                          # ROS publish mode (1s sleep per frame)
                                          python tools/replay_to_sim.py --h5 /media/nvidia/T7/Demo/demo_001.h5 --rby1 127.0.0.1:50051 --ros-topic /rby1/joint_states

                                        Note: ROS mode requires ROS2 and the `rclpy` Python package available in the environment.
                                        """

                                        import argparse
                                        import time
                                        import logging
                                        from pathlib import Path
                                        import h5py
                                        import numpy as np

                                        import rby1_sdk as rby

                                        try:
                                            import rclpy
                                            from rclpy.node import Node
                                            from sensor_msgs.msg import JointState
                                            ROS_AVAILABLE = True
                                        except Exception:
                                            ROS_AVAILABLE = False


                                        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")


                                        def clamp_and_fill(arr, last_valid):
                                            """Replace NaNs in arr with values from last_valid (if provided)."""
                                            if arr is None:
                                                return last_valid
                                            a = np.asarray(arr, dtype=float)
                                            if last_valid is None:
                                                if np.all(np.isnan(a)):
                                                    return None
                                                nan_mask = np.isnan(a)
                                                if nan_mask.any():
                                                    a[nan_mask] = 0.0
                                            else:
                                                nan_mask = np.isnan(a)
                                                if nan_mask.any():
                                                    try:
                                                        a[nan_mask] = last_valid[nan_mask]
                                                    except Exception:
                                                        a[nan_mask] = 0.0
                                            return a


                                        def connect(address: str, model: str = "a"):
                                            logging.info(f"Connecting to RB-Y1 at {address} (model={model})")
                                            robot = rby.create_robot(address, model)
                                            if not robot.connect():
                                                raise RuntimeError("Failed to connect to RB-Y1")
                                            logging.info("Connected")
                                            try:
                                                robot.enable_control_manager(unlimited_mode_enabled=True)
                                            except Exception:
                                                logging.debug("enable_control_manager not available or failed")
                                            try:
                                                robot.start_state_update(lambda s: None, 30)
                                            except Exception:
                                                pass
                                            return robot


                                        def replay_file(robot, h5path: Path, speed: float = 1.0, use_target_joints: bool = True,
                                                        dt_min: float = 0.01, ros_topic: str = None, ros_node: Node = None,
                                                        use_streams: bool = True):
                                            logging.info(f"Replaying {h5path} (speed={speed}, use_target_joints={use_target_joints}, ros_topic={ros_topic})")

                                            with h5py.File(h5path, "r") as f:
                                                grp = f.get("samples")
                                                if grp is None:
                                                    raise ValueError("H5 file does not contain 'samples' group")

                                                times = grp["time"][()] if "time" in grp else None
                                                rp = grp["robot_position"][()] if "robot_position" in grp else None
                                                rtj = grp["robot_target_joints"][()] if "robot_target_joints" in grp else None
                                                gripper_state = grp["gripper_state"][()] if "gripper_state" in grp else None
                                                gripper_target = grp["gripper_target"][()] if "gripper_target" in grp else None

                                                n = int(times.shape[0]) if times is not None else (rp.shape[0] if rp is not None else (rtj.shape[0] if rtj is not None else 0))
                                                logging.info(f"Found {n} samples")

                                                last_cmd_pos = None
                                                last_grip = None

                                                # Create command stream (preferred) if available and requested
                                                stream = None
                                                if use_streams:
                                                    try:
                                                        stream = robot.create_command_stream()
                                                    except Exception:
                                                        stream = None

                                                # ROS publisher setup (if requested)
                                                ros_pub = None
                                                created_ros_node = False
                                                if ros_topic is not None:
                                                    if not ROS_AVAILABLE:
                                                        raise RuntimeError("ROS2 / rclpy not available in this Python environment")
                                                    if ros_node is None:
                                                        rclpy.init()
                                                        ros_node = Node("rby1_replay_publisher")
                                                        created_ros_node = True
                                                    ros_pub = ros_node.create_publisher(JointState, ros_topic, 10)

                                                # helper to send a RobotCommandBuilder either via stream or sync; recreates stream if expired
                                                def send_command_with_retry(cmd_builder):
                                                    nonlocal stream
                                                    if stream is not None:
                                                        try:
                                                            stream.send_command(cmd_builder)
                                                            return True
                                                        except Exception as e:
                                                            msg = str(e).lower()
                                                            logging.warning(f"Stream send failed: {e}")
                                                            if "expired" in msg or "stream is expired" in msg or "streamexpired" in msg:
                                                                for attempt in range(3):
                                                                    try:
                                                                        logging.info(f"Recreating expired command stream (attempt {attempt+1})")
                                                                        try:
                                                                            stream.cancel()
                                                                        except Exception:
                                                                            pass
                                                                        time.sleep(0.05 * (attempt + 1))
                                                                        try:
                                                                            robot.wait_for_control_ready(100)
                                                                        except Exception:
                                                                            pass
                                                                        new_stream = robot.create_command_stream()
                                                                        if new_stream is not None:
                                                                            stream = new_stream
                                                                            stream.send_command(cmd_builder)
                                                                            return True
                                                                    except Exception as e2:
                                                                        logging.warning(f"Recreate/send attempt {attempt+1} failed: {e2}")
                                                                logging.warning("All recreate attempts failed; falling back to synchronous send")
                                                    try:
                                                        robot.send_command(cmd_builder).get()
                                                        return True
                                                    except Exception as e:
                                                        logging.warning(f"Synchronous send failed: {e}")
                                                        return False

                                                # iterate samples
                                                for i in range(n):
                                                    t0 = times[i] if times is not None else float(i) * dt_min

                                                    # Determine desired joint array (prefer targets)
                                                    desired = None
                                                    if use_target_joints and (rtj is not None):
                                                        desired = rtj[i]
                                                    if desired is None and (rp is not None):
                                                        desired = rp[i]

                                                    desired = clamp_and_fill(desired, last_cmd_pos)

                                                    # Gripper
                                                    grip_val = None
                                                    if gripper_target is not None:
                                                        grip_val = gripper_target[i]
                                                    elif gripper_state is not None:
                                                        grip_val = gripper_state[i]

                                                    # If requested, publish ROS JointState and sleep 1s per frame
                                                    if ros_topic is not None and ros_pub is not None:
                                                        try:
                                                            js = JointState()
                                                            try:
                                                                names = robot.model().robot_joint_names
                                                                js.name = names
                                                            except Exception:
                                                                js.name = [f"joint_{k}" for k in range(len(desired))] if desired is not None else []

                                                            js.position = ([] if desired is None else [float(x) for x in desired.tolist()])
                                                            js.header.stamp = ros_node.get_clock().now().to_msg()
                                                            ros_pub.publish(js)
                                                            logging.debug(f"Published JointState to {ros_topic} (sample {i})")
                                                        except Exception as e:
                                                            logging.warning(f"Failed to publish JointState for sample {i}: {e}")

                                                        # per your request: 1 second sleep for each frame
                                                        time.sleep(1.0)
                                                        last_cmd_pos = np.array(desired, dtype=float) if desired is not None else last_cmd_pos
                                                        continue

                                                    # SDK command mode: send joint impedance command OR cartesian body commands
                                                    if desired is not None:
                                                        try:
                                                            if use_target_joints and (rtj is not None):
                                                                body_cmd = (
                                                                    rby.JointImpedanceControlCommandBuilder()
                                                                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.1))
                                                                    .set_position([float(x) for x in desired.tolist()])
                                                                    .set_stiffness([80.] * len(desired))
                                                                    .set_torque_limit([30.] * len(desired))
                                                                    .set_minimum_time(max(dt_min, 0.01))
                                                                )

                                                                cbc = rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
                                                                send_command_with_retry(rby.RobotCommandBuilder().set_command(cbc))
                                                                last_cmd_pos = np.array(desired, dtype=float)
                                                            else:
                                                                # fallback: attempt to use cartesian targets if available
                                                                if "robot_target_cartesian" in grp:
                                                                    try:
                                                                        rtc_sample = grp["robot_target_cartesian"][i]
                                                                    except Exception:
                                                                        rtc_sample = grp["robot_target_cartesian"][()]

                                                                    if rtc_sample is not None:
                                                                        rtc_arr = np.asarray(rtc_sample).reshape(-1)
                                                                        if rtc_arr.size >= 9:
                                                                            def make_T(t_vec):
                                                                                T = np.eye(4)
                                                                                T[0:3, 3] = t_vec
                                                                                return T

                                                                            right_T = make_T(rtc_arr[0:3])
                                                                            left_T = make_T(rtc_arr[3:6])
                                                                            torso_T = make_T(rtc_arr[6:9])

                                                                            torso_builder = (
                                                                                rby.CartesianImpedanceControlCommandBuilder()
                                                                                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(dt_min * 10))
                                                                                .set_minimum_time(max(dt_min, 0.01))
                                                                                .set_joint_stiffness([400.] * 6)
                                                                                .set_joint_torque_limit([500.] * 6)
                                                                            )
                                                                            right_builder = (
                                                                                rby.CartesianImpedanceControlCommandBuilder()
                                                                                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(dt_min * 10))
                                                                                .set_minimum_time(max(dt_min, 0.01))
                                                                                .set_joint_stiffness([80.] * 7)
                                                                                .set_joint_torque_limit([30.] * 7)
                                                                            )
                                                                            left_builder = (
                                                                                rby.CartesianImpedanceControlCommandBuilder()
                                                                                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(dt_min * 10))
                                                                                .set_minimum_time(max(dt_min, 0.01))
                                                                                .set_joint_stiffness([80.] * 7)
                                                                                .set_joint_torque_limit([30.] * 7)
                                                                            )

                                                                            torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                                                                            right_builder.add_target("base", "link_right_arm_6", right_T, 2, np.pi * 2, 20, np.pi * 80)
                                                                            left_builder.add_target("base", "link_left_arm_6", left_T, 2, np.pi * 2, 20, np.pi * 80)

                                                                            ctrl_builder = (
                                                                                rby.BodyComponentBasedCommandBuilder()
                                                                                .set_torso_command(torso_builder)
                                                                                .set_right_arm_command(right_builder)
                                                                                .set_left_arm_command(left_builder)
                                                                            )

                                                                            stream_cmd = rby.ComponentBasedCommandBuilder().set_body_command(ctrl_builder)
                                                                            send_command_with_retry(rby.RobotCommandBuilder().set_command(stream_cmd))
                                                                            last_cmd_pos = np.array(desired, dtype=float)
                                                                else:
                                                                    # no cartesian data, fall back to joints
                                                                    body_cmd = (
                                                                        rby.JointImpedanceControlCommandBuilder()
                                                                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.1))
                                                                        .set_position([float(x) for x in desired.tolist()])
                                                                        .set_stiffness([80.] * len(desired))
                                                                        .set_torque_limit([30.] * len(desired))
                                                                        .set_minimum_time(max(dt_min, 0.01))
                                                                    )
                                                                    cbc = rby.ComponentBasedCommandBuilder().set_body_command(body_cmd)
                                                                    send_command_with_retry(rby.RobotCommandBuilder().set_command(cbc))
                                                                    last_cmd_pos = np.array(desired, dtype=float)
                                                        except Exception as e:
                                                            logging.warning(f"Failed to send joint/cartesian command for sample {i}: {e}")

                                                    # Gripper command handling (best-effort)
                                                    if grip_val is not None:
                                                        try:
                                                            if hasattr(grip_val, "tolist"):
                                                                val = float(np.asarray(grip_val).reshape(-1)[0])
                                                            else:
                                                                val = float(grip_val)
                                                            try:
                                                                robot.set_tool_position("right", val)
                                                            except Exception:
                                                                pass
                                                            last_grip = grip_val
                                                        except Exception as e:
                                                            logging.debug(f"Gripper command failed: {e}")

                                                    # compute sleep until next sample when using recorded timestamps
                                                    if i < n - 1 and times is not None:
                                                        t_next = times[i + 1]
                                                        sleep = max(0.0, (t_next - t0) / max(1e-6, speed))
                                                        if sleep > 0:
                                                            time.sleep(sleep)
                                                    else:
                                                        time.sleep(dt_min)

                                                # ensure final command is applied
                                                time.sleep(0.1)
                                                logging.info("Replay finished")

                                                # cleanup temporary ROS node if we created one
                                                if created_ros_node and ros_node is not None:
                                                    try:
                                                        ros_node.destroy_node()
                                                        rclpy.shutdown()
                                                    except Exception:
                                                        pass


                                        def parse_args():
                                            p = argparse.ArgumentParser()
                                            p.add_argument("--h5", required=True, help="Path to .h5 file (single)")
                                            p.add_argument("--rby1", required=True, help="RB-Y1 gRPC address (e.g. 127.0.0.1:50051)")
                                            p.add_argument("--model", default="a", help="Robot model (a or m)")
                                            p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real time)")
                                            p.add_argument("--no-targets", dest="use_targets", action="store_false", help="Do not use robot_target_joints, only robot_position")
                                            p.add_argument("--dt-min", type=float, default=0.01, help="Minimum command time step (s)")
                                            p.add_argument("--ros-topic", type=str, default=None, help="If provided, publish sensor_msgs/JointState to this ROS2 topic and sleep 1s per frame")
                                            p.add_argument("--no-streams", dest="use_streams", action="store_false", help="Do not use command streams; send synchronously")
                                            p.add_argument("--no-poweroff", dest="no_poweroff", action="store_true", help="Do not call robot.power_off() on exit")
                                            return p.parse_args()


                                        if __name__ == "__main__":
                                            args = parse_args()
                                            p = Path(args.h5)
                                            if not p.exists():
                                                raise SystemExit(f"H5 file not found: {p}")

                                            robot = connect(args.rby1, args.model)
                                            try:
                                                ros_node = None
                                                if args.ros_topic is not None:
                                                    if not ROS_AVAILABLE:
                                                        raise SystemExit("ROS2 / rclpy not available in this Python environment")
                                                    ros_node = None
                                                replay_file(robot, p, speed=args.speed, use_target_joints=args.use_targets, dt_min=args.dt_min, ros_topic=args.ros_topic, ros_node=ros_node, use_streams=args.use_streams)
                                            finally:
                                                if not getattr(args, 'no_poweroff', False):
                                                    try:
                                                        robot.power_off(".*")
                                                    except Exception:
                                                        pass
                        p.add_argument("--no-poweroff", dest="no_poweroff", action="store_true", help="Do not call robot.power_off() on exit")
                        return p.parse_args()


                    if __name__ == "__main__":
                        args = parse_args()
                        p = Path(args.h5)
                        if not p.exists():
                            raise SystemExit(f"H5 file not found: {p}")

                        robot = connect(args.rby1, args.model)
                        try:
                            ros_node = None
                            if args.ros_topic is not None:
                                if not ROS_AVAILABLE:
                                    raise SystemExit("ROS2 / rclpy not available in this Python environment")
                                ros_node = None
                            replay_file(robot, p, speed=args.speed, use_target_joints=args.use_targets, dt_min=args.dt_min, ros_topic=args.ros_topic, ros_node=ros_node, use_streams=args.use_streams)
                        finally:
                            if not getattr(args, 'no_poweroff', False):
                                try:
                                    robot.power_off(".*")
                                except Exception:
                                    pass