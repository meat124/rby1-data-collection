import logging
import threading
import rby1_sdk as rby
import socket
from typing import Union
import json
import numpy as np
import time
from setup import SystemContext, Settings
from helper import *
from utils import elbows_bending_check

def setup_meta_quest_udp_communication(local_ip: str, local_port: int, meta_quest_ip: str, meta_quest_port: int,
                                       power_off=None, video_enabled: bool = True):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        target_info = {
            "protocol_version": 2,
            "ip": local_ip,
            "port": local_port,
        }
        if video_enabled:
            video_config = Settings.quest_video
            signaling_host = video_config.get("advertise_host") or local_ip
            target_info["video"] = {
                "signaling_url": f"http://{signaling_host}:{video_config.get('port', 8080)}/offer",
                "codec": video_config.get("codec", "H264"),
            }
        message = json.dumps(target_info).encode('utf-8')
        sock.sendto(message, (meta_quest_ip, meta_quest_port))
        logging.info(f"Sent local PC info to Meta Quest: {target_info}")

    def udp_server():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
            server_sock.bind((local_ip, local_port))
            logging.info(f"UDP server running to receive Meta Quest Controller data... {local_ip}:{local_port}")
            while True:
                data, addr = server_sock.recvfrom(4096)
                udp_msg = data.decode('utf-8')
                try:
                    controller_state = json.loads(udp_msg)
                    SystemContext.vr_state.controller_state = controller_state
                    SystemContext.vr_state.controller_state_received_at = time.monotonic()
                    hands = controller_state.get("hands", {})
                    if "left" in hands:
                        buttons = hands["left"]["buttons"]
                        primary_button = buttons["primaryButton"]
                        secondary_button = buttons["secondaryButton"]

                        SystemContext.vr_state.event_left_a_pressed |= primary_button  #X button
                        SystemContext.vr_state.event_left_b_pressed |= secondary_button

                        # # HACK: record/stop trigger
                        # if primary_button: # NOTE: equivalent to 'X' in left controller
                        #     if power_off is not None:
                        #         logging.warning("Left X button pressed. Shutting down power.")
                        #         power_off()
                        #     pass

                    if "right" in hands:
                        buttons = hands["right"]["buttons"]
                        primary_button = buttons["primaryButton"]
                        secondary_button = buttons["secondaryButton"]

                        SystemContext.vr_state.event_right_a_pressed |= primary_button
                        SystemContext.vr_state.event_right_b_pressed |= secondary_button

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logging.warning(f"Failed to parse Meta Quest packet: {e} (from {addr}) - received data: {udp_msg[:100]}")

    thread = threading.Thread(target=udp_server, daemon=True)
    thread.start()

started = False
torso_mode = False  # Return-shape compatibility; HMD-driven torso tracking is disabled.
last_x_button_time = 0  # X 버튼 마지막 입력 시간 (debounce용)
X_BUTTON_DEBOUNCE_TIME = 0.5  # 0.5초 이상의 간격만 인정

def handle_vr_button_event(robot: Union[rby.Robot_A, rby.Robot_M], no_head: bool):
    global started, torso_mode, last_x_button_time
    model = robot.model()
    torso_dof = len(model.torso_idx)
    head_dof = len(model.head_idx)
    stream_reset_requested = False

    if SystemContext.vr_state.event_right_a_pressed:
        logging.info("Right A button pressed. Torso-locked ready mode initialized. Moving robot to ready pose.")
        if robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            robot.cancel_control()
        if robot.wait_for_control_ready(1000):
            # Waypoint logic (same as Bimanual Mode) for collision-safe init pose approach:
            #   elbows NOT bent (from zero)  → midpoint1 → midpoint2 → final init
            #   elbows bent     (task pose)  → (skip mp1) midpoint2  → final init
            skip_movej_due_to_elbow = elbows_bending_check(robot)

            if not started and not skip_movej_due_to_elbow and \
               Settings.right_arm_midpoint1 is not None and Settings.left_arm_midpoint1 is not None:
                movej(
                    robot,
                    np.zeros(torso_dof),
                    Settings.right_arm_midpoint1,
                    Settings.left_arm_midpoint1,
                    np.zeros(head_dof),
                    minimum_time=3,
                )
                started = True

            if Settings.right_arm_midpoint2 is not None and Settings.left_arm_midpoint2 is not None:
                movej(
                    robot,
                    np.zeros(torso_dof),
                    Settings.right_arm_midpoint2,
                    Settings.left_arm_midpoint2,
                    np.zeros(head_dof),
                    minimum_time=3,
                )
                started = True

            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(Settings.body_init_pose)
                    .set_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                    .set_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                    .set_minimum_time(3)
                )
            )

            started = True

            # If head is present, move head to zero pose on initialization
            if (not no_head) and getattr(SystemContext, "robot_model", None) is not None:
                try:
                    head_len = len(SystemContext.robot_model.head_idx)
                    if head_len > 0:
                        cbc.set_head_command(
                            rby.JointPositionCommandBuilder()
                            .set_position(np.deg2rad(Settings.torso_head_init_pose))
                            .set_minimum_time(3)
                        )
                except Exception as e:
                    logging.warning(f"Failed to set head zero pose on init: {e}")

            robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    cbc
                )
            ).get()
        torso_mode = False
        stream_reset_requested = True
        SystemContext.vr_state.is_initialized = True
        SystemContext.vr_state.is_stopped = False
        SystemContext.vr_state.head_recenter_requested = True

    elif SystemContext.vr_state.event_left_b_pressed:
        logging.info("Left Y button pressed. Bimanual Mode initialized. Moving robot to ready pose.")
        if robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            robot.cancel_control()
        if robot.wait_for_control_ready(1000):

            skip_movej_due_to_elbow = elbows_bending_check(robot)

            # Only execute movej if 'started' is False and we didn't skip due to elbow angle
            if not started and not skip_movej_due_to_elbow and \
               Settings.right_arm_midpoint1 is not None and Settings.left_arm_midpoint1 is not None:
                movej(
                    robot,
                    np.zeros(torso_dof),
                    Settings.right_arm_midpoint1,
                    Settings.left_arm_midpoint1,
                    np.zeros(head_dof),
                    minimum_time=3,
                )
                started = True


            if Settings.right_arm_midpoint2 is not None and Settings.left_arm_midpoint2 is not None:
                movej(
                    robot,
                    np.zeros(torso_dof),
                    Settings.right_arm_midpoint2,
                    Settings.left_arm_midpoint2,
                    np.zeros(head_dof),
                    minimum_time=3,
                )
                started = True

            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(Settings.body_init_pose)
                    .set_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                    .set_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                    .set_minimum_time(3)
                )
            )

            # If head is present, move head to zero pose on initialization
            if (not no_head) and getattr(SystemContext, "robot_model", None) is not None:
                try:
                    head_len = len(SystemContext.robot_model.head_idx)
                    if head_len > 0:
                        cbc.set_head_command(
                            rby.JointPositionCommandBuilder()
                            .set_position(np.deg2rad(Settings.bimanual_head_init_pose))
                            .set_minimum_time(3)
                        )
                except Exception as e:
                    logging.warning(f"Failed to set head zero pose on init: {e}")

            robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    cbc
                )
            ).get()
        torso_mode = False
        stream_reset_requested = True
        SystemContext.vr_state.is_initialized = True
        SystemContext.vr_state.is_stopped = False
        SystemContext.vr_state.head_recenter_requested = True

    elif SystemContext.vr_state.event_right_b_pressed:
        logging.info("Right B button pressed. Entering standby state.")
        # In data-collection mode, stop & save any active recording first so no data is lost.
        manager = getattr(SystemContext, "recording_manager", None)
        if manager is not None:
            if manager.request_stop_and_save():
                logging.info("Stop/save requested through recording manager before standby")
            else:
                logging.info("No active recording to stop; entering standby only")
        # Enter standby: the main control loop cancels the command stream, deactivates
        # teleop (is_initialized -> False), and holds the robot in place.
        # Press Right A again to re-initialize and resume teleop.
        SystemContext.vr_state.is_stopped = True
        stream_reset_requested = True

    elif SystemContext.vr_state.event_left_a_pressed:
        # Debounce: X 버튼 연타 방지 (0.5초 이상의 간격만 인정)
        current_time = time.time()
        if current_time - last_x_button_time < X_BUTTON_DEBOUNCE_TIME:
            logging.debug(f"X button debounced (interval: {current_time - last_x_button_time:.3f}s < {X_BUTTON_DEBOUNCE_TIME}s)")
        else:
            last_x_button_time = current_time
            manager = getattr(SystemContext, "recording_manager", None)
            if manager is not None:
                status = manager.get_status_snapshot()["status"]
                logging.info("Left X button pressed. Current recording status: %s", status)
                if manager.request_toggle_recording():
                    logging.info("Recording toggle accepted by recording manager")
                else:
                    logging.info("Recording toggle ignored in current state")
            else:
                logging.warning("Recording manager is not initialized; ignoring toggle request")

    else:
        return False, torso_mode

    SystemContext.vr_state.event_right_a_pressed = False
    SystemContext.vr_state.event_right_b_pressed = False
    SystemContext.vr_state.event_left_a_pressed = False
    SystemContext.vr_state.event_left_b_pressed = False

    return stream_reset_requested, torso_mode