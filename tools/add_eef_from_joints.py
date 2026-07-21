import argparse
import glob
import os
import sys
from pathlib import Path

# repo root를 python path에 추가
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from local_sdk_bootstrap import ensure_local_rby1_sdk_path
ensure_local_rby1_sdk_path()

import rby1_sdk as rby

from setup import Settings


def se3_to_pose7(T: np.ndarray) -> np.ndarray:
    """
    4x4 SE(3) pose를 7D pose로 변환.
    output: [x, y, z, qx, qy, qz, qw]
    """
    T = np.asarray(T, dtype=np.float64)
    xyz = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()  # scipy format: [qx, qy, qz, qw]
    return np.concatenate([xyz, quat], axis=0)


def pose7_to_se3(pose7: np.ndarray) -> np.ndarray:
    """
    7D pose [x, y, z, qx, qy, qz, qw]를 4x4 SE(3)로 변환.
    """
    pose7 = np.asarray(pose7, dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose7[:3]
    T[:3, :3] = R.from_quat(pose7[3:7]).as_matrix()

    return T


def compute_relative_action(
    cur_pose7: np.ndarray,
    next_pose7: np.ndarray,
    translation_frame: str = "base",
) -> np.ndarray:
    """
    현재 EEF pose에서 다음 EEF pose로 가는 relative action 계산.

    output:
        [dx, dy, dz, dRx, dRy, dRz]

    translation_frame:
        "base": translation delta를 base/world frame 기준으로 계산
        "local": translation delta를 현재 EEF frame 기준으로 계산
    """
    cur_T = pose7_to_se3(cur_pose7)
    next_T = pose7_to_se3(next_pose7)

    if translation_frame == "base":
        # base frame 기준 위치 변화량
        dpos = next_T[:3, 3] - cur_T[:3, 3]

    elif translation_frame == "local":
        # 현재 EEF frame 기준 위치 변화량
        T_rel = np.linalg.inv(cur_T) @ next_T
        dpos = T_rel[:3, 3]

    else:
        raise ValueError(f"Unknown translation_frame: {translation_frame}")

    # 회전 변화량은 현재 orientation에서 다음 orientation까지의 relative rotation
    R_cur = R.from_matrix(cur_T[:3, :3])
    R_next = R.from_matrix(next_T[:3, :3])
    drot = (R_cur.inv() * R_next).as_rotvec()

    return np.concatenate([dpos, drot], axis=0)


def make_dataset(group, name: str, data: np.ndarray, overwrite: bool = False):
    """
    H5 group 안에 dataset 저장.
    overwrite=True이면 기존 dataset 삭제 후 다시 생성.
    """
    if name in group:
        if overwrite:
            del group[name]
        else:
            print(f"[SKIP] Dataset already exists: {name}")
            return

    group.create_dataset(
        name,
        data=data,
        dtype="f8",
        compression="gzip",
        chunks=True,
    )
    print(f"[SAVE] {name}: shape={data.shape}")


def compute_eef_poses_from_joints(
    robot_model,
    joint_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RB-Y1 joint trajectory에서 right/left EEF absolute pose를 FK로 계산.

    input:
        joint_positions: shape (N, J)

    output:
        right_eef_poses: shape (N, 7)
        left_eef_poses : shape (N, 7)
    """
    dyn_robot = robot_model.get_dynamics()

    # main.py에서 쓰는 body frame 구성과 동일하게 맞춤
    dyn_state = dyn_robot.make_state(
        ["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
        robot_model.model().robot_joint_names,
    )

    base_link_idx = 0
    link_right_arm_6_idx = 2
    link_left_arm_6_idx = 3

    right_eef_poses = []
    left_eef_poses = []

    for i, q in enumerate(joint_positions):
        q = np.asarray(q, dtype=np.float64)

        try:
            dyn_state.set_q(q)
            dyn_robot.compute_forward_kinematics(dyn_state)

            right_T = (
                dyn_robot.compute_transformation(
                    dyn_state,
                    base_link_idx,
                    link_right_arm_6_idx,
                )
                @ Settings.T_hand_offset
            )

            left_T = (
                dyn_robot.compute_transformation(
                    dyn_state,
                    base_link_idx,
                    link_left_arm_6_idx,
                )
                @ Settings.T_hand_offset
            )

            right_eef_poses.append(se3_to_pose7(right_T))
            left_eef_poses.append(se3_to_pose7(left_T))

        except Exception as e:
            print(f"[WARN] FK failed at index {i}: {e}")
            right_eef_poses.append(np.full(7, np.nan, dtype=np.float64))
            left_eef_poses.append(np.full(7, np.nan, dtype=np.float64))

    return (
        np.asarray(right_eef_poses, dtype=np.float64),
        np.asarray(left_eef_poses, dtype=np.float64),
    )


def compute_actions_from_poses(
    eef_poses: np.ndarray,
    translation_frame: str = "base",
) -> np.ndarray:
    """
    absolute EEF pose sequence에서 relative EEF action sequence 생성.

    action[i] = pose[i] -> pose[i+1]
    마지막 frame은 다음 frame이 없으므로 0으로 채움.
    """
    n = len(eef_poses)
    actions = np.zeros((n, 6), dtype=np.float64)

    for i in range(n - 1):
        cur_pose = eef_poses[i]
        next_pose = eef_poses[i + 1]

        if np.any(np.isnan(cur_pose)) or np.any(np.isnan(next_pose)):
            actions[i] = np.full(6, np.nan, dtype=np.float64)
            continue

        actions[i] = compute_relative_action(
            cur_pose,
            next_pose,
            translation_frame=translation_frame,
        )

    # 마지막 action은 다음 pose가 없어서 0 유지
    actions[-1] = np.zeros(6, dtype=np.float64)

    return actions


def process_h5_file(
    h5_path: str,
    robot_model,
    overwrite: bool = False,
    translation_frame: str = "base",
):
    print(f"\n[PROCESS] {h5_path}")

    with h5py.File(h5_path, "a") as f:
        if "samples" not in f:
            print("[SKIP] No /samples group")
            return

        samples = f["samples"]

        if "robot_position" not in samples:
            print("[SKIP] No /samples/robot_position")
            return

        joint_positions = np.asarray(samples["robot_position"], dtype=np.float64)

        if joint_positions.ndim != 2:
            print(f"[SKIP] robot_position must be 2D, got shape={joint_positions.shape}")
            return

        print(f"[INFO] robot_position shape: {joint_positions.shape}")

        right_eef_pose, left_eef_pose = compute_eef_poses_from_joints(
            robot_model=robot_model,
            joint_positions=joint_positions,
        )

        right_eef_action = compute_actions_from_poses(
            right_eef_pose,
            translation_frame=translation_frame,
        )

        left_eef_action = compute_actions_from_poses(
            left_eef_pose,
            translation_frame=translation_frame,
        )

        make_dataset(samples, "right_eef_pose", right_eef_pose, overwrite=overwrite)
        make_dataset(samples, "left_eef_pose", left_eef_pose, overwrite=overwrite)
        make_dataset(samples, "right_eef_action", right_eef_action, overwrite=overwrite)
        make_dataset(samples, "left_eef_action", left_eef_action, overwrite=overwrite)

    print("[DONE]")


def collect_h5_files(path: str) -> list[str]:
    """
    path가 파일이면 그 파일만 처리.
    path가 폴더이면 하위의 모든 .h5 파일 처리.
    """
    if os.path.isfile(path) and path.endswith(".h5"):
        return [path]

    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.h5"), recursive=True))

    raise FileNotFoundError(f"Invalid path: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add RB-Y1 EEF pose/action datasets to existing H5 demos using FK."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="H5 file path or directory containing H5 files",
    )

    parser.add_argument(
        "--rby1",
        required=True,
        type=str,
        help="RB-Y1 robot address. Example: 192.168.x.x:50051",
    )

    parser.add_argument(
        "--rby1_model",
        default="a",
        type=str,
        help="RB-Y1 model type. Default: a",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing eef datasets if they already exist",
    )

    parser.add_argument(
        "--translation_frame",
        choices=["base", "local"],
        default="base",
        help="Frame for translation delta. Default: base",
    )

    args = parser.parse_args()

    print("[INFO] Creating RB-Y1 robot model...")
    robot = rby.create_robot(args.rby1, args.rby1_model)

    # FK model 접근을 위해 connect가 필요할 수 있음.
    # 단, 여기서는 power_on / servo_on / control_manager enable은 하지 않음.
    if not robot.connect():
        raise RuntimeError(f"Failed to connect to RB-Y1: {args.rby1}")

    robot_model = robot

    h5_files = collect_h5_files(args.input)
    print(f"[INFO] Found {len(h5_files)} h5 file(s).")

    for h5_path in h5_files:
        process_h5_file(
            h5_path=h5_path,
            robot_model=robot_model,
            overwrite=args.overwrite,
            translation_frame=args.translation_frame,
        )

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
