import time

import numpy as np
import pylibfranka
from perception import Perception

ROBOT_IP = "100.20.0.119"

WORKSPACE = {
    "min_reach": 0.3,
    "max_reach": 0.855,
    "safe_reach": 0.85,
    "min_z": 0.0,
    "max_z": 1.2,
}

STEPS = 5000
HOVER_HEIGHT = 0.15  # clearance height for picking
PLACE_HOVER_HEIGHT = 0.05  # smaller hover for placing — less distance to fall
INITIAL_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# --- Poses ---
CUBE_TARGET_POSE = np.array(
    [
        [9.97498870e-01, 3.84481475e-02, -5.93102686e-02, 7.05253303e-01],
        [3.83621678e-02, -9.99260545e-01, -2.58801505e-03, -2.00000000e-01],
        [-5.93659133e-02, 3.06271540e-04, -9.98236239e-01, 1.019306814e-02],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

CUBE_PLACE_POSE = np.array(
    [
        [0.99997312, 0.00157861, 0.00715585, 0.70574057],
        [0.00200356, -0.99821001, -0.05977208, 0.23353049],
        [0.00704869, 0.05978481, -0.99818641, 0.01801525],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
AIRPODS_POSE = CUBE_TARGET_POSE.copy()
AIRPODS_POSE[1, 3] = CUBE_TARGET_POSE[1, 3] + 0.2
ROUND_POSE = CUBE_TARGET_POSE.copy()
ROUND_POSE[0, 3] = CUBE_TARGET_POSE[0, 3] - 0.1
ROUND_POSE[1, 3] = CUBE_TARGET_POSE[1, 3] - 0.1
ROUND_POSE[2, 3] = CUBE_TARGET_POSE[2, 3] - 0.0050


# Add more poses here as you capture them from the robot
# BOTTLE_TARGET_POSE = np.array([...])
# BOTTLE_PLACE_POSE  = np.array([...])

# --- Objects ---
OBJECTS = {
    "cube": {
        "target_pose": CUBE_TARGET_POSE,
        "place_pose": CUBE_PLACE_POSE,
        "width": 0.045,
        "force": 10.0,
        "epsilon_inner": 0.005,
        "epsilon_outer": 0.08,
        "max_drop_height": 0.005,
        "place_speed": 3.0,
        "object_height": 0.05,
        "container_height": 0.0,  # flat surface, no walls
    },
    "scissors": {
        "target_pose": ROUND_POSE,
        "place_pose": CUBE_PLACE_POSE,
        "width": 0.025,
        "force": 15.0,
        "epsilon_inner": 0.005,
        "epsilon_outer": 0.08,
        "max_drop_height": 0.02,
        "place_speed": 6.0,
        "object_height": 0.05,
        "container_height": 0.0,  # flat surface, no walls
    },
    "airpods": {
        "target_pose": AIRPODS_POSE,
        "place_pose": CUBE_PLACE_POSE,
        "width": 0.03,
        "force": 8.0,
        "epsilon_inner": 0.003,
        "epsilon_outer": 0.03,
        "max_drop_height": 0.001,
        "place_speed": 8.0,
        "object_height": 0.12,
        "container_height": 0.08,  # 8cm box walls — measure your actual box
    },
    # "bottle": {
    #     "target_pose":   BOTTLE_TARGET_POSE,
    #     "place_pose":    BOTTLE_PLACE_POSE,
    #     "width":         0.03,
    #     "force":         8.0,
    #     "epsilon_inner": 0.003,
    #     "epsilon_outer": 0.03,
    #     "max_drop_height": 0.005,
    #     "place_speed":   8.0,
    #     "object_height": 0.12,
    # },
}


class RobotController:
    def __init__(self, robot_ip):
        self.robot = pylibfranka.Robot(robot_ip)
        self.gripper = pylibfranka.Gripper(robot_ip)
        self.stack_height = 0.0

    def _read_joints(self):
        return np.array(self.robot.read_once().q)

    def _is_reachable(self, pose: np.ndarray) -> tuple[bool, str]:
        position = pose[:3, 3]
        distance = np.linalg.norm(position)
        if distance > WORKSPACE["max_reach"]:
            return False, f"Too far: {distance:.3f}m > {WORKSPACE['max_reach']}m"
        if distance > WORKSPACE["safe_reach"]:
            return (
                False,
                f"Near singularity: {distance:.3f}m > safe limit {WORKSPACE['safe_reach']}m",
            )
        if distance < WORKSPACE["min_reach"]:
            return False, f"Too close: {distance:.3f}m < {WORKSPACE['min_reach']}m"
        z = position[2]
        if z < WORKSPACE["min_z"]:
            return False, f"Below floor: Z={z:.3f}m"
        if z > WORKSPACE["max_z"]:
            return False, f"Too high: Z={z:.3f}m"
        R = pose[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
            return False, "Rotation matrix is not orthogonal"
        return True, "OK"

    def reset_arm(self):
        print("Moving to initial joint pose...")
        try:
            self.robot.automatic_error_recovery()
        except Exception:
            pass
        self.robot.set_collision_behavior(
            [20.0] * 7,
            [40.0] * 7,
            [10.0] * 6,
            [20.0] * 6,
        )
        current = self._read_joints()
        joint_control = self.robot.start_joint_position_control(
            pylibfranka.ControllerMode.JointImpedance
        )
        for i in range(STEPS):
            state, duration = joint_control.readOnce()
            t = i / STEPS
            t_smooth = t * t * (3 - 2 * t)
            q = current + t_smooth * (INITIAL_JOINTS - current)
            joint_control.writeOnce(pylibfranka.JointPositions(q))
        time.sleep(1.0)

    def move_to_cartesian(self, pose, wait=4.0, slip_threshold_n=None):
        """
        Move end-effector to pose over `wait` seconds.
        slip_threshold_n: if set, abort mid-move if vertical load drops below
                          this value — meaning the object slipped out mid-air.
        """
        reachable, reason = self._is_reachable(pose)
        if not reachable:
            raise SystemExit(f"SAFETY STOP: {reason}")

        try:
            self.robot.automatic_error_recovery()
        except Exception:
            pass

        steps = int(wait * 1000)
        cart_control = self.robot.start_cartesian_pose_control(
            pylibfranka.ControllerMode.CartesianImpedance
        )

        state, duration = cart_control.readOnce()
        start_pose = np.array(state.O_T_EE).reshape(4, 4).T

        for i in range(steps):
            state, duration = cart_control.readOnce()

            # Slip detection: check vertical load at every control step (~1kHz)
            if slip_threshold_n is not None:
                fz = abs(np.array(state.O_F_ext_hat_K)[2])
                if fz < slip_threshold_n:
                    print(
                        f"[Slip] Load dropped to {fz:.1f}N (threshold {slip_threshold_n:.1f}N) — object lost!"
                    )
                    break

            t = i / steps
            t_smooth = t * t * (3 - 2 * t)
            interp_pose = start_pose.copy()
            interp_pose[:3, 3] = start_pose[:3, 3] + t_smooth * (
                pose[:3, 3] - start_pose[:3, 3]
            )
            cart_control.writeOnce(
                pylibfranka.CartesianPose(interp_pose.T.flatten().tolist())
            )

        time.sleep(0.5)

    def _compliant_grasp(
        self, width: float, epsilon_inner: float, epsilon_outer: float
    ) -> float:
        """
        Two-attempt compliant grasp using gripper.grasp() correctly.

        gripper.grasp() already does continuous force-limited closing internally —
        it stops the moment the commanded force is reached or the width lands
        within epsilon. No step loop needed; we just use it twice:
          - Attempt 1: gentle force  → catches fragile objects before they break
          - Attempt 2: firm force    → only if attempt 1 missed entirely

        Deformation check: if the gripper closed more than DEFORM_LIMIT beyond
        the expected width, the object is soft/fragile — we stop at gentle force.
        """
        GENTLE_FORCE = 5.0  # N — safe for eggs, foam, bags
        FIRM_FORCE = 35.0  # N — for tools, cans, hard objects
        CLOSE_SPEED = 0.03  # m/s — slow closing gives more time to detect contact
        DEFORM_LIMIT = 0.004  # 4mm compression → fragile, don't increase force

        # --- Attempt 1: gentle ---
        print(f"[Grasp] Closing at {GENTLE_FORCE:.0f}N ...")
        self.gripper.grasp(
            width=width,
            speed=CLOSE_SPEED,
            force=GENTLE_FORCE,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )
        time.sleep(0.3)

        actual = self.gripper.state().width
        compression = width - actual

        if actual > 0.003:
            if compression > DEFORM_LIMIT:
                print(
                    f"[Grasp] Fragile — {compression * 1000:.1f}mm compression, holding at {GENTLE_FORCE:.0f}N"
                )
            else:
                print(
                    f"[Grasp] Contact at {actual * 1000:.1f}mm with {GENTLE_FORCE:.0f}N"
                )
            return GENTLE_FORCE

        # --- Attempt 2: missed — try firmer ---
        print(
            f"[Grasp] Missed (closed to {actual * 1000:.1f}mm), retrying at {FIRM_FORCE:.0f}N ..."
        )
        self.gripper.move(0.08, 0.05)
        time.sleep(0.2)

        self.gripper.grasp(
            width=width,
            speed=CLOSE_SPEED,
            force=FIRM_FORCE,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )
        time.sleep(0.3)

        actual = self.gripper.state().width
        if actual < 0.003:
            raise RuntimeError("[Grasp] Could not contact object at either force level")

        print(f"[Grasp] Contact at {actual * 1000:.1f}mm with {FIRM_FORCE:.0f}N")
        return FIRM_FORCE

    def pick(self, obj: dict):
        target_pose = obj["target_pose"]
        approach_pose = target_pose.copy()
        approach_pose[2, 3] = target_pose[2, 3] + HOVER_HEIGHT

        print("Stage 4: Approach above target")
        self.move_to_cartesian(approach_pose, wait=4.0)

        print("Stage 5: Descend to target")
        self.move_to_cartesian(target_pose, wait=3.0)

        print("Stage 6: Compliant grasp")
        used_force = self._compliant_grasp(
            obj["width"], obj["epsilon_inner"], obj["epsilon_outer"]
        )

        # Lift 2cm first to get a clean weight reading with object off the surface
        print("Stage 7: Weigh object")
        weigh_pose = target_pose.copy()
        weigh_pose[2, 3] += 0.02
        self.move_to_cartesian(weigh_pose, wait=1.0)
        state = self.robot.read_once()
        fz_baseline = abs(np.array(state.O_F_ext_hat_K)[2])
        print(
            f"[F/T] Object load: {fz_baseline:.2f}N (~{fz_baseline / 9.81 * 1000:.0f}g)"
        )

        # Full lift — slip detection active the whole way up
        print("Stage 8: Lift with slip detection")
        lift_pose = target_pose.copy()
        lift_pose[2, 3] += HOVER_HEIGHT
        slip_threshold = fz_baseline * 0.4  # alert if we lose >60% of the load
        self.move_to_cartesian(lift_pose, wait=3.0, slip_threshold_n=slip_threshold)

        print(f"Picking done! ({used_force:.0f}N, ~{fz_baseline / 9.81 * 1000:.0f}g)")

    def place(self, obj: dict):
        place_pose = obj["place_pose"].copy()
        place_pose[2, 3] += self.stack_height

        max_drop_height = obj["max_drop_height"]
        place_speed = obj["place_speed"]
        container_height = obj["container_height"]

        # hover above box walls before entering
        approach_pose = place_pose.copy()
        approach_pose[2, 3] = place_pose[2, 3] + container_height + PLACE_HOVER_HEIGHT

        print("Stage 8: Approach above container")
        self.move_to_cartesian(approach_pose, wait=place_speed)

        print("Stage 9: Lower to place surface")
        self.move_to_cartesian(place_pose, wait=place_speed)

        print("Stage 10: Release")
        self.gripper.move(0.08, 0.1)
        time.sleep(1.0)

        print("Stage 11: Lift out of container")
        exit_pose = place_pose.copy()
        exit_pose[2, 3] = place_pose[2, 3] + container_height + PLACE_HOVER_HEIGHT
        self.move_to_cartesian(exit_pose, wait=place_speed)

        self.stack_height += obj["object_height"]

        print("Placing done!")

    def pick_and_place(self):
        print("Stage 1: Move to initial pose")
        if not np.allclose(self._read_joints(), INITIAL_JOINTS, atol=0.01):
            self.reset_arm()
        else:
            print("Already at initial pose, skipping reset.")
        print("Stage 2: Gripper homing")
        self.gripper.homing()

        print("Stage 3: Open gripper")
        self.gripper.move(0.08, 0.1)

        for name, obj in OBJECTS.items():
            print(f"\n=== Handling: {name} ===")
            self.pick(obj)
            self.place(obj)

        print("\nAll objects done. Returning to initial pose.")
        self.reset_arm()
        self.stack_height = 0.0

    def pick_dynamic(
        self,
        perception: Perception,
        place_pose: np.ndarray,
        epsilon_inner=0.005,
        epsilon_outer=0.08,
    ):
        """Pick using a live AnyGrasp pose. Force is estimated from object width."""
        print("Stage 1: Move to initial pose")
        if not np.allclose(self._read_joints(), INITIAL_JOINTS, atol=0.01):
            self.reset_arm()

        print("Stage 2: Gripper homing + open")
        self.gripper.homing()
        self.gripper.move(0.08, 0.1)

        print("Stage 3: Querying AnyGrasp via Record3D...")
        target_pose, width = perception.get_best_grasp()
        if target_pose is None:
            print("No grasp found. Aborting.")
            return

        obj = {
            "target_pose": target_pose,
            "place_pose": place_pose,
            "width": width,
            "epsilon_inner": epsilon_inner,
            "epsilon_outer": epsilon_outer,
            "max_drop_height": 0.005,
            "place_speed": 4.0,
            "object_height": 0.05,
            "container_height": 0.0,
        }
        self.pick(obj)
        self.place(obj)
        self.reset_arm()


# -----------------------------------------------------------------------
# Camera-to-robot calibration (hand-eye).
# Replace this with your actual measured transform.
# Run a hand-eye calibration routine once and paste the result here.
# -----------------------------------------------------------------------
T_CAMERA_TO_ROBOT = np.array(
    [
        [1, 0, 0, 0.5],  # camera is ~50 cm in front of robot base (example)
        [0, -1, 0, 0.0],
        [0, 0, -1, 0.8],  # camera is ~80 cm above table
        [0, 0, 0, 1.0],
    ]
)


if __name__ == "__main__":
    import argparse
    from anygrasp_sdk import AnyGrasp
    from anygrasp_sdk.utils.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Path to .r3d file. Omit to use live USB stream.")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame index from the file (default: middle frame).")
    args = parser.parse_args()

    cfg      = load_config()
    anygrasp = AnyGrasp(cfg)
    anygrasp.load_net()

    if args.file:
        perception = Perception(
            anygrasp=anygrasp,
            camera_to_robot=T_CAMERA_TO_ROBOT,
            mode="file",
            file_path=args.file,
            frame_index=args.frame,
            depth_min=0.1,
            depth_max=1.2,
        )
    else:
        perception = Perception(
            anygrasp=anygrasp,
            camera_to_robot=T_CAMERA_TO_ROBOT,
            mode="stream",
            depth_min=0.1,
            depth_max=1.2,
        )

    controller = RobotController(ROBOT_IP)
    controller.pick_dynamic(perception, place_pose=CUBE_PLACE_POSE)
