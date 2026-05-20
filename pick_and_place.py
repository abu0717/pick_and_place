import time

import numpy as np
import pylibfranka

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
        [ 0.99997312,  0.00157861,  0.00715585,  0.70574057],
        [ 0.00200356, -0.99821001, -0.05977208,  0.23353049],
        [ 0.00704869,  0.05978481, -0.99818641,  0.01801525],
        [ 0.0,         0.0,         0.0,         1.0       ],
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
        "container_height": 0.0,   # flat surface, no walls
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
        "container_height": 0.0,   # flat surface, no walls
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

    def move_to_cartesian(self, pose, wait=4.0):
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
            t = i / steps
            t_smooth = t * t * (3 - 2 * t)
            interp_pose = start_pose.copy()
            interp_pose[:3, 3] = start_pose[:3, 3] + t_smooth * (
                pose[:3, 3] - start_pose[:3, 3]
            )
            cart_control.writeOnce(
                pylibfranka.CartesianPose(interp_pose.T.flatten().tolist())
            )
            cart_control.readOnce()

        time.sleep(0.5)

    def pick(self, obj: dict):
        target_pose = obj["target_pose"]
        approach_pose = target_pose.copy()
        approach_pose[2, 3] = target_pose[2, 3] + HOVER_HEIGHT

        print("Stage 4: Approach above target")
        self.move_to_cartesian(approach_pose, wait=4.0)

        print("Stage 5: Descend to target")
        self.move_to_cartesian(target_pose, wait=3.0)

        print("Stage 6: Grasp")
        self.gripper.grasp(
            width=obj["width"],
            speed=0.1,
            force=obj["force"],
            epsilon_inner=obj["epsilon_inner"],
            epsilon_outer=obj["epsilon_outer"],
        )
        time.sleep(1.5)

        print("Stage 7: Lift")
        lift_pose = target_pose.copy()
        lift_pose[2, 3] += HOVER_HEIGHT
        self.move_to_cartesian(lift_pose, wait=3.0)

        print("Picking done!")

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


if __name__ == "__main__":
    controller = RobotController(ROBOT_IP)
    controller.pick_and_place()
