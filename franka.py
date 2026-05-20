import numpy as np
import pylibfranka

robot = pylibfranka.Robot("100.20.0.119")

# First recover from the error
robot.automatic_error_recovery()

# Read CURRENT position before starting control
state = robot.read_once()

current_pose = np.array(state.O_T_EE).reshape(4, 4).T
print(current_pose)
# target = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# # Start control
# control = robot.start_joint_position_control(pylibfranka.ControllerMode.JointImpedance)

# steps = 5000  # 5 seconds — go slow!

# for i in range(steps):
#     state, duration = control.readOnce()

#     # Smooth s-curve interpolation (not linear!)
#     t = i / steps
#     t_smooth = t * t * (3 - 2 * t)  # smoothstep

#     q = current + t_smooth * (target - current)
#     control.writeOnce(pylibfranka.JointPositions(q.tolist()))

# import pylibfranka

# gripper = pylibfranka.Gripper("100.20.0.119")
# state = gripper.read_once()
# print(state.width)
# # gripper.move(0.08, 0.1)
# success = gripper.grasp(
#     width=0.05,
#     speed=0.02,
#     force=1,
#     epsilon_inner=0.01,
#     epsilon_outer=0.01,
# )

# if success:
#     print("Grasp successful")
# else:
#     print("Grasp failed")


# gripper.homing()
# gripper.move(0.08, 0.1)
# gripper.move(0.0, 0.1)
