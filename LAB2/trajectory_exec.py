import time
import numpy as np
from xarm.wrapper import XArmAPI
from trajectory_def import traj, T as T_SWING

# robot's IP address
IP = '192.168.1.158'

# time expansion factor
SLOW = 1.0

# robot's maximum velocity in mm/s
V_MAX = 200

# mm blending radius in mm
RADIUS = 5

# number of waypoints for the trajectory
N = 100

# execution real time
T = T_SWING * SLOW

# N time samples equally spaced out between [0, T]
ts = np.linspace(0, T, N)

# pose at each instant. t is divided by SLOW to go back to the trajectory's "nominal" time, the path in space is the same, only the pace changes
# P is initialized like (N, 6) - columns x, y, z, roll, pitch, yaw
P = np.array([traj(t / SLOW) for t in ts])
print(P)

# open connection to the robot through IP
arm = XArmAPI(IP)
# clear pending warnings from previous runs
arm.clean_warn()
# clear pending errors from previous runs
arm.clean_error()
# power the joints motors
arm.motion_enable(True)
# setting robot's mode and state
arm.set_mode(0)
arm.set_state(0)

# the robot positions itself slowly in the initial point
move = arm.set_position(*P[0], speed=50, mvacc=200, wait=True)
assert move == 0, f"Error setting robot into initial position: {move}"

# time step between waypoints
dt = ts[1] - ts[0]
# the real pose log will be stored in this array
log = []
# time reference for log
t0 = time.time()

for i in range(1, N):
    # distance between consecutive waypoints - the angles are left out because they are in different units
    dist = np.linalg.norm(P[i, :3] - P[i-1, :3])
    # average speed needed to cover the segment in dt seconds
    speed = float(np.clip(dist / dt, 20, V_MAX))
    
    # moving through trajectory - wait=FALSE keeps the robot moving without stopping between points - radius=RADIUS rounds corners into smooth curves
    move = arm.set_position(*P[i], speed=speed, mvacc=2000, radius=RADIUS, wait=False)
    
    # non zero code means the command was rejected
    if move != 0:
        print(f"Failed at {i}, code {move}")
        # state 4 stops the movement of the robot
        arm.set_state(4)
        break

# while the robot is moving, sample the pose at ~50Hz 
# get_position() returns (x, y, z, roll, pitch, yaw)
while arm.get_is_moving():
    log.append((time.time() - t0, *arm.get_position()[1]))
    time.sleep(0.02)
    
# compare the real duration with the expected one
print("Real duration:", time.time() - t0, "s (expected ≈", T, "s)")

# save the log for analysis
np.save("pose_real.npy", np.array(log))

# close connection to the robot
arm.disconnect()