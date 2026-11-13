import numpy as np
# configure numpy to hide tiny residual floating point values (e.g. 6.12e-17 prints as 0)
np.set_printoptions(suppress=True, precision=4)

import math
from math import sin, cos

def robot_dynamics(theta1, theta2, theta3, theta4, theta5, theta6):
    # frame 0 to 1: rotation about z by theta1, then a translation of 243.3 mm along z
    T01 = np.array([
    [cos(theta1), -sin(theta1), 0, 0],
    [sin(theta1), cos(theta1), 0, 0],
    [0, 0, 1, 243.3],
    [0, 0, 0, 1]
    ])

    # frame 1 to 2: alpha turns the z axis horizontally, so joint 2 rotates in the vertical plane
    T12 = np.array([
    [cos((math.radians(-90) + theta2)), -sin((math.radians(-90) + theta2)), 0, 0],
    [0, 0, 1, 0],
    [-sin((math.radians(-90) + theta2)), -cos((math.radians(-90) + theta2)), 0, 0],
    [0, 0, 0, 1]
    ])

    # frame 2 to 3: upper arm, a = 200 mm along x,  alpha flips y and z axes (joint 3 axis is parallel to joint 2 but pointing the opposite way)
    T23 = np.array([
    [cos((math.radians(-90) + theta3)), -sin((math.radians(-90) + theta3)), 0, 200],
    [-sin((math.radians(-90) + theta3)), -cos((math.radians(-90) + theta3)), 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
    ])

    # frame 3 to 4: forearm offset, a = 87 mm along x and d = 227.6 mm (appears as -227.6 in the y entry because alpha = +90 deg maps the old z axis onto -y)
    T34 = np.array([
    [cos(theta4), -sin(theta4), 0, 87],
    [0, 0, -1, -227.6],
    [sin(theta4), cos(theta4), 0, 0],
    [0, 0, 0, 1]
    ])

    # frame 4 to 5: first wrist joint, pure rotation (a = 0, d = 0), with alpha = +90 deg so its axis is perpendicular to joint 4's
    T45 = np.array([
    [cos(theta5), -sin(theta5), 0, 0],
    [0, 0, -1, 0],
    [sin(theta5), cos(theta5), 0, 0],
    [0, 0, 0, 1]
    ])

    # frame 5 to 6: last wrist joint, alpha = -90 deg and d = 61.5 mm
    T56 = np.array([
    [cos(theta6), -sin(theta6), 0, 0],
    [0, 0, 1, 61.5],
    [-sin(theta6), -cos(theta6), 0, 0],
    [0, 0, 0, 1]
    ])

    # Chain the transforms: this gives the total transformation from the base (0) to frame 6
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

    return T06


def tip_position(q_deg):
    # conversion of the angles from degrees to radians
    T06 = robot_dynamics(*np.radians(q_deg))          
    # extracts the position (x, y, z) from the transformation matrix T06
    x, y, z = T06[:3, 3]
    # converts T06's 3x3 rotation matrix component to roll, pitch and yaw
    roll, pitch, yaw = rot_to_rpy(T06[:3, :3])
    roll, pitch, yaw = rot_to_rpy(T06[:3, :3])
    
    return np.array([x, y, z, roll, pitch, yaw])

def rot_to_rpy(R, eps=1e-6):
    # sin(pitch)
    sp = -R[2, 0]
    # cos(pitch)
    cp = np.hypot(R[0, 0], R[1, 0])
    
    pitch = np.arctan2(sp, cp)
    
    if cp > eps:
        # normal case, roll and yaw are read directly from the matrix entries
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        # roll and yaw rotate about the same axis, so only their combination is defined, based on the theoretical slides
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])

    return np.degrees([roll, pitch, yaw])