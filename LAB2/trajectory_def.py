import numpy as np

# global trajectory parameters
# total duration in seconds
T = 5.0

# reference yaw in degrees, transverse direction of the y axis
PSI = 0

def traj_base(t):

    # normalizes time to the interval [0, 1]
    s = np.clip(t / T, 0.0, 1.0)

    # goes from 0 to 1 with zero velocity at both ends, so signals driven by it start and finish without bumps
    s_smooth = 0.5 * (1.0 - np.cos(np.pi * s))

    # center of the LITE 6 Workspace - reference point around which the swing is built
    x_center = 320.0
    y_center = 0.0
    z_center = 140.0

    # lateral sweep
    y_offset = 220.0 * np.cos(np.pi * s)

    # depth x: half sine wave reaching peak forward extension at midpoint
    x_offset = 100.0 * np.sin(np.pi * s)

    # vertical profile - parabola
    z_height = z_center + 110.0 * (2.0 * s_smooth - 1.0)**2

    # final cartesian coordenates in mm
    x = x_center + x_offset
    y = y_center + y_offset
    z = z_height

    # stable orientation - avoids violent wrist rotations by keeping roll fixed and varying pitch and yaw smoothly
    roll = 180.0
    pitch = 10.0 * np.sin(np.pi * s)
    yaw = PSI + 20.0 * np.cos(np.pi * s)

    return np.array([x, y, z, roll, pitch, yaw])

ALPHA = -90.0

# Rotate by ALPHA to swing to the right part of the table
def traj(t):
    x, y, z, roll, pitch, yaw = traj_base(t)
    c, s = np.cos(np.radians(ALPHA)), np.sin(np.radians(ALPHA))
    yaw = (yaw + ALPHA + 180.0) % 360.0 - 180.0
    return np.array([c*x - s*y, s*x + c*y, z, roll, pitch, yaw])