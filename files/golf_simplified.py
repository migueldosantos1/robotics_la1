import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("golf_simplified.xml")
data = mujoco.MjData(model)


# ID-ing joints
fold_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "torso_fold")
rotate_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "torso_rotate")
arm1_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm1")

# ID-ing actuators
fold_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fold_motor")
rotate_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotate_motor")
arm1_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm1_motor")

# Angle conversion (MuJoCo works in rad)
def deg(angle):
    return np.deg2rad(angle)

# Defining start and end angles
fold_start = deg(0)
fold_peak = deg(-10)
fold_end = deg(0) 

rotate_start = deg(-30)
rotate_end = deg(45)

arm1_start = deg(10)
arm1_peak = deg(-50)
arm1_end = deg(10)

# Degining intervals (seconds)
movement_time = 0.8
start_delay = 5
final_delay = 10

total_time = movement_time + final_delay

# Ease in and ease out
# A more aggressive function would be fun, especially in the torso rotation
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (x * (x * 6 - 15) + 10)

def smooth_move(start, end, progress):
    s = smoothstep(progress)
    return start + (end - start) * s

# Relative time
real_start = time.perf_counter()

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Camera position
    viewer.cam.lookat[:] = [1, 1, 2]

    # Distance from the center point
    viewer.cam.distance = 5

    # Camera angles
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -10

    while viewer.is_running():

        elapsed = time.perf_counter() - real_start

        # Before the starting delay, keep as initial
        if elapsed < start_delay:
            fold_target = fold_start
            rotate_target = rotate_start
            arm1_target = arm1_start

        # After the starting delay, the actual movement starts
        else:
            t = elapsed - start_delay

            if t < (movement_time / 2):
                progress = t / (movement_time / 2)

                fold_target = smooth_move(fold_start, fold_peak, progress)
                arm1_target = smooth_move(arm1_start, arm1_peak, progress)

            elif t < (movement_time / 2 + movement_time / 2):
                progress = (t - movement_time / 2) / (movement_time / 2)

                fold_target = smooth_move(fold_peak, fold_end, progress)
                arm1_target = smooth_move(arm1_peak, arm1_end, progress)

            else:
                fold_target = fold_end
                arm1_target = arm1_end

            if t < movement_time:
                progress = t / movement_time
                rotate_target = smooth_move(rotate_start, rotate_end, progress)

            else:
                rotate_target = rotate_end

            # Stop when total time is reached
            if t >= total_time:
                
                mujoco.mj_step(model, data)
                viewer.sync()

                break

        # Sending the target values to the motor
        data.ctrl[fold_motor] = fold_target
        data.ctrl[rotate_motor] = rotate_target
        data.ctrl[arm1_motor] = arm1_target

        mujoco.mj_step(model, data)
        viewer.sync()

        # Run approximately in real time
        time.sleep(model.opt.timestep)