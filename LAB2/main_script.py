"""
Important Information:

Run in the lab:       python main_script.py
Swing only:           python main_script.py --traj-only   (skips steps 1 and 2, including the IK safety check)
Analyze at home:      python main_script.py --analyze     (uses ik_data.npz, no robot needed)
 
Files produced:
  ik_data.npz          desired poses + joint angles returned by the robot's IK
  ik_errors.npz        IK -> FK round-trip errors (Task 4)
  fk_validation.npy    our FK vs the robot's reported pose, per configuration (Task 1)
  traj_run_slow<k>.npz desired waypoints + real log of each swing run (input of post_processing)
"""

import sys, time, threading
import numpy as np
from xarm.wrapper import XArmAPI
# our FK: joint angles in degrees -> [x, y, z, roll, pitch, yaw]
from matrices import tip_position
# desired pose function traj(t) and nominal duration T
from trajectory_def import traj, T

# robot IP address
IP    = '192.168.1.158'
# number of trajectory points used for the Task 4 IK test
N_IK  = 100
# deg/s, joint speed during the FK validation moves
SPEED = 20
# time stretching factors
SLOW_LEVELS = [1]
# number of waypoints sent to the robot for the swing
N_WP    = 20
# mm/s, TCP speed ceiling
V_MAX   = 500
# mm/s, minimum speed sent to set_position()
V_MIN   = 10
# mm, blending radius between consecutive segments
RADIUS  = 5
# largest joint change between consecutive IK solutions that we accept as normal
MAX_JUMP = 20
# joint configurations for the FK validation
CONFIGS = [
    [0, 0, 0, 0, 0, 0],
    [30, 20, 40, 0, 20, 15],
    [-30, 30, 60, 20, 40, -20],
    [20, 40, 60, -20, 50, 30],
    [0, 30, 30, 0, 30, 0],
]

def wrap(a):
    # wraps angles in degrees to [-180, 180], so an error of 359 degrees reads as -1 degrees
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0

def pose_error(p_des, p_calc):
    # calculates pose error (p_calc - p_des) with position in mm and orientation degrees wrapped to [-180, 180]
    e = np.asarray(p_calc, float) - np.asarray(p_des, float)
    e[3:] = wrap(e[3:])
    return e

# computes controller IK along the trajectory, saves to ik_data.npz, and verifies joint continuity
def collect_ik(arm):
    # samples trajectory geometry uniformly over nominal duration
    ts = np.linspace(0, T, N_IK)
    poses = np.array([traj(t) for t in ts])
    q_ik, codes = [], []
    
    for pose in poses:
        # query controller IK (angles in degrees, code 0 = success)
        code, q = arm.get_inverse_kinematics([float(v) for v in pose], input_is_radian=False, return_is_radian=False)
        codes.append(code)
        
        # store first 6 joints, using NaNs on failure to maintain index alignment
        q_ik.append(list(q[:6]) if code == 0 else [np.nan] * 6)
        if code != 0:
            print("IK failed (code", code, ") at", np.round(pose, 1))

    q_ik = np.array(q_ik)
    q_ik = np.unwrap(q_ik, axis=0)
    np.savez("ik_data.npz", ts=ts, poses=poses, q_ik=q_ik, codes=np.array(codes))
    print("Saved ik_data.npz")
    
    # validate that all points solved and consecutive joint jumps do not exceed MAX_JUMP
    ok = all(c == 0 for c in codes)
    jump = np.nanmax(np.abs(np.diff(q_ik, axis=0))) if ok else np.nan
    
    print(f"Check: IK ok across all points: {ok} | Max joint jump: {jump:.1f}°")
    
    return ok and jump <= MAX_JUMP

# evaluates controller IK accuracy by performing an IK-to-FK round trip on saved data
def analyze_ik(path="ik_data.npz"):
    d = np.load(path)
    # FK applied to controller IK joint angles
    calc = np.array([tip_position(q) for q in d["q_ik"]])
    # pose error (calculated - desired)
    err = np.array([pose_error(a, b) for a, b in zip(d["poses"], calc)])
    # euclidean position error (mm)
    pos = np.linalg.norm(err[:, :3], axis=1)
    
    # output stats (nan* functions ignore points where IK failed)
    print(f"Position error (mm):  mean {np.nanmean(pos):.4g} | max {np.nanmax(pos):.4g}")
    
    for i, n in enumerate(["roll", "pitch", "yaw"]):
        print(f"{n} error (deg): mean {np.nanmean(np.abs(err[:, 3+i])):.4g} | max {np.nanmax(np.abs(err[:, 3+i])):.4g}")
        
    np.savez("ik_errors.npz", ts=d["ts"], desired=d["poses"], calc=calc, err=err)

# validates FK accuracy by moving to test joint configurations and comparing FK outputs against reported robot poses
def validate_fk(arm):
    rows = []
    
    for q in CONFIGS:
        # blocking move to target joint configuration
        arm.set_servo_angle(angle=q, speed=SPEED, mvacc=100, wait=True, is_radian=False)
        time.sleep(0.5)
        
        # measured joint angles in degrees
        q_real = np.array(arm.get_servo_angle(is_radian=False)[1][:6])
        # reported robot pose
        p_robot = np.array(arm.get_position(is_radian=False)[1][:6])
        # FK computed from measured joints
        p_mine = tip_position(q_real)
        # pose error (FK - reported pose)
        diff = pose_error(p_robot, p_mine)
        
        rows.append([*q_real, *p_robot, *p_mine, *diff])
        print("q =", np.round(q_real, 2))
        print("   robot:", np.round(p_robot, 2))
        print("   FK   :", np.round(p_mine, 2))
        print("   diff :", np.round(diff, 3))

    # data format: q(6) | robot_pose(6) | fk_pose(6) | diff(6)
    np.save("fk_validation.npy", np.array(rows))
    print("Saved fk_validation.npy")

# executes a time scaled trajectory while asynchronously logging real time robot pose and joint data
def run_trajectory(arm, slow):
    # stretched total duration
    Tt = T * slow
    # waypoint timestamps
    ts = np.linspace(0, Tt, N_WP)
    # resampled waypoints
    P = np.array([traj(t / slow) for t in ts])
    # constant time step between waypoints
    dt = ts[1] - ts[0]
    
    # calculates segment TCP speeds
    speeds = [float(np.clip(np.linalg.norm(P[i, :3] - P[i-1, :3]) / dt, V_MIN, V_MAX)) for i in range(1, N_WP)]
    print(f"SLOW={slow}: expected duration {Tt:.1f} s | max requested TCP speed {max(speeds):.0f} mm/s")

    # blocking move to initial waypoint
    code = arm.set_position(*[float(v) for v in P[0]], speed=50, mvacc=200, wait=True)   # ponto inicial
    if code != 0:
        print("Error moving to the starting point:", code); return False

    # start background logging thread (~50 Hz sampling rate)
    log, stop = [], threading.Event()
    t0 = time.time()
    def logger():
        while not stop.is_set():
            log.append([time.time() - t0, *arm.position[:6], *arm.angles[:6]])
            time.sleep(0.02)
    th = threading.Thread(target=logger, daemon=True); th.start()

    # queue non-blocking waypoints with corner blending (radius=RADIUS)
    ok = True
    for i in range(1, N_WP):
        code = arm.set_position(*[float(v) for v in P[i]], speed=speeds[i-1], mvacc=2000,
                                radius=RADIUS, wait=False)
        if code != 0:
            print(f"Failed at point {i}, code {code}"); arm.set_state(4); ok = False; break
            
    # monitor motion completion with safety timeout (Tt + 15s)
    time.sleep(0.3)
    while ok and arm.get_is_moving() and time.time() - t0 < Tt + 15:
        time.sleep(0.05)
    # post-motion buffer for telemetry
    time.sleep(0.3)
    
    stop.set(); th.join()
    print(f"Real duration: {time.time() - t0:.1f} s | error: {arm.error_code} | warning: {arm.warn_code}")
    # log schema: t | pose(6) | joints(6)
    np.savez(f"traj_run_slow{slow}.npz", ts=ts, desired=P, log=np.array(log))
    
    print(f"Saved traj_run_slow{slow}.npz")
    
    return ok and arm.error_code == 0

# executes the trajectory across configured SLOW levels with user confirmation between runs
def run_all_levels(arm):
    for slow in SLOW_LEVELS:
        r = input(f"\nRun the swing with SLOW={slow}? Keep a hand on the emergency stop button. [Enter = yes | s = skip | q = quit] ").strip().lower()
        if r == "q": break
        if r == "s": continue
        
        # abort sequence if execution fails to prevent running faster speeds unsafely
        if not run_trajectory(arm, slow):
            print("There was a problem; stop here and check what happened before continuing.")

if __name__ == "__main__":
    # offline mode: analyze saved IK data without connecting to the robot
    if "--analyze" in sys.argv:
        analyze_ik(); sys.exit()
        
    # connect to the robot, clear errors, enable motors, and enter position control mode (mode 0, state 0)
    arm = XArmAPI(IP)
    arm.clean_warn(); arm.clean_error()
    arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
    
    # TCP offset note: FK assumes flange coordinates, so non-zero TCP offsets cause pose mismatches
    print("Current robot TCP offset:", arm.tcp_offset, "(for validation, set to [0,0,0,0,0,0])")
    
    try:
        if "--traj-only" not in sys.argv:
            # compute IK offline & run safety check
            safe = collect_ik(arm)
            # evaluate IK -> FK error metrics
            analyze_ik()
            input("\nThe next step MOVES the robot (FK validation). Is the area clear? Press Enter to continue... ")
            # execute physical FK validation (Task 1)
            validate_fk(arm)
        else:
            # skip IK safety checks with --traj-only flag
            safe = True
            
        # prompt for confirmation if IK computation contains failures or potential singularities
        if not safe:
            print("\nWARNING: IK computation failed or contains large joint jumps.")
            if input("Type 'yes' to proceed anyway: ").strip().lower() != "yes":
                sys.exit()
                
        # execute trajectory runs across SLOW levels (Task 3)
        run_all_levels(arm)
    except KeyboardInterrupt:
        # stop robot motion immediately (state 4)
        arm.set_state(4)
        print("Interrupted: robot stopped.")
    finally:
        # terminate controller connection
        arm.disconnect()