import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
# our forward kinematics - joints deg -> [x, y, z, roll, pitch, yaw]
from matrices import tip_position
# desired trajectory function and its nominal duration
from trajectory_def import traj, T

def normalize(a):
    # normalizes angles in degrees to the range [-180, 180]
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0

def post_processing(file_path="traj_run_slow1.npz"):
    # load data
    data = np.load(file_path)
    # theoretical waypoint times [N_WP]
    ts_desired = data["ts"]
    # desired poses [N_WP, 6]: x, y, z, roll, pitch, yaw
    desired_poses = data["desired"]
    # real log [M, 13]: t, x, y, z, roll, pitch, yaw, j1..j6
    log = data["log"]

    # real timestamps of the log samples [M]
    t_real = log[:, 0]
    # pose reported by the robot thtough get_position() [M, 6]
    robot_cartesian_logged = log[:, 1:7]
    # real joint angles degrees [M, 6]
    q_actual = log[:, 7:13]

    # apply our forward kinematics to every logged joint vector
    dk_reconstructed = np.array([tip_position(q) for q in q_actual])

    # interpolates logged DK and robot poses over real timestamps to align asynchronous logs with target times
    interp_dk = interp1d(t_real, dk_reconstructed, axis=0, bounds_error=False, fill_value="extrapolate")
    interp_robot = interp1d(t_real, robot_cartesian_logged, axis=0, bounds_error=False, fill_value="extrapolate")

    # with the interpolators ready, sample both signals at the desired times ts_desired
    # executed pose from our DK, at ts_desired
    calc_poses_at_ts = interp_dk(ts_desired)
    # pose reported by the robot, at ts_desired
    robot_poses_at_ts = interp_robot(ts_desired)

    # tracking error: executed pose minus the desired pose, this is evaluated only at N_WP waypoint times, not along the whole path
    err_total = calc_poses_at_ts - desired_poses
    # wrap orientation errors to [-180, 180]
    err_total[:, 3:] = normalize(err_total[:, 3:])
    # euclidean position error in mm
    pos_error_total = np.linalg.norm(err_total[:, :3], axis=1)

    # calculates position and normalized angle errors between computed DK and reported robot poses
    err_dk_vs_robot = calc_poses_at_ts - robot_poses_at_ts
    err_dk_vs_robot[:, 3:] = normalize(err_dk_vs_robot[:, 3:])
    pos_error_dk_robot = np.linalg.norm(err_dk_vs_robot[:, :3], axis=1)

    print(f"Results for file \"{file_path}\":")
    print(f"Mean position error (expected vs DK): {np.mean(pos_error_total):.2f} mm")
    print(f"Max position error (expected vs DK): {np.max(pos_error_total):.2f} mm")
    print(f"DK vs robot internal pose error (mean): {np.mean(pos_error_dk_robot):.2f} mm")

    # measures pure geometric path deviation by finding the distance from each executed point to the nearest point on a densely sampled target path (ignoring timing)
    tt = np.linspace(0, T, 1000)
    dense = np.array([traj(t) for t in tt])[:, :3]
    d_path = cKDTree(dense).query(dk_reconstructed[:, :3])[0]
    print(f"\nDistância média ao caminho: {d_path.mean():.2f} mm")
    print(f"Distância máxima ao caminho: {d_path.max():.2f} mm")

    # estimates execution lag by finding the time shift (0–2s) that minimizes trajectory error between real and target poses
    shifts = np.linspace(0, 2.0, 1000)
    errs = [np.mean(np.linalg.norm(interp_dk(ts_desired + s)[:, :3] - desired_poses[:, :3], axis=1)) for s in shifts]
    best = shifts[np.argmin(errs)]
    print(f"Atraso estimado: {best:.2f} s")
    print(f"Erro médio com atraso compensado: {min(errs):.2f} mm")

    # plots - 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # top left: path in the XY plane, desired waypoints vs executed (DK)
    axs[0, 0].plot(desired_poses[:, 0], desired_poses[:, 1], 'r--', label='Desejado')
    axs[0, 0].plot(calc_poses_at_ts[:, 0], calc_poses_at_ts[:, 1], 'b-', label='Executado (dk)')
    axs[0, 0].set_title("Trajetória no plano XY")
    axs[0, 0].set_xlabel("X (mm)")
    axs[0, 0].set_ylabel("Y (mm)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # top right: euclidean position error over time
    axs[0, 1].plot(ts_desired, pos_error_total, 'g-')
    axs[0, 1].set_title("Erro de posição")
    axs[0, 1].set_xlabel("Tempo (s)")
    axs[0, 1].set_ylabel("Erro (mm)")
    axs[0, 1].grid(True)

    # bottom left: error split by axis (X, Y, Z) - shows which axis dominates
    axs[1, 0].plot(ts_desired, err_total[:, 0], label='Erro X')
    axs[1, 0].plot(ts_desired, err_total[:, 1], label='Erro Y')
    axs[1, 0].plot(ts_desired, err_total[:, 2], label='Erro Z')
    axs[1, 0].set_title("Decomposição do erro cartesiano")
    axs[1, 0].set_xlabel("Tempo (s)")
    axs[1, 0].set_ylabel("Erro (mm)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # bottom right: orientation errors (roll, pitch, yaw), already wrapped to [-180, 180]
    axs[1, 1].plot(ts_desired, err_total[:, 3], label='Roll')
    axs[1, 1].plot(ts_desired, err_total[:, 4], label='Pitch')
    axs[1, 1].plot(ts_desired, err_total[:, 5], label='Yaw')
    axs[1, 1].set_title("Erros de orientação")
    axs[1, 1].set_xlabel("Tempo (s)")
    axs[1, 1].set_ylabel("Erro (graus)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.show()

if __name__ == "__main__":
    # change the name of the file to load here
    post_processing("traj_run_slow1.npz")