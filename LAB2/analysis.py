import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from matrices import tip_position
from trajectory_def import traj, T

def normalize(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0

def post_processing(file_path="traj_run_slow1.npz"):

    # Carregar os dados gravados
    data = np.load(file_path)
    ts_desired = data["ts"] # Tempo teórico [N_WP]
    desired_poses = data["desired"] # Poses desejadas [N_WP, 6] (x, y, z, roll, pitch, yaw)
    log = data["log"] # Log real: [t, x, y, z, roll, pitch, yaw, j1, j2, j3, j4, j5, j6]

    t_real = log[:, 0] # Tempo real
    robot_cartesian_logged = log[:, 1:7] # Poses reais
    q_actual = log[:, 7:13] # Angulos das juntas reais

    # Aplicar a cinemática direta às juntas reais
    dk_reconstructed = np.array([tip_position(q) for q in q_actual])

    # Interpolação do SciPy para alinhar o log assíncrono com a grelha de tempo desejada (ts_desired)
    # Necessário por causa das pequenas diferenças entre os intervalos definidos perfeitos e os registados
    interp_dk = interp1d(t_real, dk_reconstructed, axis=0, bounds_error=False, fill_value="extrapolate")
    interp_robot = interp1d(t_real, robot_cartesian_logged, axis=0, bounds_error=False, fill_value="extrapolate")

    # Depois da interpolação, já dá para extrair os valores nos tempos ts_derired
    calc_poses_at_ts = interp_dk(ts_desired)
    robot_poses_at_ts = interp_robot(ts_desired)

    # Erro entre a trajetória desejada e a trajetória que o robô seguiu na realidade
    err_total = calc_poses_at_ts - desired_poses
    err_total[:, 3:] = normalize(err_total[:, 3:]) # Normalizar erros de orientação
    pos_error_total = np.linalg.norm(err_total[:, :3], axis=1) # Erro euclidiano em mm

    # Erro entre a posição obtida pela cinemática direta e a posição que o robô diz que está a fazer
    err_dk_vs_robot = calc_poses_at_ts - robot_poses_at_ts
    err_dk_vs_robot[:, 3:] = normalize(err_dk_vs_robot[:, 3:])
    pos_error_dk_robot = np.linalg.norm(err_dk_vs_robot[:, :3], axis=1)

    print(f"Resultados para o ficheiro \"{file_path}\":")
    print(f"Erro de posição médio (desejado vs DK): {np.mean(pos_error_total):.2f} mm")
    print(f"Erro de posição máximo (desejado vs DK): {np.max(pos_error_total):.2f} mm")
    print(f"Erro DK vs pose interna do robô (média): {np.mean(pos_error_dk_robot):.2f} mm")

    # Distância à trajetória desejada (ignora o timing)
    tt = np.linspace(0, T, 1000)
    dense = np.array([traj(t) for t in tt])[:, :3]
    d_path = cKDTree(dense).query(dk_reconstructed[:, :3])[0]
    print(f"\nDistância média ao caminho: {d_path.mean():.2f} mm")
    print(f"Distância máxima ao caminho: {d_path.max():.2f} mm")

    # Atraso que minimiza o erro
    shifts = np.linspace(0, 2.0, 1000)
    errs = [np.mean(np.linalg.norm(interp_dk(ts_desired + s)[:, :3] - desired_poses[:, :3], axis=1)) for s in shifts]
    best = shifts[np.argmin(errs)]
    print(f"Atraso estimado: {best:.2f} s")
    print(f"Erro médio com atraso compensado: {min(errs):.2f} mm")

    # Plots, 4 subplots nuuma figura
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Trajetória no plano XY
    axs[0, 0].plot(desired_poses[:, 0], desired_poses[:, 1], 'r--', label='Desejado')
    axs[0, 0].plot(calc_poses_at_ts[:, 0], calc_poses_at_ts[:, 1], 'b-', label='Executado (dk)')
    axs[0, 0].set_title("Trajetória no plano XY")
    axs[0, 0].set_xlabel("X (mm)")
    axs[0, 0].set_ylabel("Y (mm)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Erro euclidiano de posição ao longo do tempo
    axs[0, 1].plot(ts_desired, pos_error_total, 'g-')
    axs[0, 1].set_title("Erro de posição")
    axs[0, 1].set_xlabel("Tempo (s)")
    axs[0, 1].set_ylabel("Erro (mm)")
    axs[0, 1].grid(True)

    # Decomposição do Erro por Eixo (X, Y, Z)
    axs[1, 0].plot(ts_desired, err_total[:, 0], label='Erro X')
    axs[1, 0].plot(ts_desired, err_total[:, 1], label='Erro Y')
    axs[1, 0].plot(ts_desired, err_total[:, 2], label='Erro Z')
    axs[1, 0].set_title("Decomposição do erro cartesiano")
    axs[1, 0].set_xlabel("Tempo (s)")
    axs[1, 0].set_ylabel("Erro (mm)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Erros de Orientação (Roll, Pitch, Yaw)
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
    # Mudar aqui o nome do ficheiro a carregar
    post_processing("traj_run_slow1.npz")
