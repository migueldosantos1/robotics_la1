'''
Script para analisar os traj_run_slowX.npz

Calcula os erros brutos (sem compensar atraso), a distância média ao caminho,
o atraso, e a distância com o atrso compensado

'''

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
    interp_dk = interp1d(t_real, dk_reconstructed, axis=0, bounds_error=False, fill_value="extrapolate")
    interp_robot = interp1d(t_real, robot_cartesian_logged, axis=0, bounds_error=False, fill_value="extrapolate")

    # Extração de poses amostradas nos tempos teóricos ts_desired
    calc_poses_at_ts = interp_dk(ts_desired)
    robot_poses_at_ts = interp_robot(ts_desired)

    # Erro entre a trajetória desejada e a trajetória reconstruída (bruto)
    err_total = calc_poses_at_ts - desired_poses
    err_total[:, 3:] = normalize(err_total[:, 3:]) # Normalizar erros de orientação
    pos_error_total = np.linalg.norm(err_total[:, :3], axis=1) # Erro euclidiano em mm

    # Erro entre a posição obtida pela cinemática direta e a posição reportada pelo robô
    err_dk_vs_robot = dk_reconstructed[:, :3] - robot_cartesian_logged[:, :3]
    pos_error_dk_robot = np.linalg.norm(err_dk_vs_robot, axis=1)

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

    # Otimização do atraso temporal
    shifts = np.linspace(0, 2.0, 1000)
    errs = [np.mean(np.linalg.norm(interp_dk(ts_desired + s)[:, :3] - desired_poses[:, :3], axis=1)) for s in shifts]
    best_shift = shifts[np.argmin(errs)]
    
    # Erros com o atraso de 0.31s compensado
    poses_compensated = interp_dk(ts_desired + best_shift)
    pos_error_compensated = np.linalg.norm(poses_compensated[:, :3] - desired_poses[:, :3], axis=1)

    print(f"Atraso estimado: {best_shift:.2f} s")
    print(f"Erro médio com atraso compensado: {np.mean(pos_error_compensated):.2f} mm")

    # Trajetória nos 3 planos (XY, XZ, YZ)
    fig1, axs1 = plt.subplots(1, 3, figsize=(12, 5))

    # Plano XY
    axs1[0].plot(desired_poses[:, 0], desired_poses[:, 1], 'r--', label='Desejado')
    axs1[0].plot(calc_poses_at_ts[:, 0], calc_poses_at_ts[:, 1], '-o', markersize=3, label='Executado (DK)')
    axs1[0].set_title("Plano XY")
    axs1[0].set_xlabel("X (mm)")
    axs1[0].set_ylabel("Y (mm)")
    axs1[0].legend()
    axs1[0].grid(True)

    # Plano XZ
    axs1[1].plot(desired_poses[:, 0], desired_poses[:, 2], 'r--', label='Desejado')
    axs1[1].plot(calc_poses_at_ts[:, 0], calc_poses_at_ts[:, 2], '-o', markersize=3, label='Executado (DK)')
    axs1[1].set_title("Plano XZ")
    axs1[1].set_xlabel("X (mm)")
    axs1[1].set_ylabel("Z (mm)")
    axs1[1].legend()
    axs1[1].grid(True)

    # Plano YZ
    axs1[2].plot(desired_poses[:, 1], desired_poses[:, 2], 'r--', label='Desejado')
    axs1[2].plot(calc_poses_at_ts[:, 1], calc_poses_at_ts[:, 2], '-o', markersize=3, label='Executado (DK)')
    axs1[2].set_title("Plano YZ")
    axs1[2].set_xlabel("Y (mm)")
    axs1[2].set_ylabel("Z (mm)")
    axs1[2].legend()
    axs1[2].grid(True)

    fig1.tight_layout()

    # Distancia ao caminho e consistência do modelo (o segundo não foi usado ainda)
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

    # Distância ao caminho geométrico
    axs2[0].plot(t_real, d_path, 'g-')
    axs2[0].set_title("Distância Geométrica ao Caminho")
    axs2[0].set_xlabel("Tempo real (s)")
    axs2[0].set_ylabel("Distância (mm)")
    axs2[0].grid(True)

    # Consistência: DK vs Pose interna do robô
    axs2[1].plot(t_real, pos_error_dk_robot, 'b-')
    axs2[1].set_title("DK vs Pose Reportada pelo Robô")
    axs2[1].set_xlabel("Tempo real (s)")
    axs2[1].set_ylabel("Diferença (mm)")
    axs2[1].grid(True)

    fig2.tight_layout()

    # Guardar os gráficos
    fig1.savefig("trajetoria_planos.png", dpi=300, bbox_inches="tight")
    fig2.savefig("metricas_4_2.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    post_processing("traj_run_slow1.npz")