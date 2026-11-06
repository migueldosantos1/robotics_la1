"""
ik_accuracy.py - Tarefa 4: precisao da cinematica inversa interna do Lite 6.

Fluxo (com robot):  pose da trajetoria --(arm.get_inverse_kinematics)--> q_robot
                    q_robot --(NOSSA cinematica directa, lite6_kin.fk_pose)--> pose_FK
                    erro = pose_FK - pose_original   (posicao em mm, orientacao em graus)
Tambem compara a NOSSA FK com a FK do robot (arm.get_forward_kinematics) -> valida os
parametros DH e as convencoes de angulos.

USO:
    python ik_accuracy.py                       # offline: usa a nossa IK numerica (teste do pipeline)
    python ik_accuracy.py --ip 192.168.1.xxx    # com robot (so calculos, NAO move o robot)
Gera: fig_ik_error.png, ik_accuracy.csv
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import swing_trajectory as st
from lite6_kin import fk_pose, pose_to_T, ik_numeric


def ang_diff(a, b):
    """Diferenca angular em (-180,180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def pose_error(p_fk, p_ref):
    e_pos = p_fk[:3] - p_ref[:3]
    e_ang = ang_diff(p_fk[3:], p_ref[3:])
    return e_pos, e_ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=None)
    ap.add_argument("--n", type=int, default=56, help="numero de pontos amostrados (~1 em cada 10)")
    args = ap.parse_args()

    t, phi, dphi, P, RPY = st.build_cartesian()
    idx = np.linspace(0, len(t) - 1, args.n).astype(int)
    poses = np.hstack([P[idx], RPY[idx]])            # [mm, graus]

    # semente para a IK offline
    q_seed = st.find_start_config()
    Qs = np.zeros((len(idx), 6)); fk_vs_robot = np.full((len(idx), 6), np.nan)

    if args.ip:
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(args.ip, do_not_open=False)
        arm.motion_enable(True); arm.set_tcp_offset([0, 0, 0, 0, 0, 0]); arm.set_mode(0); arm.set_state(0)
        for k, pose in enumerate(poses):
            code, q = arm.get_inverse_kinematics(list(pose), input_is_radian=False, return_is_radian=False)
            if code != 0:
                raise RuntimeError(f"IK do robot falhou no ponto {k} (code {code}) - pose inalcancavel?")
            Qs[k] = q[:6]
            code, pf = arm.get_forward_kinematics(list(Qs[k]), input_is_radian=False, return_is_radian=False)
            if code == 0:
                fk_vs_robot[k] = np.array(pf[:6])
        arm.disconnect()
        source = "IK do robot"
    else:
        q = np.radians(q_seed)
        for k, pose in enumerate(poses):
            q, _ = ik_numeric(pose_to_T(pose), q)
            Qs[k] = np.degrees(q)
        source = "IK numerica propria (offline)"

    # --- nossa FK com as juntas obtidas
    pose_fk = np.array([fk_pose(q) for q in Qs])
    E_pos = np.zeros((len(idx), 3)); E_ang = np.zeros((len(idx), 3))
    for k in range(len(idx)):
        E_pos[k], E_ang[k] = pose_error(pose_fk[k], poses[k])
    e_pos_norm = np.linalg.norm(E_pos, axis=1)

    print(f"Fonte das juntas: {source} | pontos: {len(idx)}")
    print(f"Erro de posicao  [mm]: max {e_pos_norm.max():.4g} | medio {e_pos_norm.mean():.4g} | rms {np.sqrt((e_pos_norm**2).mean()):.4g}")
    print(f"Erro de orient.  [deg] por eixo (roll,pitch,yaw): max {np.abs(E_ang).max(axis=0)}")
    if args.ip:
        d = np.array([np.r_[fk_vs_robot[k][:3] - pose_fk[k][:3], ang_diff(fk_vs_robot[k][3:], pose_fk[k][3:])]
                      for k in range(len(idx))])
        print(f"Nossa FK vs FK do robot: max pos {np.abs(d[:, :3]).max():.4g} mm | max ang {np.abs(d[:, 3:]).max():.4g} deg")
        print("  (se isto for grande: DH/offsets/convencao RPY nao coincidem -> rever lite6_kin.py)")

    np.savetxt("ik_accuracy.csv",
               np.hstack([t[idx][:, None], poses, Qs, pose_fk, E_pos, E_ang]), delimiter=",",
               header="t,x,y,z,roll,pitch,yaw,q1,q2,q3,q4,q5,q6,x_fk,y_fk,z_fk,r_fk,p_fk,yw_fk,ex,ey,ez,eroll,epitch,eyaw",
               comments="")

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(poses[:, 0], poses[:, 2], "b-", label="trajetoria desejada")
    ax[0, 0].plot(pose_fk[:, 0], pose_fk[:, 2], "r.", ms=4, label="FK(q_IK)")
    ax[0, 0].set_xlabel("x [mm]"); ax[0, 0].set_ylabel("z [mm]"); ax[0, 0].legend(); ax[0, 0].set_title("Plano x-z")
    ax[0, 1].plot(t[idx], E_pos); ax[0, 1].legend(["ex", "ey", "ez"]); ax[0, 1].set_title("Erro de posicao [mm]")
    ax[1, 0].plot(t[idx], E_ang); ax[1, 0].legend(["roll", "pitch", "yaw"]); ax[1, 0].set_title("Erro de orientacao [deg]")
    ax[1, 1].plot(t[idx], Qs); ax[1, 1].legend([f"J{i+1}" for i in range(6)], ncol=3, fontsize=8)
    ax[1, 1].set_title("Juntas obtidas pela IK [deg]")
    for a in ax.ravel(): a.set_xlabel("t [s]") if a is not ax[0, 0] else None; a.grid(alpha=0.3)
    fig.suptitle(f"Tarefa 4 - {source}")
    fig.tight_layout(); fig.savefig("fig_ik_error.png", dpi=150)


if __name__ == "__main__":
    main()
