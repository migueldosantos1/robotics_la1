"""
Sessão de laboratório (Lite 6):
  1) recolhe os dados da IK do robô ao longo da trajetória (Task 4)  -> o robô NÃO se mexe
  2) valida a FK (Task 1): move o robô a algumas configurações e compara com tip_position

Correr no laboratório:   python session_data.py
Analisar em casa:        python session_data.py --analyze      (usa ik_data.npz, não precisa do robô)
"""
import sys, time
import numpy as np
from xarm.wrapper import XArmAPI
from matrices import tip_position
from trajectory_def import traj, T

IP    = '192.168.1.158'  # IP do robô no laboratório
N_IK  = 80                     # nº de pontos da trajetória para a Task 4
SPEED = 20                     # deg/s nos movimentos da validação (devagar!)
CONFIGS = [                    # juntas em graus; confirma que o espaço está livre antes de correr
    [0, 0, 0, 0, 0, 0],
    [30, 20, 40, 0, 20, 15],
    [-30, 30, 60, 20, 40, -20],
    [20, 40, 60, -20, 50, 30],
    [0, 30, 30, 0, 30, 0],
]

def wrap(a):                   # diferença de ângulos em [-180, 180)
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0

def pose_error(p_des, p_calc):  # calculada - desejada; mm e graus
    e = np.asarray(p_calc, float) - np.asarray(p_des, float)
    e[3:] = wrap(e[3:])
    return e

# ---------------- Task 4: IK do robô ----------------
def collect_ik(arm):
    ts = np.linspace(0, T, N_IK)
    poses = np.array([traj(t) for t in ts])
    q_ik, codes = [], []
    for pose in poses:
        code, q = arm.get_inverse_kinematics([float(v) for v in pose],
                                             input_is_radian=False, return_is_radian=False)
        codes.append(code)
        q_ik.append(list(q[:6]) if code == 0 else [np.nan] * 6)
        if code != 0:
            print("IK falhou (código", code, ") em", np.round(pose, 1))
    np.savez("ik_data.npz", ts=ts, poses=poses, q_ik=np.array(q_ik), codes=np.array(codes))
    print("Guardado ik_data.npz")

def analyze_ik(path="ik_data.npz"):
    d = np.load(path)
    calc = np.array([tip_position(q) for q in d["q_ik"]])        # a tua FK aplicada aos ângulos da IK
    err = np.array([pose_error(a, b) for a, b in zip(d["poses"], calc)])
    pos = np.linalg.norm(err[:, :3], axis=1)
    print(f"Erro de posição (mm):  média {np.nanmean(pos):.4g} | máx {np.nanmax(pos):.4g}")
    for i, n in enumerate(["roll", "pitch", "yaw"]):
        print(f"Erro de {n} (graus): média {np.nanmean(np.abs(err[:, 3+i])):.4g} | máx {np.nanmax(np.abs(err[:, 3+i])):.4g}")
    np.savez("ik_errors.npz", ts=d["ts"], desired=d["poses"], calc=calc, err=err)

# ---------------- Task 1: validar a FK ----------------
def validate_fk(arm):
    rows = []
    for q in CONFIGS:
        arm.set_servo_angle(angle=q, speed=SPEED, mvacc=100, wait=True, is_radian=False)
        time.sleep(0.5)
        q_real = np.array(arm.get_servo_angle(is_radian=False)[1][:6])
        p_robot = np.array(arm.get_position(is_radian=False)[1][:6])
        p_mine = tip_position(q_real)
        diff = pose_error(p_robot, p_mine)
        rows.append([*q_real, *p_robot, *p_mine, *diff])
        print("q =", np.round(q_real, 2))
        print("   robô :", np.round(p_robot, 2))
        print("   FK   :", np.round(p_mine, 2))
        print("   dif. :", np.round(diff, 3))
    np.save("fk_validation.npy", np.array(rows))      # colunas: q(6) | robô(6) | FK(6) | dif(6)
    print("Guardado fk_validation.npy")

if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze_ik(); sys.exit()
    arm = XArmAPI(IP)
    arm.clean_warn(); arm.clean_error()
    arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
    print("TCP offset atual do robô:", arm.tcp_offset, "(para a validação deve ser [0,0,0,0,0,0])")
    collect_ik(arm)                                    # sem movimento
    analyze_ik()
    input("\nPróximo passo MOVE o robô. Zona livre e mão perto do botão de emergência? Enter para continuar... ")
    validate_fk(arm)
    input("\nTacada. Zona livre e mão perto do botão de emergência? Enter para continuar... ")
    
    arm.disconnect()