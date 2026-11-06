"""
Sessão de laboratório (Lite 6):
  1) recolhe os dados da IK do robô ao longo da trajetória (Task 4)  -> o robô NÃO se mexe
  2) valida a FK (Task 1): move o robô a algumas configurações e compara com tip_position
  3) executa a tacada (Task 3) com SLOW = 5, 2, 1 e regista a pose real

Correr no laboratório:   python session_data.py
Só a tacada:             python session_data.py --traj-only
Analisar em casa:        python session_data.py --analyze      (usa ik_data.npz, não precisa do robô)
"""
import sys, time, threading
import numpy as np
from xarm.wrapper import XArmAPI
from matrices import tip_position
from trajectory_def import traj, T

IP    = '192.168.1.158'
N_IK  = 20                     # nº de pontos da trajetória para a Task 4
SPEED = 20                     # deg/s nos movimentos da validação (devagar!)
SLOW_LEVELS = [5, 2, 1]        # fatores de esticar o tempo (5 = 5x mais lento)
N_WP    = 20                   # nº de waypoints da tacada
V_MAX   = 500                  # mm/s, limite do TCP (confirma no manual)
V_MIN   = 10                   # mm/s, velocidade mínima enviada ao set_position
RADIUS  = 5                    # mm, raio de blending (confirmado na doc do SDK)
MAX_JUMP = 20                  # graus; salto de junta entre pontos da IK que consideramos suspeito
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
def collect_ik(arm): # Serve para correr e guardar as inverse kinematics do robô - logging
    ts = np.linspace(0, T, N_IK)
    poses = np.array([traj(t) for t in ts])
    q_ik, codes = [], []
    for pose in poses: #
        code, q = arm.get_inverse_kinematics([float(v) for v in pose],
                                             input_is_radian=False, return_is_radian=False)
        codes.append(code)
        q_ik.append(list(q[:6]) if code == 0 else [np.nan] * 6)
        if code != 0:
            print("IK falhou (código", code, ") em", np.round(pose, 1))
    q_ik = np.array(q_ik)
    np.savez("ik_data.npz", ts=ts, poses=poses, q_ik=q_ik, codes=np.array(codes))
    print("Guardado ik_data.npz")
    ok = all(c == 0 for c in codes)
    jump = np.nanmax(np.abs(np.diff(q_ik, axis=0))) if ok else np.nan
    print(f"Verificação: IK ok em todos os pontos: {ok} | maior salto de junta entre pontos: {jump:.1f}°")
    return ok and jump <= MAX_JUMP

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


# ---------------- Task 3: executar a tacada ----------------
def run_trajectory(arm, slow):
    Tt = T * slow
    ts = np.linspace(0, Tt, N_WP)
    P = np.array([traj(t / slow) for t in ts])          # mesma forma, tempo esticado
    dt = ts[1] - ts[0]
    speeds = [float(np.clip(np.linalg.norm(P[i, :3] - P[i-1, :3]) / dt, V_MIN, V_MAX)) for i in range(1, N_WP)]
    print(f"SLOW={slow}: duração esperada {Tt:.1f} s | velocidade TCP máx. pedida {max(speeds):.0f} mm/s")

    code = arm.set_position(*[float(v) for v in P[0]], speed=50, mvacc=200, wait=True)   # ponto inicial
    if code != 0:
        print("Erro a ir ao ponto inicial:", code); return False

    log, stop = [], threading.Event()
    t0 = time.time()
    def logger():                                        # regista a pose real durante o movimento
        while not stop.is_set():
            log.append([time.time() - t0, *arm.position[:6], *arm.angles[:6]])
            time.sleep(0.02)
    th = threading.Thread(target=logger, daemon=True); th.start()

    ok = True
    for i in range(1, N_WP):
        code = arm.set_position(*[float(v) for v in P[i]], speed=speeds[i-1], mvacc=2000,
                                radius=RADIUS, wait=False)
        if code != 0:
            print(f"Falhou no ponto {i}, código {code}"); arm.set_state(4); ok = False; break
    time.sleep(0.3)
    while ok and arm.get_is_moving() and time.time() - t0 < Tt + 15:
        time.sleep(0.05)
    time.sleep(0.3)
    stop.set(); th.join()
    print(f"Duração real: {time.time() - t0:.1f} s | erro: {arm.error_code} | aviso: {arm.warn_code}")
    np.savez(f"traj_run_slow{slow}.npz", ts=ts, desired=P,
             log=np.array(log))                          # log: t | x y z roll pitch yaw | j1..j6
    print(f"Guardado traj_run_slow{slow}.npz")
    return ok and arm.error_code == 0

def run_all_levels(arm):
    for slow in SLOW_LEVELS:
        r = input(f"\nCorrer a tacada com SLOW={slow}? Mão no botão de emergência. [Enter = sim | s = saltar | q = terminar] ").strip().lower()
        if r == "q": break
        if r == "s": continue
        if not run_trajectory(arm, slow):
            print("Houve um problema; pára aqui e vê o que aconteceu antes de continuar."); break
        # arm.set_position("0,0,0,0,0,0", speed=50, mvacc=200, wait=True)  # voltar à posição inicial

if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze_ik(); sys.exit()
    arm = XArmAPI(IP)
    arm.clean_warn(); arm.clean_error()
    arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
    print("TCP offset atual do robô:", arm.tcp_offset, "(para a validação deve ser [0,0,0,0,0,0])")
    try:
        if "--traj-only" not in sys.argv:
            safe = collect_ik(arm)                         # sem movimento
            analyze_ik()
            input("\nO próximo passo MOVE o robô (validação da FK). Zona livre? Enter para continuar... ")
            validate_fk(arm)
        else:
            safe = True
        if not safe:
            print("\nATENÇÃO: a IK falhou nalgum ponto ou tem saltos grandes (possível singularidade).")
            if input("Escreve 'sim' para correr a tacada mesmo assim: ").strip().lower() != "sim":
                sys.exit()
        run_all_levels(arm)
    except KeyboardInterrupt:
        arm.set_state(4); print("Interrompido: robô parado.")
    finally:
        arm.disconnect()