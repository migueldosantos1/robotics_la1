"""
run_swing_on_robot.py - Tarefa 3: faz o Lite 6 executar a trajetoria da tacada.

USO (por omissao e DRY-RUN: nao liga ao robot, so valida e mostra o que faria):
    python run_swing_on_robot.py                              # dry-run
    python run_swing_on_robot.py --ip 192.168.1.xxx --execute --time-scale 3   # 1o teste: 3x mais lento
    python run_swing_on_robot.py --ip 192.168.1.xxx --execute --time-scale 1   # velocidade nominal
    python run_swing_on_robot.py --ip ... --execute --mode cartesian           # usa IK interna do robot

Modos:
  servo_j   (default) stream de angulos de junta a 100 Hz (set_servo_angle_j, modo 1). As juntas vem
            da nossa IK (swing_trajectory.py) -> tempo e velocidade controlados ponto a ponto.
  cartesian waypoints cartesianos com set_position (modo 0); a IK e a do controlador do robot.

SEGURANCA: tenham a mao no botao de emergencia; mesa/bola no sitio previsto; TCP = flange
(o script poe tcp_offset = 0; se montarem gripper/taco ajustem BALL/R_ARM em swing_trajectory.py).
Requer: pip install xarm-python-sdk
"""
import argparse
import sys
import time
import numpy as np

import swing_trajectory as st
from lite6_kin import Q_MIN_DEG, Q_MAX_DEG, QD_MAX_DEG, fk_pose


def load_or_build():
    t, phi, dphi, P, RPY = st.build_cartesian()
    q_seed = st.find_start_config()
    Q, ok = st.solve_joint_trajectory(t, phi, q_seed)
    res, qd, v = st.check_feasibility(t, Q, P)
    return t, P, RPY, Q, res, ok


def preflight(Q, res, ok, time_scale):
    problems = []
    if not ok.all():
        problems.append("IK nao convergiu em todos os pontos")
    if not res["limits_ok"]:
        problems.append("limites de junta violados")
    if res["qd_max"] / time_scale > st.VEL_SAFETY * QD_MAX_DEG:
        problems.append(f"velocidade de junta {res['qd_max']/time_scale:.0f} deg/s acima do alvo")
    if not (res["reach_ok"] and res["table_ok"] and res["base_ok"]):
        problems.append("falha de alcance/mesa/base")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=None)
    ap.add_argument("--execute", action="store_true", help="enviar mesmo comandos ao robot")
    ap.add_argument("--mode", choices=["servo_j", "cartesian"], default="servo_j")
    ap.add_argument("--time-scale", type=float, default=3.0,
                    help=">1 = mais lento (default 3 para o 1o teste); 1 = tempos do projeto")
    args = ap.parse_args()
    if args.time_scale < 1.0:
        sys.exit("time-scale < 1 nao permitido (excederia as velocidades verificadas).")

    t, P, RPY, Q, res, ok = load_or_build()
    problems = preflight(Q, res, ok, args.time_scale)
    print(f"Pontos: {len(t)} | duracao {t[-1]*args.time_scale:.1f} s | "
          f"vel. junta max {res['qd_max']/args.time_scale:.1f} deg/s")
    if problems:
        sys.exit("ABORTADO: " + "; ".join(problems))
    if not (args.execute and args.ip):
        print("DRY-RUN OK (use --ip e --execute para mover o robot).")
        return

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip)
    arm.clean_warn(); arm.clean_error()
    arm.motion_enable(True)
    arm.set_tcp_offset([0, 0, 0, 0, 0, 0])
    arm.set_collision_sensitivity(3)
    arm.set_mode(0); arm.set_state(0); time.sleep(0.5)

    # 1) ir devagar ao ponto inicial (20 deg/s) -- movimento "normal" bloqueante
    print("A ir para a posicao inicial a 20 deg/s ...")
    code = arm.set_servo_angle(angle=list(Q[0]), speed=20, mvacc=100, is_radian=False, wait=True)
    if code != 0:
        arm.disconnect(); sys.exit(f"Falha a ir ao ponto inicial (code={code})")
    _, cur = arm.get_servo_angle(is_radian=False)
    err = np.max(np.abs(np.array(cur[:6]) - Q[0]))
    print(f"Erro maximo de junta no ponto inicial: {err:.3f} deg")
    if err > 1.0:
        arm.disconnect(); sys.exit("Erro inicial demasiado grande, abortado.")

    # 2) executar
    try:
        if args.mode == "servo_j":
            arm.set_mode(1); arm.set_state(0); time.sleep(0.3)
            dt = (t[1] - t[0]) * args.time_scale
            t0 = time.perf_counter()
            for k in range(len(t)):
                code = arm.set_servo_angle_j(list(Q[k]), is_radian=False)
                if code != 0:
                    print(f"Erro no ponto {k}: code={code}")
                    break
                while time.perf_counter() - t0 < (k + 1) * dt:
                    pass
        else:  # cartesian: waypoints a cada ~0.1 s do projeto, IK do controlador
            step = 10
            arm.set_mode(0); arm.set_state(0)
            for k in range(step, len(t), step):
                seg_t = (t[k] - t[k - step]) * args.time_scale
                dist = np.linalg.norm(P[k] - P[k - step])
                speed = max(dist / seg_t, 5.0)              # mm/s
                code = arm.set_position(*P[k], *RPY[k], speed=speed, mvacc=2000,
                                        radius=5, is_radian=False, wait=False)
                if code != 0:
                    print(f"Erro no waypoint {k}: code={code}")
                    break
            while arm.get_is_moving():
                time.sleep(0.05)
    finally:
        arm.set_mode(0); arm.set_state(0)
        arm.disconnect()
    print("Concluido.")


if __name__ == "__main__":
    main()
