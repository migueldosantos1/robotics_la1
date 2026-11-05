import time
import numpy as np
from xarm.wrapper import XArmAPI
from trajectory_def import traj, T as T_SWING

IP = '192.168.1.158'
SLOW = 5.0        # fator de esticar o tempo: começa a 5, depois 2, depois 1 (PARA TESTE)
V_MAX = 500       # mm/s (limite do fabricante, confirma no manual)
RADIUS = 5        # mm, raio de blending (confirma o significado na doc do SDK)


N = 80 # nº de waypoints
T = T_SWING * SLOW # duração real (esticada)
ts = np.linspace(0, T, N) # time samples -> array com os valores de t espaçados T (s) N vezes
P = np.array([traj(t / SLOW) for t in ts]) # Array com os pontos da trajetória

# Preparação do robô
arm = XArmAPI(IP)
arm.clean_warn()
arm.clean_error()
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)

# Ir para o ponto inicial, a baixa velocidade, a esperar
move = arm.set_position(*P[0], speed=50, mvacc=200, wait=True)
assert move == 0, f"Erro a ir ao ponto inicial: {move}"

# Percorrer a trajetória
dt = ts[1] - ts[0]
log = []
t0 = time.time()
for i in range(1, N):
    dist = np.linalg.norm(P[i, :3] - P[i-1, :3]) # Não se utilizam os ângulos
    speed = float(np.clip(dist / dt, 20, V_MAX))
    move = arm.set_position(*P[i], speed=speed, mvacc=2000,
                            radius=RADIUS, wait=False)
    if move != 0: # failsafe
        print(f"Falhou no ponto {i}, código {move}")
        arm.set_state(4)   # estado de paragem
        break

# Terminar o movimento
while arm.get_is_moving():
    log.append((time.time() - t0, *arm.get_position()[1]))
    time.sleep(0.02)

print("Duração real:", time.time() - t0, "s (esperado ≈", T, "s)")
np.save("pose_real.npy", np.array(log))
arm.disconnect()