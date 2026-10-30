import time
import math
import mujoco
import mujoco.viewer
import numpy as np
import os

folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "main_final.xml")

model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

# Ângulo entre os dois braços, com base na geometria do tronco e dos braços
ANGULO_BRACOS = math.asin(30 / 70)

# Duração de cada fase da demonstração
DURACAO_POR_FASE = 5

# 7 fases ao todo
N_FASES = 7

# Integrador utilizado - IMPLICIT / IMPLICITFAST são os mais adequados (por experiência)
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

# Atuadores:
act_rot_hip_z = model.actuator("act_rot_hip_z").id
act_rot_hip_y = model.actuator("act_rot_hip_y").id
act_rot_hip_x = model.actuator("act_rot_hip_x").id
act_left_arm_x = model.actuator("act_left_arm_x").id
act_left_arm_y = model.actuator("act_left_arm_y").id
act_right_arm_x = model.actuator("act_right_arm_x").id
act_right_arm_y = model.actuator("act_right_arm_y").id
act_left_elbow = model.actuator("act_left_elbow").id
act_right_elbow = model.actuator("act_right_elbow").id
act_wrist_y = model.actuator("act_wrist_y").id
act_wrist_x = model.actuator("act_wrist_x").id

# Posições das juntas:
adr_rot_hip_z = int(model.joint("rot_hip_z").qposadr[0])
adr_rot_hip_y = int(model.joint("rot_hip_y").qposadr[0])
adr_rot_hip_x = int(model.joint("rot_hip_x").qposadr[0])
adr_left_arm_x = int(model.joint("left_arm_x").qposadr[0])
adr_left_arm_y = int(model.joint("left_arm_y").qposadr[0])
adr_right_arm_x = int(model.joint("right_arm_x").qposadr[0])
adr_right_arm_y = int(model.joint("right_arm_y").qposadr[0])
adr_left_elbow = int(model.joint("left_elbow").qposadr[0])
adr_right_elbow = int(model.joint("right_elbow").qposadr[0])
adr_wrist_y = int(model.joint("wrist_y").qposadr[0])
adr_wrist_x = int(model.joint("wrist_x").qposadr[0])

# Ligação entre as mãos - configurada para aguentar movimentos bruscos
eq_maos = model.equality("hands_connected").id
model.eq_solref[eq_maos] = [0.001, 1.0]
model.eq_solimp[eq_maos] = [0.99, 0.99, 0.001, 0.5, 2]

# Posição inicial do robô:
# Cabeça e tronco direitos e a olhar em frente
# Braços esticados e em baixo, com um ângulo de 45 graus em relação ao eixo vertical
# Pulso a segurar o taco no plano dos braços
def posicao_inicial():
    data.qpos[adr_rot_hip_z] = 0.0
    data.qpos[adr_rot_hip_y] = 0.0
    data.qpos[adr_rot_hip_x] = 0.0
    data.qpos[adr_left_arm_x] = -ANGULO_BRACOS
    data.qpos[adr_left_arm_y] = math.radians(-60)
    data.qpos[adr_right_arm_x] = ANGULO_BRACOS
    data.qpos[adr_right_arm_y] = math.radians(-60)
    data.qpos[adr_left_elbow] = 0.0
    data.qpos[adr_right_elbow] = 0.0
    data.qpos[adr_wrist_y] = 0.0
    data.qpos[adr_wrist_x] = -ANGULO_BRACOS

posicao_inicial()
mujoco.mj_forward(model, data)

t_aux = 0.0

with mujoco.viewer.launch_passive(model, data) as v:

    # Configuração da posição inicial da câmara
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.lookat[:] = [0, 0, 1]
    v.cam.distance = 7.0
    v.cam.azimuth = -150
    v.cam.elevation = -6

    while v.is_running():
        inicio = time.time()
        t = data.time

        # Ativar a ligação entre as mãos
        data.eq_active[eq_maos] = 1

        # Impôr a posição inicial quando a simulação dá reset
        if t < t_aux:
            posicao_inicial()
            t = data.time
        t_aux = t

        # Calcular em que fase da demonstração está a simulação
        fase = int(t // DURACAO_POR_FASE) % N_FASES
        fase_t = (t % DURACAO_POR_FASE) / DURACAO_POR_FASE  # 0 a 1 dentro da fase

        onda_1 = math.sin(2 * math.pi * fase_t) # Oscila entre -1 e 1 ao longo de cada fase

        # Pose neutra (igual à posição inicial)
        # Serve para estarem sempre disponível os valores das juntas que não se mexem numa dada fase
        hip_z = 0.0
        hip_y = 0.0
        hip_x = 0.0
        left_arm_y = math.radians(-60)
        right_arm_y = math.radians(-60)
        wrist_y = 0.0
        wrist_x = -ANGULO_BRACOS
        elbow = 0.0

        # Em cada fase só se mexem as juntas pretendidas
        if fase == 0:
            hip_z = math.radians(60) * onda_1
        elif fase == 1:
            hip_y = 0.0 + math.radians(30) * onda_1
        elif fase == 2:
            hip_x = math.radians(25) * onda_1
        elif fase == 3:
            left_arm_y = math.radians(-60) + math.radians(45) * onda_1
            right_arm_y = math.radians(-60) + math.radians(45) * onda_1
        elif fase == 4:
            wrist_x = -ANGULO_BRACOS + math.radians(60) * onda_1
        elif fase == 5:
            wrist_y = math.radians(30) * onda_1
        elif fase == 6:
            elbow = math.radians(30) * onda_1

        # Sinais dos atuadores
        data.ctrl[act_rot_hip_z] = hip_z
        data.ctrl[act_rot_hip_y] = hip_y
        data.ctrl[act_rot_hip_x] = hip_x
        data.ctrl[act_left_arm_y] = left_arm_y
        data.ctrl[act_right_arm_y] = right_arm_y
        data.ctrl[act_wrist_y] = wrist_y
        data.ctrl[act_wrist_x] = wrist_x
        data.ctrl[act_right_elbow] = elbow

        # Estas juntas não se mexem em nenhuma fase (para manter as maõs juntas)
        data.ctrl[act_left_arm_x] = -ANGULO_BRACOS
        data.ctrl[act_right_arm_x] = ANGULO_BRACOS        

        mujoco.mj_step(model, data)
        v.sync()

        # Mantém o tempo real (o programa pausa durante o tempo que sobra após os cálculos para esse passo)
        resto = model.opt.timestep - (time.time() - inicio)
        if resto > 0:
            time.sleep(resto)