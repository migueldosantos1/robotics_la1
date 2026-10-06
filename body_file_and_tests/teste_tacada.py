import time
import math
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("main_v3.xml")
data = mujoco.MjData(model)

# Ângulo entre os dois braços, com base na geometria do tronco e dos braços
ANGULO_BRACOS = math.asin(3 / 7)

DELAY_INICIAL = 5 # Esperar um pouco antes da tacada
DURACAO = 0.1 # Tempo que a tacada demora

# Atuadores:
act_rot_anca_z = model.actuator("act_rot_anca_z").id
act_rot_anca_y = model.actuator("act_rot_anca_y").id
act_rot_anca_x = model.actuator("act_rot_anca_x").id
act_left_arm_x = model.actuator("act_left_arm_x").id
act_left_arm_y = model.actuator("act_left_arm_y").id
act_right_arm_x = model.actuator("act_right_arm_x").id
act_right_arm_y = model.actuator("act_right_arm_y").id
act_wrist = model.actuator("act_wrist").id

# Posições das juntas:
adr_rot_anca_z = int(model.joint("rot_anca_z").qposadr[0])
adr_rot_anca_y = int(model.joint("rot_anca_y").qposadr[0])
adr_rot_anca_x = int(model.joint("rot_anca_x").qposadr[0])
adr_left_arm_x = int(model.joint("left_arm_x").qposadr[0])
adr_left_arm_y = int(model.joint("left_arm_y").qposadr[0])
adr_right_arm_x = int(model.joint("right_arm_x").qposadr[0])
adr_right_arm_y = int(model.joint("right_arm_y").qposadr[0])
adr_wrist = int(model.joint("wrist").qposadr[0])

# Posição inicial, para não haver movimento estranho no início (corpo a 45 graus em Z)
data.qpos[adr_rot_anca_z] = - math.pi / 4
data.qpos[adr_rot_anca_y] = 0.0
data.qpos[adr_rot_anca_x] = 0.0
data.qpos[adr_left_arm_x] = ANGULO_BRACOS
data.qpos[adr_left_arm_y] = - math.pi / 2
data.qpos[adr_right_arm_x] = ANGULO_BRACOS
data.qpos[adr_right_arm_y] = - math.pi / 2
data.qpos[adr_wrist] = ANGULO_BRACOS # Sempre estático por enquanto


mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        inicio = time.time()
        t = data.time

        # Fase - 0 durante o delay inicial, entre 0 e 1 durante o moviemento, 1 após o final do movimento
        if t < DELAY_INICIAL:
            fase = 0
        elif t > DELAY_INICIAL and t < DELAY_INICIAL + DURACAO:
            fase = (t - DELAY_INICIAL) / DURACAO
        else:
            fase = 1

        onda_1 = math.sin(math.pi * fase * 0.5) # Meio período, para movimentos de um único sentido (rotação do tronco)
        onda_2 = math.sin(math.pi * fase) # Período inteiro, para movimentos de dois sentidos (subida e descida dos braços)
 
        data.ctrl[act_rot_anca_z] = math.radians(90) * onda_1 - math.pi / 4
        data.ctrl[act_rot_anca_y] = math.radians(30) * onda_2
        data.ctrl[act_rot_anca_x] = 0.0
        data.ctrl[act_left_arm_x] = -ANGULO_BRACOS
        data.ctrl[act_left_arm_y] = -math.pi / 2
        data.ctrl[act_right_arm_x] = ANGULO_BRACOS
        data.ctrl[act_right_arm_y] = -math.pi / 2
        data.ctrl[act_wrist] = -ANGULO_BRACOS # Estático por enquanto
 
        mujoco.mj_step(model, data)
        v.sync()
 
        # Mantém o tempo real (o programa pausa durante o tempo que sobra após os cálculos para esse passo)
        resto = model.opt.timestep - (time.time() - inicio)
        if resto > 0:
            time.sleep(resto)