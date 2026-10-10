import time
import math
import mujoco
import mujoco.viewer
import os

folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "main_v7.xml")

model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

ANGULO_BRACOS = math.asin(30 / 70)

DELAY_INICIAL = 2
TEMPO_ESPERA = 1
DURACAO = 0.25

model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

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

eq_maos = model.equality("hands_connected").id
model.eq_solref[eq_maos] = [0.001, 1.0]
model.eq_solimp[eq_maos] = [0.99, 0.99, 0.001, 0.5, 2]

# ---------------------------------------------------------------------------
# TASK 2 - Teste de perturbação por DOF.
#
# O enunciado pede para considerar a existência de perturbações em cada DOF
# (ex.: "o pulso do jogador a tremer"). Simulamos isso somando um ruído
# sinusoidal de alta frequência ao sinal de controlo do DOF escolhido,
# sobreposto ao movimento normal da tacada - tal como um tremor humano real,
# que se sobrepõe ao movimento voluntário em vez de o substituir.
#
# Para testar um DOF diferente, basta mudar o nome em NOISE_DOF para
# qualquer uma das chaves do dicionário ACTUADORES, por exemplo
# "rot_hip_x" para testar uma anca instável, ou None para correr a
# tacada "limpa", sem perturbação nenhuma (o caso de referência).
# ---------------------------------------------------------------------------

ACTUADORES = {
    "rot_hip_z": act_rot_hip_z, "rot_hip_y": act_rot_hip_y, "rot_hip_x": act_rot_hip_x,
    "left_arm_y": act_left_arm_y, "right_arm_y": act_right_arm_y,
    "left_elbow": act_left_elbow, "right_elbow": act_right_elbow,
    "wrist_y": act_wrist_y, "wrist_x": act_wrist_x,
}

import random

# Perturbar todos os DOF em simultaneo, cada um com uma amplitude mais pequena
# (um tremor generalizado e mais ligeiro, em vez de um unico DOF muito instavel).
# Para voltar a testar so um DOF, basta por os restantes a 0 neste dicionario.
AMPLITUDE_RUIDO = {nome: math.radians(2) for nome in ACTUADORES}
FREQ_RUIDO = 10.0  # Hz - frequencia do tremor (tremores humanos tipicos: 4-12 Hz)

# cada DOF treme com uma fase propria e independente, para nao vibrarem todos
# em sincronia (o que pareceria um movimento coordenado, nao um tremor real)
random.seed(0)
FASE_RUIDO = {nome: random.uniform(0, 2 * math.pi) for nome in ACTUADORES}


def posicao_inicial():
    data.qpos[adr_rot_hip_z] = 0.0
    data.qpos[adr_rot_hip_y] = math.radians(15)
    data.qpos[adr_rot_hip_x] = 0.0
    data.qpos[adr_left_arm_x] = -ANGULO_BRACOS
    data.qpos[adr_left_arm_y] = math.radians(-45)
    data.qpos[adr_right_arm_x] = ANGULO_BRACOS
    data.qpos[adr_right_arm_y] = math.radians(-45)
    data.qpos[adr_left_elbow] = 0.0
    data.qpos[adr_right_elbow] = 0.0
    data.qpos[adr_wrist_y] = 0.0
    data.qpos[adr_wrist_x] = -ANGULO_BRACOS


posicao_inicial()
mujoco.mj_forward(model, data)

t_aux = 0.0

with mujoco.viewer.launch_passive(model, data) as v:

    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.lookat[:] = [0, 0, 1]
    v.cam.distance = 7.0
    v.cam.azimuth = -150
    v.cam.elevation = -6

    while v.is_running():
        inicio = time.time()
        t = data.time

        data.eq_active[eq_maos] = 1

        if t < t_aux:
            posicao_inicial()
            t = data.time
        t_aux = t

        if t < DELAY_INICIAL:
            prep = 0.5 * (1 - math.cos(math.pi * t / DELAY_INICIAL))
        else:
            prep = 1

        if t < DELAY_INICIAL + TEMPO_ESPERA:
            fase = 0
        elif t < DELAY_INICIAL + TEMPO_ESPERA + DURACAO:
            fase = (t - (DELAY_INICIAL + TEMPO_ESPERA)) / DURACAO
        else:
            fase = 1

        onda_1 = math.sin(math.pi * fase / 2)
        onda_2 = math.sin(math.pi * fase)

        data.ctrl[act_rot_hip_z] = -math.radians(75) * prep + math.radians(150) * onda_1
        data.ctrl[act_rot_hip_y] = math.radians(15) + math.radians(35) * onda_2
        data.ctrl[act_rot_hip_x] = math.radians(30) * onda_1
        data.ctrl[act_left_arm_x] = -ANGULO_BRACOS
        data.ctrl[act_left_arm_y] = -math.radians(45) - math.radians(55) * prep + math.radians(15) * onda_2
        data.ctrl[act_right_arm_x] = ANGULO_BRACOS
        data.ctrl[act_right_arm_y] = -math.radians(45) - math.radians(55) * prep + math.radians(15) * onda_2
        data.ctrl[act_right_elbow] = -math.radians(20) * prep + math.radians(20) * onda_2
        data.ctrl[act_wrist_y] = (-math.radians(100) * prep) + math.radians(100) * onda_2
        data.ctrl[act_wrist_x] = -ANGULO_BRACOS

        # TASK 2 - sobrepõe um tremor a TODOS os DOF em simultâneo, cada um com a sua
        # própria amplitude e fase (ver dicionários acima). A perturbação atua durante
        # todo o movimento (preparação + espera + swing), tal como um tremor real não
        # "desliga" antes de uma ação voluntária começar.
        for nome, act_id in ACTUADORES.items():
            ruido = AMPLITUDE_RUIDO[nome] * math.sin(2 * math.pi * FREQ_RUIDO * t + FASE_RUIDO[nome])
            data.ctrl[act_id] += ruido

        mujoco.mj_step(model, data)
        v.sync()

        resto = model.opt.timestep - (time.time() - inicio)
        if resto > 0:
            time.sleep(resto)