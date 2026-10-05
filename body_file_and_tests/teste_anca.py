import time
import math
import mujoco
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("main.xml")
data = mujoco.MjData(model)

#definir as juntas a serem alteradas
addr_z = model.joint("rot_anca_z").qposadr
addr_y = model.joint("rot_anca_y").qposadr
addr_x = model.joint("rot_anca_x").qposadr

# amplitudes de varrimento, em radianos (dentro dos intervalos do XML)
AMPLITUDE_Z = math.radians(60)   # [-80, 80]
AMPLITUDE_Y = math.radians(25)   # [-15, 40]
AMPLITUDE_X = math.radians(25)   # [-30, 30]

DURACAO_POR_FASE = 6.0  # segundos que cada varrimento demora

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    t0 = time.time()
    while v.is_running():
        t = time.time() - t0

        # fase 0: só Z | fase 1: só Y | fase 2: só X | fase 3: os 3 juntos
        fase = int(t // DURACAO_POR_FASE) % 4
        fase_t = (t % DURACAO_POR_FASE) / DURACAO_POR_FASE  # 0 a 1 dentro da fase
        onda = math.sin(2 * math.pi * fase_t)  # oscila suavemente entre -1 e 1

        data.qpos[addr_z] = 0.0
        data.qpos[addr_y] = 0.0
        data.qpos[addr_x] = 0.0
        if fase == 0:
            data.qpos[addr_z] = AMPLITUDE_Z * onda
        elif fase == 1:
            data.qpos[addr_y] = AMPLITUDE_Y * onda
        elif fase == 2:
            data.qpos[addr_x] = AMPLITUDE_X * onda
        else:
            data.qpos[addr_z] = AMPLITUDE_Z * onda
            data.qpos[addr_y] = AMPLITUDE_Y * onda
            data.qpos[addr_x] = AMPLITUDE_X * onda

        mujoco.mj_forward(model, data)
        v.sync()
        time.sleep(0.01)