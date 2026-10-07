import os
import time
import math
import psutil
import mujoco
import mujoco.viewer

# --- fechar outras instâncias deste script (não a atual) ---------------
for p in psutil.process_iter(["pid", "cmdline"]):
    cmdline = " ".join(p.info["cmdline"] or [])
    if p.pid != os.getpid() and "teste_bracos.py" in cmdline:
        p.terminate()

model = mujoco.MjModel.from_xml_path("main_v5.xml")
data = mujoco.MjData(model)

# --- joints a testar, por ordem de aparição no varrimento ---------------
NOMES = [
    "left_arm_y", "left_arm_x",
    "right_arm_y", "right_arm_x",
    "left_elbow", "right_elbow",
]

enderecos = {nome: model.joint(nome).qposadr for nome in NOMES}

# amplitude do varrimento de cada joint: metade do seu próprio range
# (lido diretamente do XML, para nunca ultrapassar os limites definidos)
amplitudes = {}
for nome in NOMES:
    jid = model.joint(nome).id
    lo, hi = model.jnt_range[jid]
    centro = (lo + hi) / 2
    amplitudes[nome] = (centro, (hi - lo) / 2)

DURACAO_POR_FASE = 5.0  # segundos que cada joint individual demora a testar

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    t0 = time.time()
    while v.is_running():
        t = time.time() - t0

        # uma fase por joint, mais uma fase final com todos em simultâneo
        n_fases = len(NOMES) + 1
        fase = int(t // DURACAO_POR_FASE) % n_fases
        fase_t = (t % DURACAO_POR_FASE) / DURACAO_POR_FASE
        onda = math.sin(2 * math.pi * fase_t)  # oscila suavemente entre -1 e 1

        # repõe tudo no centro do seu range antes de aplicar a fase atual
        for nome in NOMES:
            centro, _ = amplitudes[nome]
            data.qpos[enderecos[nome]] = centro

        if fase < len(NOMES):
            nome_ativo = NOMES[fase]
            centro, amp = amplitudes[nome_ativo]
            data.qpos[enderecos[nome_ativo]] = centro + amp * onda
            if int(t) % 1 == 0 and abs(t % DURACAO_POR_FASE) < 0.02:
                print(f"a testar: {nome_ativo}")
        else:
            # fase final: todos os joints a variar ao mesmo tempo
            for nome in NOMES:
                centro, amp = amplitudes[nome]
                data.qpos[enderecos[nome]] = centro + amp * onda
            if abs(t % DURACAO_POR_FASE) < 0.02:
                print("a testar: todos em simultâneo")

        mujoco.mj_forward(model, data)
        v.sync()
        time.sleep(0.01)