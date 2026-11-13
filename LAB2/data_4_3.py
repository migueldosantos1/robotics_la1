'''
Script para analisar o erro DK - robô

Definido entre a DK aplicada às juntas reportadas pelo robô
e a posição reportada pelo robô - permite isolar o erro provocado pelo modelo
que estamos a usar do modelo interno do robô

'''

import numpy as np
import matplotlib.pyplot as plt
from matrices import tip_position

np.set_printoptions(precision=3, suppress=True)

def wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0

def summarize(err, label):
    pos = np.linalg.norm(err[:, :3], axis=1)
    print(f"\n - {label} (N = {len(err)})")
    print(f"Erro de posição médio: {pos.mean():.3f} mm")
    print(f"Erro de posição máximo: {pos.max():.3f} mm")
    print(f"Vetor médio (x, y, z): {err[:, :3].mean(0)} mm")
    print(f"Desvio-padrão: {err[:, :3].std(0)} mm")
    for i, n in enumerate(["roll", "pitch", "yaw"]):
        e = np.abs(err[:, 3 + i])
        print(f"Erro de {n}: média {e.mean():.4f}°; máx {e.max():.4f}°")
    return pos

# Durante a execução: usa as amostras BRUTAS do log (sem interpolar)
def analyze_run(file_path="traj_run_slow1.npz", still_thr=10.0):
    log = np.load(file_path)["log"]
    t = log[:, 0]
    robot = log[:, 1:7]
    q = log[:, 7:13]

    dk = np.array([tip_position(qi) for qi in q])
    err = dk - robot
    err[:, 3:] = wrap(err[:, 3:])
    pos = summarize(err, "Execução: DK vs pose do robô (amostras brutas)")

    # velocidade do TCP (janela de 5 amostras para reduzir ruído)
    k = 5
    v = np.full(len(t), np.nan)
    v[:-k] = np.linalg.norm(robot[k:, :3] - robot[:-k, :3], axis=1) / (t[k:] - t[:-k])

    still = v < still_thr
    moving = v >= still_thr
    if still.any():
        print(f"\nCom o robô (quase) parado, v < {still_thr:.0f} mm/s (N = {still.sum()}):")
        print(f"  erro médio {pos[still].mean():.3f} mm | vetor médio {err[still, :3].mean(0)}")
    if moving.any():
        print(f"Em movimento (N = {moving.sum()}): erro médio {pos[moving].mean():.3f} mm")
        c = np.corrcoef(v[moving], pos[moving])[0, 1]
        print(f"Correlação erro-velocidade (em movimento): {c:.2f}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, n in enumerate(["X", "Y", "Z"]):
        axs[0].plot(t, err[:, i], label=f"Erro {n}")
    axs[0].set_xlabel("Tempo (s)"); axs[0].set_ylabel("DK − robô (mm)")
    axs[0].set_title("Componentes do erro"); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(t, pos, "g-")
    axs[1].set_xlabel("Tempo (s)"); axs[1].set_ylabel("Erro (mm)")
    axs[1].set_title("Norma do erro"); axs[1].grid(True)

    axs[2].plot(v, pos, "k.", markersize=3)
    axs[2].set_xlabel("Velocidade do TCP (mm/s)"); axs[2].set_ylabel("Erro (mm)")
    axs[2].set_title("Erro vs velocidade"); axs[2].grid(True)
    fig.tight_layout()
    fig.savefig("dk_vs_robo_execucao.png", dpi=200)
    return err

# Comparação com o vetor médio do erro da IK (ik_errors.npz)
def compare_with_ik(err_dk_robot, path="ik_errors.npz"):
    ik_err = np.load(path)["err"]
    m_ik = ik_err[:, :3].mean(0)
    m_rob = err_dk_robot[:, :3].mean(0)
    print("\nVetor médio do erro (x, y, z), em mm:")
    print(f"  IK  (DK - desejada): {m_ik}")
    print(f"  DK - pose do robô  : {m_rob}")
    cos = m_ik @ m_rob / (np.linalg.norm(m_ik) * np.linalg.norm(m_rob))
    print(f"  Cosseno entre os dois vetores: {cos:.2f}  (próximo de 1 => mesma origem)")

if __name__ == "__main__":
    err_run = analyze_run("traj_run_slow1.npz")
    try:
        compare_with_ik(err_run, "ik_errors.npz")
    except FileNotFoundError:
        print("\n(ik_errors.npz não encontrado: salta a comparação com a IK)")
    plt.show()