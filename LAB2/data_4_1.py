'''
Script para gerar as figuras e métricas da 4.1

Trajetória desejada (analítica densa) vs. pontos devolvidos pela cinemática inversa
Norma do erro de posição ao longo do tempo
Componentes do erro de posição (X, Y, Z) e de orientação (Roll, Pitch, Yaw)

'''

import numpy as np
import matplotlib.pyplot as plt
from trajectory_def import traj, T

def main():
    # Carregar os dados obtidos na validação da cinemática inversa
    d = np.load("ik_errors.npz")
    ts = d["ts"]
    calc = d["calc"]
    err = d["err"]

    # Calcular erros de posição e orientação para o terminal
    pos_err = np.linalg.norm(err[:, :3], axis=1)
    print(f"Erro de posição: média {np.nanmean(pos_err):.3f} mm | máx {np.nanmax(pos_err):.3f} mm")
    for i, n in enumerate(["roll", "pitch", "yaw"]):
        e = np.abs(err[:, 3 + i])
        print(f"Erro de {n}: média {np.nanmean(e):.4f}° | máx {np.nanmax(e):.4f}°")

    # --- 1. Trajetória desejada (analítica densa) vs obtida pela IK ---
    tt = np.linspace(0, T, 500)
    dense = np.array([traj(t) for t in tt])
    
    planos = [
        (0, 1, "X (mm)", "Y (mm)", "Plano XY"),
        (0, 2, "X (mm)", "Z (mm)", "Plano XZ"),
        (1, 2, "Y (mm)", "Z (mm)", "Plano YZ")
    ]
    
    fig1, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (i, j, xl, yl, title) in zip(axs, planos):
        ax.plot(dense[:, i], dense[:, j], "r--", linewidth=1.5, label="Desejada")
        ax.plot(calc[:, i], calc[:, j], "b.", markersize=6, label="DK sobre IK")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(True)
        ax.axis("equal")
        if ax is axs[0]:
            ax.legend(loc="center left", fontsize=8)
            
    fig1.tight_layout()
    fig1.savefig("ik_trajetoria.png", dpi=200, bbox_inches="tight")

    # --- 2. Norma do erro de posição ao longo do tempo ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(ts, pos_err, "g-o", markersize=4)
    ax2.set_title("Norma do erro de posição")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Erro (mm)")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig("ik_erro_posicao.png", dpi=200, bbox_inches="tight")

    # --- 3. Componentes dos erros (Posição e Orientação) ---
    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 4.5))

    # Erros de posição (X, Y, Z)
    for i, n in enumerate(["X", "Y", "Z"]):
        axs3[0].plot(ts, err[:, i], "-o", markersize=4, label=f"Erro {n}")
    axs3[0].set_title("Componentes do erro de posição")
    axs3[0].set_xlabel("Tempo (s)")
    axs3[0].set_ylabel("Erro (mm)")
    axs3[0].grid(True)
    axs3[0].legend()

    # Erros de orientação (Roll, Pitch, Yaw)
    for i, n in enumerate(["Roll", "Pitch", "Yaw"]):
        axs3[1].plot(ts, err[:, 3 + i], "-o", markersize=4, label=n)
    axs3[1].set_title("Componentes do erro de orientação")
    axs3[1].set_xlabel("Tempo (s)")
    axs3[1].set_ylabel("Erro (graus)")
    axs3[1].grid(True)
    axs3[1].legend()

    fig3.tight_layout()
    fig3.savefig("ik_erros_componentes.png", dpi=200, bbox_inches="tight")

    print("\nFiguras geradas e guardadas com sucesso:")
    print("- ik_trajetoria.png")
    print("- ik_erro_posicao.png")
    print("- ik_erros_componentes.png")

    plt.show()

if __name__ == "__main__":
    main()