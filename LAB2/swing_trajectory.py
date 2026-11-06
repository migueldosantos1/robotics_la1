"""
swing_trajectory.py - Tarefa 2: trajetoria de "tacada" para o Lite 6.

Modelo geometrico (analogo ao swing do Lab 1):
  * A cabeca do "taco" (TCP/gripper) descreve um arco circular de raio R_ARM
    num plano vertical (plano x-z da base), centrado no ponto C (o "ombro" do golfista).
  * A bola esta no ponto mais baixo do arco (phi = 0).
  * phi(t): 0 -> -PHI_BACK (backswing) -> +PHI_FOLLOW (downswing + impacto + follow-through)
            -> 0 (retorno). Cada segmento e repouso-a-repouso com polinomio de 5a ordem
            (posicao, velocidade e aceleracao continuas, sem impulsos no arranque).
  * Orientacao: ferramenta aponta para baixo (roll=180) e roda em torno do eixo y da
    base acompanhando a tangente do arco (a cabeca do taco "segue" o arco).

Executar:  python swing_trajectory.py    -> gera graficos + trajectory.npz + relatorio no ecra
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lite6_kin import (fk, fk_all, ik_numeric, rpy_to_R, R_to_rpy, _ry, _rx,
                       Q_MIN_DEG, Q_MAX_DEG, QD_MAX_DEG, REACH_MM)

# ----------------------------- parametros da tacada --------------------------
BALL = np.array([300.0, 0.0, 100.0])   # [mm] posicao fixa da bola (base do robot = origem)
R_ARM = 100.0                          # [mm] raio do arco
PHI_BACK = np.radians(60.0)            # amplitude do backswing
PHI_FOLLOW = np.radians(50.0)          # amplitude do follow-through
TILT_GAIN = 0.6                        # fracao de phi usada como inclinacao da ferramenta
T_BACK, T_DOWN, T_RET = 2.0, 1.6, 2.0  # duracao de cada fase [s]
DT = 0.01                              # passo de amostragem 100 Hz
Z_TABLE = 0.0                          # plano da mesa [mm]
Z_SAFE = 30.0                          # folga minima a mesa para os elos [mm]
BASE_KEEPOUT_R = 60.0                  # raio de exclusao em torno do eixo da base (so para pontos TCP) [mm]
VEL_SAFETY = 0.75                      # usar no maximo 75% do limite de velocidade

CENTER = BALL + np.array([0.0, 0.0, R_ARM])


def quintic(tau):
    """Perfil repouso-a-repouso s(tau), tau in [0,1]; devolve s, ds/dtau."""
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    return s, ds


def phi_profile(dt=DT):
    """phi(t) e dphi/dt(t)."""
    segs = [(0.0, -PHI_BACK, T_BACK), (-PHI_BACK, PHI_FOLLOW, T_DOWN), (PHI_FOLLOW, 0.0, T_RET)]
    t_all, ph, dph = [], [], []
    t0 = 0.0
    for (a, b, T) in segs:
        n = int(round(T / dt))
        tau = np.arange(n) / n
        s, ds = quintic(tau)
        ph.append(a + (b - a) * s)
        dph.append((b - a) * ds / T)
        t_all.append(t0 + tau * T)
        t0 += T
    t_all.append([t0]); ph.append([0.0]); dph.append([0.0])
    return np.concatenate(t_all), np.concatenate(ph), np.concatenate(dph)


def pose_from_phi(phi):
    """Posicao e matriz de rotacao do TCP para um dado angulo de arco."""
    p = CENTER + R_ARM * np.array([np.sin(phi), 0.0, -np.cos(phi)])
    R = _ry(TILT_GAIN * phi) @ _rx(np.pi)      # ferramenta para baixo + inclinacao
    return p, R


def build_cartesian():
    t, phi, dphi = phi_profile()
    P = np.zeros((len(t), 3)); RPY = np.zeros((len(t), 3))
    for k, ph in enumerate(phi):
        p, R = pose_from_phi(ph)
        P[k] = p
        RPY[k] = np.degrees(R_to_rpy(R))
    # desembrulhar roll (+-180) para evitar saltos nos graficos/comandos
    RPY[:, 0] = np.degrees(np.unwrap(np.radians(RPY[:, 0]))) % 360.0
    RPY[:, 0] = np.where(RPY[:, 0] > 180, RPY[:, 0] - 360.0, RPY[:, 0])
    return t, phi, dphi, P, RPY


def solve_joint_trajectory(t, phi, q_seed_deg):
    """IK ponto a ponto com continuidade (semente = solucao anterior)."""
    Q = np.zeros((len(t), 6)); ok = np.zeros(len(t), bool)
    q = np.radians(q_seed_deg)
    for k, ph in enumerate(phi):
        p, R = pose_from_phi(ph)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
        q, ok[k] = ik_numeric(T, q)
        Q[k] = q
    return np.degrees(Q), ok


def find_start_config():
    """Procura uma configuracao inicial (cotovelo 'para cima') para o primeiro ponto."""
    p, R = pose_from_phi(0.0)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    best = None
    rng = np.random.default_rng(0)
    for _ in range(300):
        q0 = np.radians(rng.uniform([-30, -60, 20, -30, -90, -30], [30, 60, 160, 30, 90, 30]))
        q, ok = ik_numeric(T, q0)
        if ok and (best is None or np.linalg.norm(q - np.radians([0, 0, 90, 0, 0, 0])) <
                   np.linalg.norm(best - np.radians([0, 0, 90, 0, 0, 0]))):
            best = q
    return np.degrees(best)


def check_feasibility(t, Q, P):
    """Verificacoes de seguranca/exequibilidade. Devolve dict com resultados."""
    res = {}
    qd = np.gradient(Q, t, axis=0)                        # deg/s
    res["qd_max_per_joint"] = np.abs(qd).max(axis=0)
    res["qd_max"] = np.abs(qd).max()
    res["vel_ok"] = res["qd_max"] <= VEL_SAFETY * QD_MAX_DEG
    res["q_min"] = Q.min(axis=0); res["q_max"] = Q.max(axis=0)
    res["limits_ok"] = bool(np.all(Q >= Q_MIN_DEG) and np.all(Q <= Q_MAX_DEG))
    # alcance: distancia ombro (0,0,243.3) -> TCP < 440 mm
    shoulder = np.array([0, 0, 243.3])
    res["reach_max"] = np.linalg.norm(P - shoulder, axis=1).max()
    res["reach_ok"] = res["reach_max"] < REACH_MM
    # velocidade cartesiana do TCP
    v = np.linalg.norm(np.gradient(P, t, axis=0), axis=1)
    res["tcp_v_max"] = v.max()
    # mesa: z minimo das origens de todos os frames e do TCP
    zmin = np.inf; rmin = np.inf
    for q in np.radians(Q):
        Ts = fk_all(q)
        zs = [T[2, 3] for T in Ts[2:]]       # frames 2..6 (exclui base fixa)
        zmin = min(zmin, min(zs))
    res["z_min_links"] = zmin
    res["table_ok"] = zmin > Z_TABLE + Z_SAFE
    res["tcp_xy_min"] = np.linalg.norm(P[:, :2], axis=1).min()
    res["base_ok"] = res["tcp_xy_min"] > BASE_KEEPOUT_R
    # velocidade no impacto
    i0, i1 = int(round(T_BACK / DT)), int(round((T_BACK + T_DOWN) / DT))   # fase de downswing
    k_imp = i0 + int(np.argmin(np.abs(P[i0:i1, 0] - BALL[0])))             # phi = 0 durante o downswing
    res["impact_idx"] = k_imp
    res["impact_speed_mm_s"] = v[k_imp]
    return res, qd, v


def plots(t, phi, P, RPY, Q, qd, v, res, out="."):
    # --- 1) trajetoria 3D + plano xz ------------------------------------------------
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*P.T, "b", lw=2, label="TCP")
    ax.scatter(*BALL, c="r", s=60, label="Bola")
    ax.scatter(0, 0, 0, c="k", marker="s", label="Base")
    xx, yy = np.meshgrid([-100, 450], [-250, 250]); ax.plot_surface(xx, yy, 0 * xx, alpha=0.15, color="gray")
    for k in range(0, len(t), 25):
        Ts = fk_all(np.radians(Q[k])); pts = np.array([T[:3, 3] for T in Ts])
        ax.plot(*pts.T, color="orange", alpha=0.35)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]"); ax.legend()
    ax.set_title("Trajetoria 3D, elos (laranja) e mesa (z=0)")
    ax.set_box_aspect((1.2, 1, 0.8))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(P[:, 0], P[:, 2], "b", lw=2, label="TCP (arco)")
    ax2.plot(BALL[0], BALL[2], "ro", label="Bola")
    ax2.axhline(Z_TABLE, color="gray", label="Mesa")
    ax2.axhline(Z_TABLE + Z_SAFE, color="r", ls="--", lw=0.8, label="Folga minima")
    th = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(REACH_MM * np.cos(th), 243.3 + REACH_MM * np.sin(th), "g:", label="Alcance 440 mm")
    for k in range(0, len(t), 40):
        Ts = fk_all(np.radians(Q[k])); pts = np.array([T[:3, 3] for T in Ts])
        ax2.plot(pts[:, 0], pts[:, 2], color="orange", alpha=0.4)
    ax2.set_xlim(-100, 500); ax2.set_ylim(-20, 700); ax2.set_aspect("equal")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("z [mm]"); ax2.legend(fontsize=7); ax2.set_title("Vista lateral (plano x-z)")
    fig.tight_layout(); fig.savefig(f"{out}/fig_trajectory_3d.png", dpi=150); plt.close(fig)

    # --- 2) posicao e orientacao do TCP no tempo --------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for i, n in enumerate("xyz"):
        ax[0, 0].plot(t, P[:, i], label=n)
    ax[0, 0].set_title("Posicao do TCP [mm]"); ax[0, 0].legend()
    for i, n in enumerate(["roll", "pitch", "yaw"]):
        ax[0, 1].plot(t, RPY[:, i], label=n)
    ax[0, 1].set_title("Orientacao do TCP [graus]"); ax[0, 1].legend()
    ax[1, 0].plot(t, v, "b"); ax[1, 0].axvline(t[res["impact_idx"]], color="r", ls="--", label="impacto")
    ax[1, 0].axhline(500, color="k", ls=":", label="500 mm/s (max. TCP datasheet)")
    ax[1, 0].set_title("Velocidade do TCP [mm/s]"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, np.degrees(phi)); ax[1, 1].set_title("phi(t) [graus] (angulo do arco)")
    for a in ax.ravel(): a.set_xlabel("t [s]"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/fig_tcp_vs_time.png", dpi=150); plt.close(fig)

    # --- 3) juntas: posicao e velocidade -----------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    for i in range(6):
        l, = ax[0].plot(t, Q[:, i], label=f"J{i+1}")
        ax[0].axhline(Q_MAX_DEG[i], color=l.get_color(), ls=":", lw=0.6, alpha=0.6) if Q_MAX_DEG[i] < 200 else None
        ax[0].axhline(Q_MIN_DEG[i], color=l.get_color(), ls=":", lw=0.6, alpha=0.6) if Q_MIN_DEG[i] > -200 else None
        ax[1].plot(t, qd[:, i], label=f"J{i+1}")
    ax[0].set_title("Posicao das juntas [graus] (tracejado: limites fisicos)"); ax[0].legend(ncol=6, fontsize=8)
    for s in (+1, -1):
        ax[1].axhline(s * QD_MAX_DEG, color="r", ls="--"); ax[1].axhline(s * VEL_SAFETY * QD_MAX_DEG, color="orange", ls=":")
    ax[1].set_title("Velocidade das juntas [graus/s] (vermelho: 200 deg/s; laranja: 75% = 150 deg/s)")
    ax[1].legend(ncol=6, fontsize=8)
    for a in ax: a.set_xlabel("t [s]"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/fig_joints.png", dpi=150); plt.close(fig)

    # --- 4) alturas dos elos (anti-colisao com a mesa) ---------------------------------
    zs = np.array([[T[2, 3] for T in fk_all(np.radians(q))[2:]] for q in Q])
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(zs.shape[1]):
        ax.plot(t, zs[:, i], label=f"origem do frame {i+2}")
    ax.axhline(Z_TABLE, color="gray", label="mesa"); ax.axhline(Z_TABLE + Z_SAFE, color="r", ls="--", label="folga minima")
    ax.set_xlabel("t [s]"); ax.set_ylabel("z [mm]"); ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    ax.set_title("Altura das articulacoes e do flange vs. mesa")
    fig.tight_layout(); fig.savefig(f"{out}/fig_clearance.png", dpi=150); plt.close(fig)


def main():
    t, phi, dphi, P, RPY = build_cartesian()
    q_seed = find_start_config()
    print("Configuracao inicial (graus):", np.round(q_seed, 2))
    Q, ok = solve_joint_trajectory(t, phi, q_seed)
    print("IK convergiu em todos os pontos:", ok.all(), f"({ok.sum()}/{len(ok)})")
    res, qd, v = check_feasibility(t, Q, P)
    print("\n=== Verificacao de exequibilidade ===")
    print(f"Duracao total: {t[-1]:.2f} s | pontos: {len(t)}")
    print("Vel. max por junta [deg/s]:", np.round(res["qd_max_per_joint"], 1))
    print(f"Vel. max global: {res['qd_max']:.1f} deg/s (limite {QD_MAX_DEG:.0f}; alvo <= {VEL_SAFETY*QD_MAX_DEG:.0f}) -> {'OK' if res['vel_ok'] else 'FALHA'}")
    print("Q min [deg]:", np.round(res["q_min"], 1)); print("Q max [deg]:", np.round(res["q_max"], 1))
    print("Limites de junta respeitados:", res["limits_ok"])
    print(f"Alcance max (ombro->TCP): {res['reach_max']:.1f} mm (< {REACH_MM}) -> {res['reach_ok']}")
    print(f"Vel. TCP max: {res['tcp_v_max']:.0f} mm/s | no impacto: {res['impact_speed_mm_s']:.0f} mm/s")
    print(f"z minimo dos elos: {res['z_min_links']:.1f} mm (folga > {Z_SAFE}) -> {res['table_ok']}")
    print(f"Distancia horizontal minima TCP-eixo da base: {res['tcp_xy_min']:.1f} mm -> {res['base_ok']}")
    plots(t, phi, P, RPY, Q, qd, v, res, out=".")
    np.savez("trajectory.npz", t=t, phi=phi, P=P, RPY=RPY, Q=Q, q_seed=q_seed)
    return res


if __name__ == "__main__":
    main()
