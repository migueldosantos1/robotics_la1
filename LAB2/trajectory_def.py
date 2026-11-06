import numpy as np

# --- Parâmetros (mm, graus, s) ---
B     = np.array([320.0, 0.0, 80.0])   # posição da bola
L     = 140.0                          # raio do arco (comprimento do pêndulo)
DELTA = 10.0                           # shaft lean (graus)
PSI   = 0.0                            # direção da tacada no plano horizontal (0 = +x)
TH_BACK, TH_FWD, TH_IMP = 25.0, 30.0, 2.0   # backswing, follow-through, ângulo no impacto (graus)
T     = 1.0                            # duração do swing (s)

# --- Geometria fixa (calculada uma vez) ---
d  = np.array([np.cos(np.radians(PSI)), np.sin(np.radians(PSI)), 0.0])  # direção horizontal da tacada
ez = np.array([0.0, 0.0, 1.0])                                          # vertical
th_imp = np.radians(TH_IMP)
C  = B - L * (np.sin(th_imp) * d - np.cos(th_imp) * ez)   # pivô: faz o TCP passar na bola em th_imp

def min_jerk(tau):
    # Definiu-se esta função em [0,1] e garante que as velocidades/acelerações iniciais e finais são nulas (smoothing)
    tau = np.clip(tau, 0, 1)
    return 10*tau**3 - 15*tau**4 + 6*tau**5

def traj(t):
    # Definição da função que gera os vários pontos da tragetória -> Devolve o pto. do instante t
    th = np.radians(-TH_BACK) + np.radians(TH_BACK + TH_FWD) * min_jerk(t / T)  # ângulo do pêndulo
    p  = C + L * (np.sin(th) * d - np.cos(th) * ez)                              # posição no arco
    return np.array([*p, 180.0, DELTA - np.degrees(th), PSI])                    # orientação (forma fechada)