import numpy as np

# Parâmetros globais da trajetória da tacada
T = 3.0       # Duração total reduzida (mais rápido, mas controlado)
PSI = 90.0    # Direção transversal (eixo Y)

def traj(t):
    """
    Gera uma trajetória 3D expressiva de golfe (Direita -> Impacto -> Esquerda)
    otimizada com perfis temporais suaves para suportar velocidades mais elevadas.
    """
    # Normalizar o tempo para o intervalo [0, 1]
    s = np.clip(t / T, 0.0, 1.0)

    # --- PERFIL TEMPORAL SUAVE (Cosseno Acoplado) ---
    # Garanta transições de velocidade e aceleração contínuas nos extremos (sem solavancos)
    s_smooth = 0.5 * (1.0 - np.cos(np.pi * s))

    # --- CENTRO DO ESPAÇO DE TRABALHO DO LITE6 ---
    x_center = 260.0
    y_center = 0.0
    z_center = 140.0

    # --- 1. VARRIMENTO LATERAL (Y): Da Direita (+160mm) para a Esquerda (-160mm) ---
    # Usamos s direto para fazer a viagem completa de um lado ao outro de forma fluida
    y_offset = 160.0 * np.cos(np.pi * s)

    # --- 2. PROFUNDIDADE (X): O "Loop" do Taco ---
    # Afasta-se no início/fim e projeta-se para a frente no momento exato do impacto (s = 0.5)
    x_offset = 50.0 * np.sin(np.pi * s)

    # --- 3. PERFIL VERTICAL (Z): Parábola do Balanço ---
    # Mais alto nas pontas, atingindo o ponto mais baixo (impacto) no centro
    z_height = z_center + 110.0 * (2.0 * s_smooth - 1.0)**2

    # Coordenadas cartesianas finais
    x = x_center + x_offset
    y = y_center + y_offset
    z = z_height

    # --- 4. ORIENTAÇÃO ESTÁVEL E EXPRESSIVA ---
    # Evita rotações violentas do pulso mantendo o roll fixo e variando o pitch/yaw de forma suave
    roll = 180.0
    pitch = 10.0 * np.sin(np.pi * s)        # Inclinação leve do taco na subida e descida
    yaw = PSI + 20.0 * np.cos(np.pi * s)    # Adaptação suave à direção do arco

    return np.array([x, y, z, roll, pitch, yaw])