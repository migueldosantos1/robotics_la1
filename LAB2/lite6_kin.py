"""
lite6_kin.py - Cinematica directa e inversa (numerica) do UFactory Lite 6.

Convencao: Denavit-Hartenberg MODIFICADO (Craig), igual a usada pela UFactory:
    T^{i-1}_i = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i + offset_i) * Tz(d_i)

Unidades: mm e graus nas interfaces "xArm" (pose = [x,y,z,roll,pitch,yaw]);
radianos internamente nas juntas.
Convencao de angulos xArm: R = Rz(yaw) * Ry(pitch) * Rx(roll)  (eixos fixos da base).
"""
import numpy as np

# ---------------------------------------------------------------------------
# Parametros DH modificados do Lite 6 (documentacao UFactory; VERIFICAR com a
# vossa versao do manual - o script ik_accuracy.py valida-os contra o robot).
# Linha i:  a_{i-1} [mm], alpha_{i-1} [rad], d_i [mm], offset_i [rad]
# ---------------------------------------------------------------------------
DH = np.array([
    #  a      alpha          d       offset
    [  0.0,   0.0,         243.3,   0.0        ],
    [  0.0,  -np.pi/2,       0.0,  -np.pi/2    ],
    [200.0,   np.pi,         0.0,  -np.pi/2    ],
    [ 87.0,   np.pi/2,     227.6,   0.0        ],
    [  0.0,   np.pi/2,       0.0,   0.0        ],
    [  0.0,  -np.pi/2,      61.5,   0.0        ],
])

# Limites de junta [graus] (datasheet Lite 6)
Q_MIN_DEG = np.array([-360.0, -150.0,  -3.5, -360.0, -124.0, -360.0])
Q_MAX_DEG = np.array([ 360.0,  150.0, 300.0,  360.0,  124.0,  360.0])
Q_MIN = np.radians(Q_MIN_DEG)
Q_MAX = np.radians(Q_MAX_DEG)

REACH_MM = 440.0          # alcance maximo do datasheet
QD_MAX_DEG = 200.0        # limite de velocidade de junta indicado no enunciado


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def dh_transform(a, alpha, d, theta):
    """Transformacao homogenea 4x4 (DH modificado)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,      -st,      0.0,  a],
        [st * ca,  ct * ca, -sa, -sa * d],
        [st * sa,  ct * sa,  ca,  ca * d],
        [0.0,      0.0,     0.0,  1.0],
    ])


def fk_all(q):
    """Lista das 7 transformacoes {0}->{i}, i=0..6 (q em rad)."""
    T = np.eye(4)
    out = [T.copy()]
    for i in range(6):
        a, al, d, off = DH[i]
        T = T @ dh_transform(a, al, d, q[i] + off)
        out.append(T.copy())
    return out


def fk(q):
    """Transformacao base->flange (q em rad)."""
    return fk_all(q)[-1]


def rpy_to_R(roll, pitch, yaw):
    """Angulos em rad -> R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def R_to_rpy(R):
    """R -> (roll, pitch, yaw) em rad (convencao xArm)."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # singularidade (gimbal lock)
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def fk_pose(q_deg):
    """q [graus] -> pose xArm [x,y,z (mm), roll,pitch,yaw (graus)]."""
    T = fk(np.radians(q_deg))
    r, p, y = R_to_rpy(T[:3, :3])
    return np.array([*T[:3, 3], *np.degrees([r, p, y])])


def pose_to_T(pose):
    """pose xArm [mm, graus] -> T 4x4."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*np.radians(pose[3:6]))
    T[:3, 3] = pose[:3]
    return T


def _rotvec(R):
    """Vetor de rotacao (eixo*angulo) de R."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(ang - np.pi) < 1e-6:
        w = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        return ang * w / (np.linalg.norm(w) + 1e-12)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang * w / (2 * np.sin(ang))


def _jacobian(q, eps=1e-6):
    """Jacobiana numerica 6x6 (v em mm/rad ; w em rad/rad), expressa na base."""
    T0 = fk(q)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = q.copy()
        dq[i] += eps
        T1 = fk(dq)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        J[3:, i] = _rotvec(T1[:3, :3] @ T0[:3, :3].T) / eps
    return J


def ik_numeric(T_des, q0, tol_pos=1e-6, tol_rot=1e-9, max_iter=200, lam=1e-2):
    """
    Cinematica inversa numerica (Levenberg-Marquardt) com limites de junta.
    T_des: 4x4 (mm). q0: semente em rad (continuidade da trajetoria).
    Devolve (q, convergiu).
    """
    q = np.clip(np.array(q0, float), Q_MIN, Q_MAX)
    for _ in range(max_iter):
        T = fk(q)
        e = np.hstack([T_des[:3, 3] - T[:3, 3], _rotvec(T_des[:3, :3] @ T[:3, :3].T)])
        # escala: 1 mm ~ 1 rad nao e fisico; pesos simples para equilibrar
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
            return q, True
        J = _jacobian(q)
        W = np.diag([1, 1, 1, 100, 100, 100.0])  # 1 rad ~ 100 mm
        A = J.T @ W @ J + lam * np.eye(6)
        dq = np.linalg.solve(A, J.T @ W @ e)
        q = np.clip(q + dq, Q_MIN, Q_MAX)
    T = fk(q)
    e = np.hstack([T_des[:3, 3] - T[:3, 3], _rotvec(T_des[:3, :3] @ T[:3, :3].T)])
    return q, (np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-6)
