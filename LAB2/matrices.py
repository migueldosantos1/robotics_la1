import numpy as np
import math
from math import sin, cos



def robot_dynamics(theta1, theta2, theta3, theta4, theta5, theta6):
       # Configura o NumPy para esconder os números flutuantes residuais muito pequenos (ex: 6.12e-17 vira 0)
       np.set_printoptions(suppress=True, precision=4)

       # Declaração direta com np.array([ ... ])
       T01 = np.array([
       [cos(theta1), -sin(theta1), 0, 0],
       [sin(theta1), cos(theta1), 0, 0],
       [0, 0, 1, 243.3],
       [0, 0, 0, 1]
       ])

       T12 = np.array([
       [cos((math.radians(-90) + theta2)), -sin((math.radians(-90) + theta2)), 0, 0],
       [0, 0, 1, 0],
       [-sin((math.radians(-90) + theta2)), -cos((math.radians(-90) + theta2)), 0, 0],
       [0, 0, 0, 1]
       ])

       T23 = np.array([
       [cos((math.radians(-90) + theta3)), -sin((math.radians(-90) + theta3)), 0, 200],
       [-sin((math.radians(-90) + theta3)), -cos((math.radians(-90) + theta3)), 0, 0],
       [0, 0, -1, 0],
       [0, 0, 0, 1]
       ])

       T34 = np.array([
       [cos(theta4), -sin(theta4), 0, 87],
       [0, 0, -1, -227.6],
       [sin(theta4), cos(theta4), 0, 0],
       [0, 0, 0, 1]
       ])

       T45 = np.array([
       [cos(theta5), -sin(theta5), 0, 0],
       [0, 0, -1, 0],
       [sin(theta5), cos(theta5), 0, 0],
       [0, 0, 0, 1]
       ])

       T56 = np.array([
       [cos(theta6), -sin(theta6), 0, 0],
       [0, 0, 1, 61.5],
       [-sin(theta6), -cos(theta6), 0, 0],
       [0, 0, 0, 1]
       ])

       # Isto calcula a transformação total da base (0) até ao referencial (6)
       T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

       return T06


def tip_position(q_deg):
    T06 = robot_dynamics(np.radians(q_deg))          
    x, y, z = T06[:3, 3] # Extrai a posição (x, y, z) da matriz de transformação T06
    roll, pitch, yaw = rot_to_rpy(T06[:3, :3])
    return np.array([x, y, z, roll, pitch, yaw])

def rot_to_rpy(R, eps=1e-6):
    sp = -R[2, 0]
    cp = np.hypot(R[0, 0], R[1, 0])
    pitch = np.arctan2(sp, cp)
    if cp > eps:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0                              # convenção do slide: α = 0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])    # confirma o sinal com o teste abaixo
    return np.degrees([roll, pitch, yaw])

