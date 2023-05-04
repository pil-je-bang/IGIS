import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import math
q1, q2, q3, q4, q5, q6 = sym.symbols('q1 q2 q3 q4 q5 q6')
DH_params = sym.Matrix([
    [q1, 169.2, 0, -sym.pi/2],
    [q2-sym.pi/2, -148.4, 0, 0],
    [0, 0, 566.9, 0],
    [q3, 148.4, 0, 0],
    [0, 0, 522.4, 0],
    [q4+sym.pi/2, -110.7, 0, 0],
    [0, 0, 0, sym.pi/2],
    [q5, 110.7, 0, -sym.pi/2],
    [q6, -96.7, 0, sym.pi/2]
])

# 변환 행렬 계산
def DH_transform(q, d, a, alpha):
    transform = sym.Matrix([
        [sym.cos(q), -sym.sin(q)*sym.cos(alpha), sym.sin(q)*sym.sin(alpha), a*sym.cos(q)],
        [sym.sin(q), sym.cos(q)*sym.cos(alpha), -sym.cos(q)*sym.sin(alpha), a*sym.sin(q)],
        [0, sym.sin(alpha), sym.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return transform


def solve_ik(p, theta):
    # 각각의 심볼 변수 정의
    q1, q2, q3, q4, q5, q6 = sym.symbols('q1:7')
    
    # 각 관절의 DH 파라미터
    d1 = 169.2
    d2 = 148.4
    d3 = 148.4
    d4 = 110.7
    d5 = 110.7
    d6 = 96.7
    a1 = 566.9
    a2 = 522.4
    # 로봇의 기준 프레임
    T_base = sym.eye(4)

    # 로봇의 각 관절에서의 변환 행렬
    T1_0 = DH_transform(q1, d1, a1, -sym.pi/2)
    T2_1 = DH_transform(q2 - sym.pi/2, d2, 0, 0)
    T3_2 = DH_transform(q3, d3, a2, 0)
    T4_3 = DH_transform(q4, d4, 0, 0)
    T5_4 = DH_transform(q5, d5, 0, -sym.pi/2)
    T6_5 = DH_transform(q6, d6, 0, sym.pi/2)

    # 로봇의 각 관절에서의 좌표계
    T0_1 = T_base * T1_0
    T0_2 = T0_1 * T2_1
    T0_3 = T0_2 * T3_2
    T0_4 = T0_3 * T4_3
    T0_5 = T0_4 * T5_4
    T0_6 = T0_5 * T6_5
    
    T6_EE = DH_transform(0, 0, 0, 0)

# 로봇의 end-effector에서의 좌표계
    T0_EE = T0_6 * T6_EE

# end-effector의 위치와 방향 (x, y, z, R, P, Y)
    x, y, z = p
    R, P, Y = theta

# 방향 산출을 위한 회전 행렬 생성
    R_yaw = sym.Matrix([
    [sym.cos(Y), -sym.sin(Y), 0],
    [sym.sin(Y), sym.cos(Y), 0],
    [0, 0, 1]
    ])
    R_pitch = sym.Matrix([
    [sym.cos(P), 0, sym.sin(P)],
    [0, 1, 0],
    [-sym.sin(P), 0, sym.cos(P)]
    ])
    R_roll = sym.Matrix([
    [1, 0, 0],
    [0, sym.cos(R), -sym.sin(R)],
    [0, sym.sin(R), sym.cos(R)]
    ])
    R_EE = R_yaw * R_pitch * R_roll

# 각 Euler angle에 대한 미분 계산
    dR_yaw = sym.diff(R_yaw, Y)
    dR_pitch = sym.diff(R_pitch, P)
    dR_roll = sym.diff(R_roll, R)

# Jacobian 행렬 계산을 위한 미분 계산
    J_v = T0_EE[:3, 3].jacobian([q1, q2, q3, q4, q5, q6])
    J_R = sym.Matrix([dR_roll * R_pitch * R_yaw * T0_EE[:3, :3][:, i] for i in range(3)] + 
                 [R_roll * dR_pitch * R_yaw * T0_EE[:3, :3][:, i] for i in range(3)] + 
                 [R_roll * R_pitch * dR_yaw * T0_EE[:3, :3][:, i] for i in range(3)])
    J = sym.simplify(sym.Matrix.vstack(J_v, J_R))

# 위치 오차 계산
    pos_error = sym.Matrix([x, y, z]) - T0_EE[:3, 3]

# 방향 오차 계산
    R_error = 0.5 * (R_EE * T0_EE[:3, :3].T - T0_EE[:3, :3].T * R_EE.T).vec()

# Jacobian의 유사 역행렬을 사용하여 역기구학 해 계산
    dq = J.pinv() * sym.Matrix.vstack(pos_error, R_error)

# 역기구학 해 반환
    return dq

p = [0, -207.4, 1392.4]
p = np.array(p)
theta = [0,0,0]
theta = np.array(theta)

# solve_ik() 함수 호출
solve_ik(p, theta)