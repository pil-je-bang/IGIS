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


def rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def forward_kinematics(q1, q2, q3, q4, q5, q6):
    T01 = DH_transform(q1, 169.2, 0, -sym.pi/2)
    T12 = DH_transform(q2-sym.pi/2, -148.4, 0, 0)
    T23 = DH_transform(0, 0, 566.9, 0)
    T34 = DH_transform(q3, 148.4, 0, 0)
    T45 = DH_transform(0, 0, 522.4, 0)
    T56 = DH_transform(q4+sym.pi/2, -110.7, 0, 0)
    T67 = DH_transform(0, 0, 0, sym.pi/2)
    T78 = DH_transform(q5, 110.7, 0, -sym.pi/2)
    T89= DH_transform(q6, -96.7, 0, sym.pi/2)

    T09 = T01 * T12 * T23 * T34 * T45 * T56 * T67 * T78 * T89

    return T09

def inverse_kinematics(x, y, z, roll, pitch, yaw):
    # 로봇의 현재 자세를 나타내는 회전 행렬을 계산합니다.
    R = rotation_matrix(roll, pitch, yaw)
    # 3x3 회전 행렬 R을 4x4 변환 행렬로 변환합니다.
    R_4x4 = sym.eye(4)
    R_4x4[0:3, 0:3] = R
    # 로봇의 현재 위치를 나타내는 변환 행렬을 계산합니다.
    T = sym.eye(4)
    T[0:3, 3] = sym.Matrix([x, y, z])
    # 로봇의 마지막 조인트 위치와 자세를 계산합니다.
    q = sym.Matrix([0, 0, 0, 0, 0, 0])
    
    T09 = T * R_4x4 * forward_kinematics(q1, q2, q3, q4, q5, q6)
    


    # 끝 단의 위치를 추출합니다.
    px = T09[0, 3]
    py = T09[1, 3]
    pz = T09[2, 3]

    # 끝 단의 자세를 추출합니다.
    nx = T09[0, 0]
    ny = T09[1, 0]
    nz = T09[2, 0]
    ox = T09[0, 1]
    oy = T09[1, 1]
    oz = T09[2, 1]
    ax = T09[0, 2]
    ay = T09[1, 2]
    az = T09[2, 2]

    # Newton-Raphson 방법을 사용하여 역기구학을 계산합니다.
    q = sym.Matrix([q1, q2, q3, q4, q5, q6])

    for i in range(10):
    # Jacobi 행렬을 계산합니다.
        J = sym.Matrix([[sym.diff(px, q1), sym.diff(px, q2), sym.diff(px, q3), sym.diff(px, q4), sym.diff(px, q5), sym.diff(px, q6)],
                    [sym.diff(py, q1), sym.diff(py, q2), sym.diff(py, q3), sym.diff(py, q4), sym.diff(py, q5), sym.diff(py, q6)],
                    [sym.diff(pz, q1), sym.diff(pz, q2), sym.diff(pz, q3), sym.diff(pz, q4), sym.diff(pz, q5), sym.diff(pz, q6)],
                    [sym.diff(nx, q1), sym.diff(nx, q2), sym.diff(nx, q3), sym.diff(nx, q4), sym.diff(nx, q5), sym.diff(nx, q6)],
                    [sym.diff(ny, q1), sym.diff(ny, q2), sym.diff(ny, q3), sym.diff(ny, q4), sym.diff(ny, q5), sym.diff(ny, q6)],
                    [sym.diff(nz, q1), sym.diff(nz, q2), sym.diff(nz, q3), sym.diff(nz, q4), sym.diff(nz, q5), sym.diff(nz, q6)],
                    [sym.diff(ox, q1), sym.diff(ox, q2), sym.diff(ox, q3), sym.diff(ox, q4), sym.diff(ox, q5), sym.diff(ox, q6)],
                    [sym.diff(oy, q1), sym.diff(oy, q2), sym.diff(oy, q3), sym.diff(oy, q4), sym.diff(oy, q5), sym.diff(oy, q6)],
                    [sym.diff(oz, q1), sym.diff(oz, q2), sym.diff(oz, q3), sym.diff(oz, q4), sym.diff(oz, q5), sym.diff(oz, q6)]
                    [sym.diff(ax, q1), sym.diff(ax, q2), sym.diff(ax, q3), sym.diff(ax, q4), sym.diff(ax, q5), sym.diff(ax, q6)],
                    [sym.diff(ay, q1), sym.diff(ay, q2), sym.diff(ay, q3), sym.diff(ay, q4), sym.diff(ay, q5), sym.diff(ay, q6)],
                    [sym.diff(az, q1), sym.diff(az, q2), sym.diff(az, q3), sym.diff(az, q4), sym.diff(az, q5), sym.diff(az, q6)]])

    # 끝 단의 위치 오차 벡터를 계산합니다.
    dx = sym.Matrix([x - px, y - py, z - pz, 0, 0, 0])

    # Newton-Raphson 단계에서의 역기구학을 계산합니다.
    dq = J.pinv() * dx
    q += dq

    # 역기구학 결과를 반환합니다.
    return [float(qi) for qi in q]

q1,q2,q3,q4,q5,q6=0,0,0,0,0,0
forward_kinematics(q1, q2, q3, q4, q5, q6)

x, y, z, roll, pitch, yaw = 0,-207.4,1369.2,0,0,0
inverse_kinematics(roll, pitch, yaw, x, y, z)