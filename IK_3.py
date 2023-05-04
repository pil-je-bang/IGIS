import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import math

# DH 매개변수 설정 (theta, d, a, alpha)
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

def inverse_kinematics(x, y, z, roll, pitch, yaw):
    # Calculate the rotation matrix from roll, pitch, and yaw angles
    R = rotation_matrix(roll, pitch, yaw)

    # Calculate the position of the wrist center
   #Pw = sym.Matrix([x, y, z]) - 96.7*R[:, 2]
    Pw = sym.Matrix([x, y, z]) - np.dot(96.7*R[:, 2], np.ones(3))

    # Calculate the joint angles q1 to q6 using inverse kinematics
    Pc = Pw - 148.4*R[:, 0]
    q1 = sym.atan2(Pc[1], Pc[0])

    D = (Pc[0]**2 + Pc[1]**2 + (Pc[2]-169.2)**2 - 148.4**2 - 566.9**2) / (2*148.4*566.9)
    q3 = sym.atan2(-sym.sqrt(1 - D**2), D)
    q2 = sym.atan2(Pc[2]-169.2, sym.sqrt(Pc[0]**2 + Pc[1]**2)) - sym.atan2(566.9*sym.sin(q3), 148.4+566.9*sym.cos(q3))

    R03 = DH_transform(q1, 0, 0, DH_params[0, 3])[:3, :3] \
          * DH_transform(q2, DH_params[1, 1], 0, DH_params[1, 3])[:3, :3] \
          * DH_transform(q3, DH_params[3, 1], 0, DH_params[3, 3])[:3, :3]
    R36 = R03.T * R

    q5 = sym.atan2(sym.sqrt(1 - R36[2, 2]**2), R36[2, 2])
    q4 = sym.atan2(R36[1, 2], R36[0, 2])
    q6 = sym.atan2(R36[2, 1], -R36[2, 0])

    # Return the joint angles as a tuple
    return q1, q2, q3, q4, q5, q6

q= inverse_kinematics(146.653946418090,-146.653946418090,1369.20000000000,0,0,0)

#q1,q2,q3,q4,q5,q6=math.pi/4,0,0,0,0,0

# 각 관절 위치 계산
T01 = DH_transform(q1, 169.2, 0, -sym.pi/2)
T12 = DH_transform(q2-sym.pi/2, -148.4, 0, 0)
T23 = DH_transform(0, 0, 566.9, 0)
T34 = DH_transform(q3, 148.4, 0, 0)
T45 = DH_transform(0, 0, 522.4, 0)
T56 = DH_transform(q4+sym.pi/2, -110.7, 0, 0)
T67 = DH_transform(0, 0, 0, sym.pi/2)
T78 = DH_transform(q5, 110.7, 0, -sym.pi/2)
T89= DH_transform(q6, -96.7, 0, sym.pi/2)

# 각 관절 위치 표시를 위한 homogeneous transform matrix 계산
T02 = T01 * T12
T03 = T02 * T23
T04 = T03 * T34
T05 = T04 * T45
T06 = T05 * T56
T07 = T06 * T67
T08 = T07 * T78
T09 = T08 * T89

# homogeneous transform matrix를 이용하여 각 관절의 위치 계산
origin = np.array([[0, 0, 0, 1]]).T
P0 = np.array([0, 0, 0])
P1 = np.array(T01 * origin)[:3].flatten()
P2 = np.array(T02 * origin)[:3].flatten()
P3 = np.array(T03 * origin)[:3].flatten()
P4 = np.array(T04 * origin)[:3].flatten()
P5 = np.array(T05 * origin)[:3].flatten()
P6 = np.array(T06 * origin)[:3].flatten()
P7 = np.array(T07 * origin)[:3].flatten()
P8 = np.array(T08 * origin)[:3].flatten()
P9 = np.array(T09 * origin)[:3].flatten()


#좌표 시각화


# Define the plot settings
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(0, 1000)

# Plot the arm
ax.plot([P0[0], P1[0], P2[0], P3[0], P4[0], P5[0], P6[0], P7[0], P8[0],P9[0]],
        [P0[1], P1[1], P2[1], P3[1], P4[1], P5[1], P6[1], P7[1], P8[1],P9[1]],
        [P0[2], P1[2], P2[2], P3[2], P4[2], P5[2], P6[2], P7[2], P9[2],P9[2]], 'ro-')

result = P9

print(result)