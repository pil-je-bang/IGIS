import numpy as np
from scipy.optimize import minimize
import sympy as sym
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math


# Define symbols for joint variables
q1, q2, q3, q4, q5, q6 = sym.symbols('q1 q2 q3 q4 q5 q6')

# Define DH parameters as a matrix
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

def calc_T(DH_params, i, j):
    T = sym.eye(4)
    for k in range(i, j+1):
        alpha = DH_params[k, 3]
        a = DH_params[k, 2]
        d = DH_params[k, 1]
        q = DH_params[k, 0]
        
        T_k = DH_transform(q, d, a, alpha)
        T = T * T_k
    
    return T

def DH_transform(q, d, a, alpha):
    # q: Joint variable (angle in radians)
    # d: Link offset
    # a: Link length
    # alpha: Link twist
    T = sym.Matrix([[sym.cos(q), -sym.sin(q)*sym.cos(alpha), sym.sin(q)*sym.sin(alpha), a*sym.cos(q)],
                    [sym.sin(q), sym.cos(q)*sym.cos(alpha), -sym.cos(q)*sym.sin(alpha), a*sym.sin(q)],
                    [0, sym.sin(alpha), sym.cos(alpha), d],
                    [0, 0, 0, 1]])
    return T


# Define the forward kinematics function
def forward_kinematics(q_values):
    # Replace the symbol values with the given joint angles
    DH_params_subs = DH_params.subs({q1: q_values[0], q2: q_values[1], q3: q_values[2], 
                                     q4: q_values[3], q5: q_values[4], q6: q_values[5]})
    
    # Calculate the transformation matrix from base to end effector
    T_06 = calc_T(DH_params_subs, 0, 5) * calc_T(DH_params_subs, 6, 8)
    
    # Extract the position and orientation of the end effector from the transformation matrix
    pos = np.array(T_06[0:3, 3], dtype=np.float64)
    ori = np.array(T_06[0:3, 0:3], dtype=np.float64)
    
    return pos, ori

# Define the inverse kinematics function
def inverse_kinematics(x_desired, ori_desired):
    def error_function(q_values):
        pos, ori = forward_kinematics(q_values)
        pos_error = np.linalg.norm(pos - np.array(x_desired))
        ori_error = np.linalg.norm(ori_desired - np.array(ori))

        return pos_error + ori_error
    
    # Set initial guess for joint angles
    q_initial_guess = np.array([0, 0, 0, 0, 0, 0])
    
    # Use the SLSQP algorithm to minimize the error function
    result = minimize(error_function, q_initial_guess, method='BFGS')
    
    return result.x

roll = 0
pitch= 0
yaw = 0




# Convert Euler angles to rotation matrix
r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
rot_mat = r.as_matrix()

print(rot_mat)

# Set the desired end effector position and orientation
x_desired = np.array([[1000], [100], [100]])
#ori_desired = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# Calculate the joint angles using inverse kinematics
q_values = inverse_kinematics(x_desired,rot_mat)

# Print the joint angles
print(q_values)

a=forward_kinematics(q_values)
print(a)

#visualization
q1, q2, q3, q4, q5, q6 = q_values

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



# Define the plot settings
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(0, 1000)

# Plot the arm
ax.plot([P0[0], P1[0], P2[0], P3[0], P4[0], P5[0], P6[0], P7[0], P8[0], P9[0]],
        [P0[1], P1[1], P2[1], P3[1], P4[1], P5[1], P6[1], P7[1], P8[1], P9[1]],
        [P0[2], P1[2], P2[2], P3[2], P4[2], P5[2], P6[2], P7[2], P9[2], P9[2]], 'ro-')

# Extract the rotation matrix from the homogeneous transform matrix T09
R = np.array(T09[:3, :3])

# Calculate the roll, pitch, and yaw angles using the rotation matrix
roll = math.atan2(R[2, 1], R[2, 2])
pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
yaw = math.atan2(R[1, 0], R[0, 0])
ori = np.array([roll, pitch, yaw])

x = P9
ori = np.array([roll, pitch, yaw])
ee = np.concatenate((x, ori), axis=0)
print(ee)
