import numpy as np
from scipy.optimize import minimize
import sympy as sym
from scipy.spatial.transform import Rotation

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
    DH_params_subs = DH_params.subs({q1: q_values[0], q2: q_values[1], q3: q_values[2], q4: q_values[3], q5: q_values[4], q6: q_values[5]})
    
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
    result = minimize(error_function, q_initial_guess, method='SLSQP')
    
    return result.x


roll = 0
pitch = 0
yaw = 0

# Convert Euler angles to rotation matrix
r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
rot_mat = r.as_matrix()

print(rot_mat)

# Set the desired end effector position and orientation
x_desired = np.array([[0], [-207.4], [1369.2]])
#ori_desired = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# Calculate the joint angles using inverse kinematics
q_values = inverse_kinematics(x_desired,rot_mat)

# Print the joint angles
print(q_values)

a=forward_kinematics(q_values)
print(a)