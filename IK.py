
import numpy as np
import math


# Define the robot geometry
L1 = 148.4
L2 = 566.9
L3 = 148.4
L4 = 522.4
L5 = 110.7
L6 = 110.7



# Compute the inverse kinematics
def inverse_kinematics(x, y, z, roll, pitch, yaw):
    
    # Compute the wrist position
    wx = x - L6 * np.cos(yaw) * np.cos(pitch) * np.sin(roll) + L5 * np.cos(yaw) * np.sin(pitch) - L4 * np.sin(yaw) * np.cos(pitch) * np.sin(roll) + L3 * np.sin(yaw) * np.sin(pitch) + L2 * np.cos(pitch) * np.sin(roll)
    wy = y - L6 * np.sin(yaw) * np.cos(pitch) * np.sin(roll) + L4 * np.cos(yaw) * np.cos(pitch) * np.sin(roll) + L2 * np.cos(pitch) * np.cos(roll) - L3 * np.sin(pitch) + L5 * np.sin(yaw) * np.sin(pitch)    
    wz = z + L6 * np.sin(pitch) * np.sin(yaw) + L4 * np.cos(pitch) * np.cos(yaw) + L3 * np.cos(pitch) * np.sin(yaw) + L5 * np.cos(pitch) * np.cos(yaw) * np.sin(roll)  - L2 * np.sin(pitch) * np.sin(roll)

    # Compute the joint angles
    q1 = np.arctan2(wy, wx)
    q2 = np.arctan2(np.sqrt(wx**2 + wy**2) - L1, wz)
    q3 = np.arctan2(np.sqrt(L4**2 + L5**2) - np.sqrt((wx**2 + wy**2 - L1**2 - (wz - L2)**2)), wz - L2 - L3)
    
    q4 = np.arctan2((wz - L2 - L3), np.sqrt(wx**2 + wy**2 - L1**2) - np.sqrt((wx**2 + wy**2 - L1**2 - (wz - L2 - L3)**2)))
    
    q5 = np.arctan2(np.cos(q4) * (L4 * np.sin(q3) + L5 * np.cos(q3)), L4 * np.cos(q3) - L5 * np.sin(q3))
    q6 = roll - q4 - q5
    q= np.array([q1, q2, q3, q4, q5, q6])
    q= q*180/math.pi
    
    
    q_max = np.array([360, 360, 165, 360, 360, 360 ])
    q_min = np.array([-360, -360, -165, -360, -360, -360 ])
    
    if (q_min <= q).all() and (q_max >= q).all():

        return q


# Test the inverse kinematics
q = inverse_kinematics(600, 600, 600, 1, 1, 1)






