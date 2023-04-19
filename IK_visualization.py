import numpy as np
import matplotlib.pyplot as plt
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



# Test the inverse kinematics
q = inverse_kinematics(600, 600, 600, 1, 1, 1)

# Define the joint angles
#q = np.array([0, 0, 0, 0, 0, 0])

# Define the positions of each joint
P0 = np.array([0, 0, 0])
P1 = P0 + np.array([0, 0, L1])
P2 = P1 + np.array([L2 * np.sin(q[1]) * np.cos(q[0]), L2 * np.sin(q[1]) * np.sin(q[0]), L2 * np.cos(q[1])])
P3 = P2 + np.array([L3 * np.sin(q[1] + q[2]) * np.cos(q[0]), L3 * np.sin(q[1] + q[2]) * np.sin(q[0]), L3 * np.cos(q[1] + q[2])])
P4 = P3 + np.array([L4 * np.sin(q[1] + q[2] + q[3]) * np.cos(q[0]), L4 * np.sin(q[1] + q[2] + q[3]) * np.sin(q[0]), L4 * np.cos(q[1] + q[2] + q[3])])
P5 = P4 + np.array([L5 * np.sin(q[1] + q[2] + q[3] + q[4]) * np.cos(q[0]), L5 * np.sin(q[1] + q[2] + q[3] + q[4]) * np.sin(q[0]), L5 * np.cos(q[1] + q[2] + q[3] + q[4])])
P6 = P5 + np.array([L6 * np.sin(q[1] + q[2] + q[3] + q[4] + q[5]) * np.cos(q[0]), L6 * np.sin(q[1] + q[2] + q[3] + q[4] + q[5]) * np.sin(q[0]), L6 * np.cos(q[1] + q[2] + q[3] + q[4] + q[5])])

# Define the plot settings
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(0, 1000)

# Plot the arm
ax.plot([P0[0], P1[0], P2[0], P3[0], P4[0], P5[0], P6[0]],
        [P0[1], P1[1], P2[1], P3[1], P4[1], P5[1], P6[1]],
        [P0[2], P1[2], P2[2], P3[2], P4[2], P5[2], P6[2]], 'ro-')

# Show the plot
plt.show()







