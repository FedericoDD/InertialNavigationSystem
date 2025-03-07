import numpy as np

def MEKF(dt, acc_meas, gyro_meas, mag_meas, sigma_acc, sigma_ARW, sigma_RRW, sigma_mag ,x_k_1, P_k_1):
    # dt: time interval (use gnss time)
    # acc_meas: 3 by 1, Body frame
    # gyro_meas: 3 by 1, Body frame
    # mag_meas: 3 by 1, Body frame
    # R_BODY2ENU: 3 by 3, rotation matrix from body frame to ENU frame
    # x_k_1: 14 by 1, 
        # [0:4] quaternion
        # [4:7] gyro bias
        # [7:13] delta X
    # P_k_1: 13 by 13, covariance matrix

    
    I3 = np.eye(3)
    I4 = np.eye(4)
    I6 = np.eye(6)
    O3 = np.zeros((3,3))

    # Propagated value from previous step
    q_k = x_k_1[0:4]
    beta_k = x_k_1[4:7]
    delta_x_k = x_k_1[7:13]
    x_k = np.zeros(13)

    # Predition
    # Depolarize bias from gyroscope
    omega_k = gyro_meas - beta_k

    # Quaternion propagation
    Omega_k = np.array([[0, omega_k[2], -omega_k[1], omega_k[0]],
            [-omega_k[2], 0, omega_k[0], omega_k[1]],
            [omega_k[1], -omega_k[0], 0, omega_k[2]],
            [-omega_k[0], -omega_k[1], -omega_k[2], 0]])
    q_k = np.dot(I4 + 0.5*dt*Omega_k, q_k)
    q_k = q_k/np.linalg.norm(q_k)

    # Covariance equation propagation
    F = np.array([[1,0,0,-dt,0,0],
                  [0,1,0,0,-dt,0],
                  [0,0,1,0,0,-dt],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]])
    G = np.array([[-1,0,0,0,0,0],
                    [0,-1,0,0,0,0],
                    [0,0,-1,0,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]])
    
    q_Q1 = sigma_ARW**2*dt + 1/3 * sigma_RRW**2*dt**3
    q_Q2 = -0.5*sigma_RRW**2*dt**2
    q_Q3 = sigma_RRW**2*dt
    Q = np.array([[q_Q1,0,0,q_Q2,0,0],
                  [0,q_Q1,0,0,q_Q2,0],
                  [0,0,q_Q1,0,0,q_Q2],
                  [q_Q2,0,0,q_Q3,0,0],
                  [0,q_Q2,0,0,q_Q3,0],
                  [0,0,q_Q2,0,0,q_Q3]])
    P_k = np.dot(F, np.dot(P_k_1, F.transpose())) + np.dot(G, np.dot(Q, G.transpose()))

    # Update
    A_k = Quaternion2Rotation(q_k)

    # Measurement model
    for i in range(3):
        
        if i == 1:
            r_acc = np.array([0,0,-1])

            # Normalized accelerometer measurement
            b = acc_meas/np.linalg.norm(acc_meas)
            Aq_r = np.dot(A_k, r_acc)
            R = sigma_acc**2 * I3
        else:
            b = mag_meas/np.linalg.norm(mag_meas)
            h = np.dot(A_k.transpose(), mag_meas)
            r_mag = np.array([np.sqrt(h[0]**2 + h[1]**2), 0, h[2]])
            Aq_r = np.dot(A_k, r_mag)
            R = sigma_mag**2 * I3

        # Sensitivity matrix
        H = np.array([[0, -Aq_r[2], Aq_r[1], 0, 0, 0],
            [Aq_r[2], 0, -Aq_r[0], 0, 0, 0],
            [-Aq_r[1], Aq_r[0], 0, 0, 0, 0]])

        # Kalman gain
        K = np.dot(np.dot(P_k, H.transpose()), np.linalg.inv(np.dot(H, np.dot(P_k, H.transpose())) + R))

        # Update covariance
        P_k = np.dot(np.dot((I6 - np.dot(K, H)), P_k), (I6 - np.dot(K, H)).transpose()) + np.dot(K, np.dot(R, K.transpose()))

        # Update state
        e_k = b - Aq_r
        y_k = e_k - np.dot(H, delta_x_k)
        delta_x_k = delta_x_k + np.dot(K, y_k)

        # Update quaternion
        drho = 0.5*delta_x_k[0:3]
        q_4 = np.sqrt(1 - np.dot(drho, drho))
        d_q = np.array([drho[0],drho[1],drho[2], q_4])
        x_k[0:4] = QuaternionProduct(q_k,d_q)

        # Update gyro bias
        delta_beta = delta_x_k[3:6]
        x_k[4:7] = beta_k + delta_beta

        # Update delta X
        x_k[7:13] = delta_x_k
    
    return x_k, P_k

def Quaternion2Rotation(q):
    # q: 4 by 1, quaternion
    # R: 3 by 3, rotation matrix from body to ENU
    q1, q2, q3, q4 = q
    q12=q1**2
    q22=q2**2
    q32=q3**2
    q42=q4**2
    '''
    R = np.array([[q12 - q22 - q32 + q42, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)],
                [2*(q1*q2 - q3*q4), -q12 + q22 - q32 + q42, 2*(q2*q3 + q1*q4)],
                [2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), -q12 - q22 + q32 + q42]])
    R = np.transpose(R)
    '''
    R = np.array([[q12 - q22 - q32 + q42, 2*(q1*q2 - q3*q4), 2*(q1*q3 + q2*q4)],
                [2*(q1*q2 + q3*q4), -q12 + q22 - q32 + q42, 2*(q2*q3 - q1*q4)],
                [2*(q1*q3 - q2*q4), 2*(q2*q3 + q1*q4), -q12 - q22 + q32 + q42]])

    return R

def QuaternionProduct(q1_v, q2_v):
    # q1, q2: 4 by 1, quaternion
    # q: 4 by 1, quaternion
    q1, q2, q3, q4 = q2_v
    Q = np.array([[q4, q3, -q2, q1],
                  [-q3, q4, q1, q2],
                [q2, -q1, q4, q3],
                [-q1, -q2, -q3, q4]])
    q = np.dot(Q, q1_v)
    q = q/np.linalg.norm(q)
    return q

'''
def QuaternionProduct(q1_v, q2_v):
    # q1, q2: 4 by 1, quaternion
    # q: 4 by 1, quaternion
    qv1 = q2_v[0:3]
    qv2 = q2_v[0:3]
    qs1 = q1_v[3]
    qs2 = q2_v[3]
    q=np.zeros((4,))
    q[0:3] = np.array([qv1*qs2 + qs1*qv2 - np.cross(qv1, qv2)])
    print(q[0:3])
    q[3] = qs1*qs2 - np.dot(qv1, qv2)
    
    q = q/np.linalg.norm(q)
    return q
'''