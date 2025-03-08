import numpy as np

def MEKF_fast(dt, acc_meas, gyro_meas, mag_meas, sigma_acc, sigma_ARW, sigma_RRW, sigma_mag ,x_k_1, P_k_1):
    # dt: time interval (use gnss time)
    # acc_meas: 3 by 1, Body frame
    # gyro_meas: 3 by 1, Body frame
    # mag_meas: 3 by 1, Body frame
    # R_BODY2ENU: 3 by 3, rotation matrix from body frame to ENU frame
    # x_k_1: 14 by 1, 
        # [0:4] quaternion
        # [4:7] gyro bias
        # [7:13] delta X (delta rho, delta beta)
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

def MEKF(dt, acc_meas, gyro_meas, mag_meas, sigma_acc_ARW, sigma_acc_RRW, sigma_ARW, sigma_RRW, sigma_mag ,x_k_1, P_k_1):
    # dt: time interval (use gnss time)
    # acc_meas: 3 by 1, Body frame
    # gyro_meas: 3 by 1, Body frame
    # mag_meas: 3 by 1, Body frame
    # R_BODY2ENU: 3 by 3, rotation matrix from body frame to ENU frame
    # x_k_1: 14 by 1, state vector:
        # [0:3] orientation error
        # [3:6] velocity error
        # [6:9] position error
        # [9:12] gyro bias
        # [12:15] accelerometer bias
        # [15:18] magnetometer bias
    # P_k_1: 13 by 13, covariance matrix

    
    I3 = np.eye(3)
    I4 = np.eye(4)
    I18 = np.eye(18)

    # Propagated value from previous step
    q_k = x_k_1[0:4]
    beta_gyro_k = x_k_1[9:12]
    beta_acc_k = x_k_1[12:15]
    beta_mag_k = x_k_1[15:18]
    x_k = np.zeros(18)

    # Predition
    # Depolarize bias from gyroscope
    omega_k = gyro_meas - beta_gyro_k
    a_k = acc_meas - beta_acc_k
    m_k = mag_meas - beta_mag_k

    # Quaternion propagation
    Omega_k = np.array([[0, omega_k[2], -omega_k[1], omega_k[0]],
            [-omega_k[2], 0, omega_k[0], omega_k[1]],
            [omega_k[1], -omega_k[0], 0, omega_k[2]],
            [-omega_k[0], -omega_k[1], -omega_k[2], 0]])
    q_k = np.dot(I4 + 0.5*dt*Omega_k, q_k)
    q_k = q_k/np.linalg.norm(q_k)

    # Covariance equation propagation
    R_k = Quaternion2Rotation(q_k).transpose()
    G = np.zeros((18,18))
    G[0:3,9:12] = -I3
    G[6:9,3:6] = I3
    G[0:3,0:3] = - skew_symmetric(omega_k)
    G[3:6,0:3] = - np.dot(R_k, skew_symmetric(a_k))
    G[3:6, 12:15] = - R_k
    F = np.eye(18) + G*dt
    Q = np.zeros((18, 18))
    Q[0:3, 0:3] = (sigma_ARW**2*dt + 1/3 * sigma_RRW**2*dt**3)*I3
    Q[0:3, 9:12] = -0.5*sigma_RRW**2*dt**2*I3
    Q[3:6, 3:6] = (sigma_acc_ARW**2*dt + 1/3 * sigma_acc_RRW**2*dt**3)*I3
    Q[3:6, 6:9] = sigma_acc_RRW**2*I3*(dt**4)/8.0 + sigma_acc_ARW**2*I3*(dt**2)/2.0
    Q[3:6, 12:15] = -sigma_acc_RRW**2*I3*(dt**2)/2.0
    Q[6:9, 3:6] = sigma_acc_ARW**2*I3*(dt**2)/2.0 + sigma_acc_RRW**2*I3*(dt**4)/8.0
    Q[6:9, 6:9] = sigma_acc_ARW**2*I3*(dt**3)/3.0 + sigma_acc_RRW**2*I3*(dt**5)/20.0
    Q[6:9, 12:15] = -sigma_acc_RRW**2*I3*(dt**3)/6.0
    Q[9:12, 0:3] = -sigma_RRW**2*(dt**2)/2.0
    Q[9:12, 9:12] = sigma_RRW**2*dt
    Q[12:15, 3:6] = -sigma_acc_RRW**2*I3*(dt**2)/2.0
    Q[12:15, 6:9] = -sigma_acc_RRW**2*I3*(dt**3)/6.0
    Q[12:15, 12:15] = sigma_acc_RRW**2*I3*dt
    Q[15:18, 15:18] = sigma_mag**2*dt
    
    P_k = np.dot(F, np.dot(P_k_1, F.transpose())) + Q

    # Update
    H = np.zeros((6,18))
    H[0:3, 0:3] = skew_symmetric(np.dot(R_k, np.array([0,0,-1])))
    H[0:3, 12:15] = I3
    H[0:3, 12:15] = skew_symmetric(np.dot(R_k, np.array([1,0,0])))
    H[3:6, 15:18] = I3
    R = np.identity(6)
    R[0:3, 0:3] = sigma_acc_ARW * np.identity(3)
    R[3:6, 3:6] = sigma_mag * np.identity(3)
    # Kalman gain
    K = np.dot(np.dot(P_k, H.transpose()), np.linalg.inv(np.dot(H, np.dot(P_k, H.transpose())) + R))
    
    # Update covariance
    P_k = np.dot(np.dot((I18 - np.dot(K, H)), P_k), (I18 - np.dot(K, H)).transpose()) + np.dot(K, np.dot(R, K.transpose()))
    
    observation = np.zeros(shape=(6, ))
    observation[0:3] = a_k
    observation[3:6] = m_k
    predicted_observation = np.zeros(shape=(6, ), dtype=float)
    predicted_observation[0:3] = np.dot(R_k, np.array([0,0,-1]))
    predicted_observation[3:6] = np.dot(R_k, np.array([1,0,0]))
    x_k = np.dot(K, (observation - predicted_observation))

    # Update x_k
    drho = 0.5*x_k[0:3]
    q_4 = np.sqrt(1 - np.dot(drho, drho))
    d_q = np.array([drho[0],drho[1],drho[2], q_4])
    x_k[0:4] = QuaternionProduct(q_k,d_q)
    x_k[9:12] = beta_gyro_k + x_k[9:12]
    x_k_1[12:15] = beta_acc_k + x_k[12:15]
    x_k_1[15:18] = beta_mag_k + x_k[15:18]

    
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

def skew_symmetric(v):
    # v: 3 by 1, vector
    # v_skew: 3 by 3, skew symmetric matrix
    v_skew = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
    return v_skew