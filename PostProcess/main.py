import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import DataAnal.PostProcessing as pp
import DataAnal.MEKF as mekf

FILE_PATH = 'PostProcess' + os.sep + 'Data' + os.sep
FILE_NAME = FILE_PATH + 'data_20250307.csv'

data = pp.import_data(FILE_NAME)
#pp.plot_all(data)

# MEKF
# Initial state
x_k = np.zeros(18)
x_k[0:4] = data.iloc[0, 10:14].values
P_k = np.eye(18)*1e-12

sigma_acc_ARW = 1
sigma_acc_RRW = 0.5
sigma_ARW = 0.66 * np.pi / 180 / 3600  # rad/sqrt(s)
sigma_RRW = 0.4 * np.pi / 180 / 3600  # rad/s
sigma_mag = 1.1e3 * 3

q_vec = np.zeros((len(data),4))
q_vec[0, :] = x_k[0:4]
x_vect = np.zeros((len(data),3))
x_vect[0, :] = np.zeros(3,)
vel = np.zeros(3,)
acc_bias = np.array([0,0,np.linalg.norm(data.iloc[1,1:4])])
for i in range(1, len(data)):
    R = mekf.Quaternion2Rotation(x_k[0:4])
    dt = data.iloc[i, 0] - data.iloc[i-1, 0]
    acc_meas = data.iloc[i, 1:4].values 
    gyro_meas = data.iloc[i, 4:7].values
    mag_meas = data.iloc[i, 7:10].values
    
    x_k = x_k
    P_k = P_k

    x_k, P_k = mekf.MEKF(dt, acc_meas, gyro_meas, mag_meas, sigma_acc_ARW, sigma_acc_RRW, sigma_ARW, sigma_RRW, sigma_mag, x_k, P_k)
    
    # save x_k to data
    q_vec[i, :] = x_k[0:4]
    acc_clean = acc_meas - np.dot(R.transpose(), acc_bias)
    x_vect[i, :] =  x_vect[i-1, :] + acc_clean*dt**2

# MEKF
# Initial state
x_k = np.zeros(13)
x_k[0:4] = data.iloc[0, 10:14].values
P_k = np.eye(6)*1e-12

sigma_acc = sigma_acc_ARW

q_vec_fast = np.zeros((len(data),4))
q_vec_fast[0, :] = x_k[0:4]
for i in range(1, len(data)):
    R = mekf.Quaternion2Rotation(x_k[0:4])
    dt = data.iloc[i, 0] - data.iloc[i-1, 0]
    acc_meas = data.iloc[i, 1:4].values
    gyro_meas = data.iloc[i, 4:7].values
    mag_meas = data.iloc[i, 7:10].values
    
    x_k = x_k
    P_k = P_k

    x_k, P_k = mekf.MEKF_fast(dt, acc_meas, gyro_meas, mag_meas, sigma_acc, sigma_ARW, sigma_RRW, sigma_mag, x_k, P_k)
    
    # save x_k to data
    q_vec_fast[i, :] = x_k[0:4]


# plot result
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(data.iloc[:, 0], data.iloc[:, 10], label='v1')
plt.plot(data.iloc[:, 0], q_vec[:, 0], label='qm1')
plt.plot(data.iloc[:, 0], q_vec_fast[:, 0], label='qm1')
plt.subplot(4, 1, 2)
plt.plot(data.iloc[:, 0], data.iloc[:, 11], label='v2')
plt.plot(data.iloc[:, 0], q_vec[:, 1], label='qm2')
plt.plot(data.iloc[:, 0], q_vec_fast[:, 1], label='qm2')
plt.subplot(4, 1, 3)
plt.plot(data.iloc[:, 0], data.iloc[:, 12], label='v3')
plt.plot(data.iloc[:, 0], q_vec[:, 2], label='qm3')
plt.plot(data.iloc[:, 0], q_vec_fast[:, 2], label='qm3')
plt.subplot(4, 1, 4)
plt.plot(data.iloc[:, 0], data.iloc[:, 13], label='4')
plt.plot(data.iloc[:, 0], q_vec[:, 3], label='qm4')
plt.plot(data.iloc[:, 0], q_vec_fast[:, 3], label='qm4')
plt.legend(['q_IMU', 'q_MEKF', 'q_MEKF_fast'])
plt.show()

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(data.iloc[:, 0], x_vect[:, 0], label='x')
plt.subplot(3, 1, 2)
plt.plot(data.iloc[:, 0], x_vect[:, 1], label='y')
plt.subplot(3, 1, 3)
plt.plot(data.iloc[:, 0], x_vect[:, 2], label='z')
plt.show()

# 3D plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vect[20:, 0], x_vect[20:, 1], x_vect[20:, 2])
plt.show()