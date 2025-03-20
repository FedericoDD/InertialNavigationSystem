import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import DataAnal.PostProcessing as pp
import DataAnal.MEKF as mekf
import DataAnal.EKF as ekf

FILE_PATH = 'PostProcess' + os.sep + 'Data' + os.sep
FILE_NAME = FILE_PATH + 'InertialTest' +'.csv'

data = pp.import_data(FILE_NAME)
#pp.plot_all(data)

sigma_acc_VRW = 0.2
sigma_acc_RRW = 1e-3
sigma_acc = 0.2
sigma_ARW = 10 * np.pi / 180 / 3600  # rad/s
sigma_RRW = 0.00001 * np.pi / 180 / np.sqrt(3600)  # rad/sqrt(s)
sigma_mag = 1
hdop = 3
vdot = 5
acc_bias = data.iloc[1,1:4].values

# MEKF
# Initial state
x_k = np.zeros(18)
x_k[0:4] = data.iloc[0, 10:14].values
p_0 = 1e-12
P_k = np.eye(18)*p_0

q_vec = np.zeros((len(data),4))
q_vec[0, :] = x_k[0:4]
x_vect = np.zeros((len(data),3))
x_vect[0, :] = np.zeros(3,)
vel = np.zeros(3,)

x_EKF_k = np.zeros((6,))
P_EKF_k = np.eye(6)*1e-3
x_EKF_k[0:3] = np.zeros(3,)
if np.isnan(data.iloc[0, 17]):  
    x_EKF_k[3:6] = np.zeros(3,)
    origin = np.zeros(3,)
else:
    x_EKF_k[3:6] = ekf.getSpeed(data.iloc[0, 17], data.iloc[0, 18])
    origin = ekf.GPS2ENU(data.iloc[0, 14], data.iloc[0, 15], data.iloc[0, 16], np.zeros(3,))

x_vect = np.zeros((len(data),3))
x_vect[0, :] = np.zeros(3,)
x_vect_EKF = np.zeros((len(data),3))
x_vect_EKF[0, :] = np.zeros(3,)
v_vect_EKF = np.zeros((len(data),3))
v_vect_EKF[0, :] = np.zeros(3,)
acc_vec = np.zeros((len(data),3))
acc_vec[0, :] = np.zeros(3,)

dt_vec = np.zeros((len(data),1))
dt_vec[0] = 0
ds_vec = np.zeros((len(data),1))
ds_vec[0] = 0
R = mekf.Quaternion2Rotation(x_k[0:4])

position_ENU = np.zeros((len(data),3))
velocities_ENU = np.zeros((len(data),3))

integrated_vel = np.zeros((len(data),3))
integrated_pos = np.zeros((len(data),3))

for i in range(1, len(data)):
    dt = data.iloc[i, 0] - data.iloc[i-1, 0]
    dt_vec[i] = dt
    acc_meas = data.iloc[i, 1:4].values 
    gyro_meas = data.iloc[i, 4:7].values
    mag_meas = data.iloc[i, 7:10].values

    lat = data.iloc[i, 14]
    lon = data.iloc[i, 15]
    alt = data.iloc[i, 16]
    speed_kmh = data.iloc[i, 17]
    track_angle = data.iloc[i, 18]

    x_k, P_k = mekf.MEKF(dt, acc_meas, gyro_meas, mag_meas, sigma_acc_VRW, sigma_acc_RRW, sigma_ARW, sigma_RRW, sigma_mag, x_k, P_k)

    if not True:
        R_NB = mekf.Quaternion2Rotation(x_k[0:4])
        acc_meas = np.dot(R_NB, data.iloc[i, 1:4].values - x_k[12:15] - acc_bias)  
    else:
        R_NB = mekf.Quaternion2Rotation(data.iloc[i, 10:14].values)
        acc_meas = np.dot(R_NB, data.iloc[i, 1:4].values - acc_bias) 
    
    acc_vec[i, :] = acc_meas 
    
    if np.isnan(lat):
        x_EKF_k[3:6] +=  acc_meas*dt
        # norm of acc_meas
        ds_vec[i] =  np.linalg.norm(acc_meas) * dt**2 + ds_vec[i-1]
        x_EKF_k[0:3] += x_EKF_k[3:6]*dt
        # get position in ENU
        position_ENU[i, :] = position_ENU[i-1, :]
        # get velocity in ENU
        velocities_ENU[i, :] = velocities_ENU[i-1, :]
    else:
        # get position in ENU

        position_ENU[i, :] = ekf.GPS2ENU(lat, lon, alt, origin)
        # get velocity in ENU
        velocities_ENU[i, :] = ekf.getSpeed(speed_kmh, track_angle)
        x_EKF_k, P_EKF_k = ekf.EKF(dt,lat ,lon,alt,origin,speed_kmh, track_angle, acc_meas, x_EKF_k, P_EKF_k, sigma_acc, hdop, vdot)

    # get integrated velocity and position
    integrated_vel[i, :] = integrated_vel[i-1, :] + acc_meas*dt
    integrated_pos[i, :] = integrated_pos[i-1, :] + integrated_vel[i-1, :]*dt

    # save x_EKF_k to data
    x_vect_EKF[i, :] = x_EKF_k[0:3]
    v_vect_EKF[i,:] = x_EKF_k[3:6]

# plot dt
plt.figure()
plt.title(' ||x|| [m]')
plt.plot(data.iloc[:, 0], ds_vec)
plt.grid()
plt.minorticks_on()

# plot acc_vec
plt.figure()
plt.title('Acceleration [m/s^2]')
plt.subplot(3, 1, 1)
plt.plot(data.iloc[:, 0], acc_vec[:, 0], label='x')
plt.grid()
plt.minorticks_on()
plt.tight_layout()
plt.subplot(3, 1, 2)
plt.plot(data.iloc[:, 0], acc_vec[:, 1], label='y')
plt.grid()
plt.minorticks_on()
plt.tight_layout()
plt.subplot(3, 1, 3)
plt.plot(data.iloc[:, 0], acc_vec[:, 2], label='z')
plt.grid()
plt.minorticks_on()
plt.tight_layout()

# plot velocities
plt.figure()
plt.title('Velocities [m/s]')
plt.subplot(3, 1, 1)
plt.plot(data.iloc[:, 0], integrated_vel[:, 0], label='N_int')
plt.plot(data.iloc[:, 0], velocities_ENU[:, 0], label='N')
plt.plot(data.iloc[:, 0], v_vect_EKF[:,0], label='N')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.subplot(3, 1, 2)
plt.plot(data.iloc[:, 0], integrated_vel[:, 1], label='E_int')
plt.plot(data.iloc[:, 0], velocities_ENU[:, 1], label='E')
plt.plot(data.iloc[:, 0], v_vect_EKF[:,1], label='N')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.subplot(3, 1, 3)
plt.plot(data.iloc[:, 0], integrated_vel[:, 2], label='D_int')
plt.plot(data.iloc[:, 0], velocities_ENU[:, 2], label='D')
plt.plot(data.iloc[:, 0], v_vect_EKF[:,2], label='N')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.legend(['Integrator','GNSS','EKF'])


# plot result
plt.figure()
plt.title('Position [m]')
plt.subplot(3, 1, 1)
plt.plot(data.iloc[:, 0], integrated_pos[:, 0], label='x')
plt.plot(data.iloc[:, 0], position_ENU[:, 0], label='x_GNSS')
plt.plot(data.iloc[:, 0], x_vect_EKF[:, 0], label='x_EKF')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.subplot(3, 1, 2)
plt.plot(data.iloc[:, 0], integrated_pos[:, 1], label='y')
plt.plot(data.iloc[:, 0], position_ENU[:, 1], label='y_GNSS')
plt.plot(data.iloc[:, 0], x_vect_EKF[:, 1], label='y_EKF')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.subplot(3, 1, 3)
plt.plot(data.iloc[:, 0], integrated_pos[:, 2], label='z')
plt.plot(data.iloc[:, 0], position_ENU[:, 2], label='z_GNSS')
plt.plot(data.iloc[:, 0], x_vect_EKF[:, 2], label='z_EKF')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.legend(['Integrator','GNSS','EKF'])


# plot result
plt.figure()
plt.title('Position [m]')
plt.subplot(3, 1, 1)
plt.plot(data.iloc[:, 0], position_ENU[:, 0], label='N')
plt.plot(data.iloc[:, 0], x_vect_EKF[:, 0], label='x_EKF')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.subplot(3, 1, 2)
plt.plot(data.iloc[:, 0], position_ENU[:, 1], label='y')
plt.plot(data.iloc[:, 0], x_vect_EKF[:, 1], label='y_EKF')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.subplot(3, 1, 3)
plt.plot(data.iloc[:, 0], position_ENU[:, 2], label='z')
plt.plot(data.iloc[:, 0], x_vect_EKF[:, 2], label='z_EKF')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.legend(['GNSS','EKF'])

# plot result
plt.figure()
plt.title('X-Y Plane [m]')  
plt.plot(position_ENU[:, 0], position_ENU[:, 1], label='GNSS')
plt.plot(x_vect_EKF[:, 0], x_vect_EKF[:, 1], label='EKF')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.axis('equal')
plt.legend(['GNSS','EKF'])
plt.show()



'''
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
'''