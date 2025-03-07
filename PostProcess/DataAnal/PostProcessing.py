import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import MEKF as mekf


# Function to import data from data.csv
def import_data(file_name):
    # Avoid to use the first column as index
    data = pd.read_csv(file_name, sep=';', index_col=False,  skiprows=[0,1],skip_blank_lines=True)    
    return data

# Function to show plot of the first three columns of the data over first column
def plot_gyro(data):
    # plot data
    plt.plot(data.iloc[:,0],data.iloc[:,4], 'r')
    plt.plot(data.iloc[:,0],data.iloc[:,5], 'g')
    plt.plot(data.iloc[:,0],data.iloc[:,6], 'b')
    # add grid
    plt.grid()
    # add legend: X, Y, Z
    plt.legend(['X', 'Y', 'Z'])
    # add title
    plt.title('Gyro Data [rad/s]')

def plot_acc(data):
    # bias = norm of the first row
    bias = np.array([0,0,np.linalg.norm(data.iloc[0,1:4])])
    acc_delta = np.zeros((len(data),3))
    # caluclate norm of the data
    for i in range(len(data)):
        R = mekf.Quaternion2Rotation(data.iloc[i,10:14].values)
        acc_delta[i,:] = data.iloc[i,1:4].values - np.dot(R.transpose(), bias)

    # plot data
    plt.plot(data.iloc[:,0],acc_delta[:,0], 'r')
    plt.plot(data.iloc[:,0],acc_delta[:,1], 'g')
    plt.plot(data.iloc[:,0],acc_delta[:,2], 'b')
    # add grid
    plt.grid()
    # add legend: X, Y, Z
    plt.legend(['X', 'Y', 'Z'])
    # add title
    plt.title('Acc Data [m/s^2]')
    

# plot acc norm
def plot_acc_norm(data):
    
    # plot norm of the data over time
    plt.plot(data.iloc[:,0],np.sqrt(np.square(data.iloc[:,1:4]).sum(axis=1)), 'k')
    # plot constant line at 9.805
    plt.plot([data.iloc[0,0], data.iloc[len(data)-1,0]], [9.805, 9.805], 'r--')
    # add grid
    plt.grid()
    # add legend: X, Y, Z
    plt.legend(['||Acc||', '9.805'])
    # add title
    plt.title('||Acc Data|| [m/s^2]')
    
def plot_all(data):
    plt.figure()
    plt.subplot(311)
    plot_gyro(data)
    plt.subplot(312)
    plot_acc(data)
    plt.subplot(313)
    plot_acc_norm(data)
    plt.show()

if __name__ == '__main__':
    import os
    FILE_PATH = 'PostProcess' + os.sep + 'Data' + os.sep
    FILE_NAME = FILE_PATH + 'data_20250307.csv'
    data = import_data(FILE_NAME)
    plt.figure()
    plot_acc(data)
    plt.show()
