import time
import board
import busio
from ulab import numpy as np
## --------------------------------------
##  I M U   C O N F I G U R A T I O N
## --------------------------------------

from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
bno = BNO08X_I2C(i2c)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_MAGNETOMETER)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

## --------------------------------------
##  G N S S   C O N F I G U R A T I O N
## --------------------------------------
import adafruit_gps

# Create a serial connection for the GPS connection using default speed and
# a slightly higher timeout (GPS modules typically update once a second).
# These are the defaults you should use for the GPS FeatherWing.
# For other boards set RX = GPS module TX, and TX = GPS module RX pins.
uart = busio.UART(board.TX, board.RX, baudrate=9600, timeout=10)

# for a computer, use the pyserial library for uart access
# import serial
# uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=10)

# If using I2C, we'll create an I2C interface to talk to using default pins
# i2c = board.I2C()  # uses board.SCL and board.SDA
# i2c = board.STEMMA_I2C()  # For using the built-in STEMMA QT connector on a microcontroller

# Create a GPS module instance.
gps = adafruit_gps.GPS(uart, debug=False)  # Use UART/pyserial
# gps = adafruit_gps.GPS_GtopI2C(i2c, debug=False)  # Use I2C interface

# Initialize the GPS module by changing what data it sends and at what rate.
# These are NMEA extensions for PMTK_314_SET_NMEA_OUTPUT and
# PMTK_220_SET_NMEA_UPDATERATE but you can send anything from here to adjust
# the GPS module behavior:
#   https://cdn-shop.adafruit.com/datasheets/PMTK_A11.pdf

# Turn on the basic GGA and RMC info (what you typically want)
gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
# Turn on the basic GGA and RMC info + VTG for speed in km/h
# gps.send_command(b"PMTK314,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
# Turn on just minimum info (RMC only, location):
# gps.send_command(b'PMTK314,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
# Turn off everything:
# gps.send_command(b'PMTK314,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
# Turn on everything (not all of it is parsed!)
# gps.send_command(b'PMTK314,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0')

# Set update rate to once a second (1hz) which is what you typically want.
gps.send_command(b"PMTK220,1000")
# Or decrease to once every two seconds by doubling the millisecond value.
# Be sure to also increase your UART timeout above!
# gps.send_command(b'PMTK220,2000')
# You can also speed up the rate, but don't go too fast or else you can lose
# data during parsing.  This would be twice a second (2hz, 500ms delay):
# gps.send_command(b'PMTK220,500')

# Main loop runs forever printing the location, etc. every second.
last_print = time.monotonic()

## --------------------------------------
##  G N S S   CHANGE REFERENCE FRAME
## --------------------------------------

#def GPS2ECEF(latitude, longitude, altitude, flag_R_ECEF2ENU, ellipsoid):
def GPS2ECEF(*argv):
    # latitude in decimal degree, [-90, +90]
    # longitude in decimal degree, [-180, +180]
    # geodetic altitude in meter, obtained by GPS

    # flag_R_ECEF2ENU = 0 or 1, default is 0
    # if flag = 1, this point is the origin of ENU, calculate the rotation matrix from the ECEF frame to the local ENU frame
    # if flag = 0, output a 3 by 3 Identity matrix, won't use it

    # ellipsoid = (a, rf), a is semi-major axis [meter], rf is reciprocal flattening (1/f)
    # default: WGS84 = 6378137, 298.257223563
    
    latitude_radian = argv[0]*np.pi/180   # degree to radian
    longitude_radian = argv[1]*np.pi/180  # degree to radian
    altitude = argv[2]                   # meter

    # calculate some values in advanced
    sin_lat = np.sin(latitude_radian)
    cos_lat = np.cos(latitude_radian)
    sin_lon = np.sin(longitude_radian)
    cos_lon = np.cos(longitude_radian)

    if len(argv) > 3:
        '''
        R_ECEF2NED = np.array([\
            [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], \
            [-sin_lon, cos_lon, 0.0], \
            [-cos_lat*cos_lon, -cos_lat*sin_lon, -sin_lat]])
        '''
        R_ECEF2ENU = np.array([\
            [-sin_lon, cos_lon, 0.0], \
            [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], \
            [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]])
        
    else:
        R_ECEF2ENU = np.eye(3)

    if len(argv) > 4:
        a, rf = argv[4]
    else:
        a, rf = 6378137, 298.257223563
    
    e2 = 1 - (1 - 1 / rf) ** 2           # squared eccentricity
    n = a / np.sqrt(1 - e2 * sin_lat ** 2)  # prime vertical radius
    r = (n + altitude) * cos_lat         # perpendicular distance in z axis
    x = r * cos_lon
    y = r * sin_lon
    z = (n * (1 - e2) + altitude) * sin_lat
    # ECEF coordinates for GPS
    # 3 by 1
    point_ECEF = np.array([x, y, z])
    return point_ECEF, R_ECEF2ENU


def ECEF2ENU(origin_ECEF, point_ECEF, R_ECEF2ENU):
    # ENU coordinates, the origin is defined by yourself
    # 3 by 1, North, East, Down
    position_ENU = np.dot(R_ECEF2ENU, (point_ECEF - origin_ECEF))
    return position_ENU


def GPS2ENU(latitude, longitude, altitude, origin_ECEF):
    # latitude in decimal degree, [-90, +90]
    # longitude in decimal degree, [-180, +180]
    # altitude in meter, obtained by GPS
    # origin_ECEF is the origin of ENU in ECEF frame

    point_ECEF, R_ECEF2ENU = GPS2ECEF(latitude, longitude, altitude)
    position_ENU = ECEF2ENU(origin_ECEF, point_ECEF, R_ECEF2ENU)
    # 3 by 1, North, East, Down
    return position_ENU

## --------------------------------------
##  M E K F   A L G O R I T H M
## --------------------------------------
def skew_symmetric(v):
    """ Returns the skew-symmetric matrix of a vector """
    return np.array([[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])

def Attitude_MEKF(dt, acc_meas, gyro_meas, mag_meas, sigma_acc, sigma_ARW, sigma_RRW, sigma_mag ,x_k_1, P_k_1):
    # dt: time interval (use gnss time)
    # acc_meas: 3 by 1, Body frame
    # gyro_meas: 3 by 1, Body frame
    # mag_meas: 3 by 1, Body frame
    # R_BODY2ENU: 3 by 3, rotation matrix from body frame to ENU frame
    # x_k_1: 18 by 1, 
        # [0:4] quaternion
        # [4:7] gyro bias
        # [7:12] delta X
    # P_k_1: 18 by 18, covariance matrix

    I3 = np.eye(3)
    I4 = np.eye(4)
    I6 = np.eye(6)
    O3 = np.zeros((3,3))

    # Propagated value from previous step
    q_k = x_k_1[0:4]
    beta_k = x_k_1[4:7]
    delta_x_k = x_k_1[7:12]
    x_k = np.zeros(12)

    # Predition
    # Depolarize bias from gyroscope
    omega_k = gyro_meas - beta_k

    # Quaternion propagation
    omegax_k = skew_symmetric(omega_k)
    Omega_k = np.block([[- omegax_k, omega_k], [-omega_k.transpose(), 0]])
    q_k = np.dot(I4 + 0.5*dt*Omega_k, q_k)
    q_k = q_k/np.linalg.norm(q_k)

    # Covariance equation propagation
    F = np.block([[I3, -I3*dt], [O3, I3]])
    G = np.block([[-I3, O3], [O3, I3]])
    Q = np.array([[(sigma_ARW**2*dt + 1/3 * sigma_RRW**2*dt**3)*I3, -(0.5*sigma_RRW**2*dt**2)*I3],
                  [-0.5*sigma_RRW**2*dt**2*I3, sigma_RRW**2*dt*I3]])
    P_k = np.dot(np.dot(F, P_k_1), F.transpose()) + np.dot(np.dot(G, Q), G.transpose())

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
        Aq_rx = skew_symmetric(Aq_r)
        H = np.block([Aq_rx, O3])

        # Kalman gain
        K = np.dot(np.dot(P_k, H.transpose()), np.linalg.inv(np.dot(np.dot(H, P_k), H.transpose()) + R))

        # Update covariance
        P_k = np.dot(np.dot((I6 - np.dot(K, H)), P_k), (I6 - np.dot(K, H)).transpose()) + np.dot(np.dot(K, R), K.transpose())

        # Update state
        e_k = b - Aq_r
        y_k = e_k - np.dot(H, delta_x_k)
        delta_x_k = delta_x_k + np.dot(K, y_k)

        # Update quaternion
        drho = 0.5*delta_x_k[0:3]
        q_4 = np.sqrt(1 - np.dot(drho, drho))
        d_q = np.concatenate((drho, q_4))
        x_k[0:4] = QuaternionProduct(d_q)

        # Update gyro bias
        delta_beta = delta_x_k[3:6]
        x_k[4:7] = beta_k + delta_beta

        # Update delta X
        x_k[7:12] = delta_x_k
    
    return x_k, P_k

def QuaternionProduct(q1_v, q2_v):
    # q1, q2: 4 by 1, quaternion
    # q: 4 by 1, quaternion
    q1, q2, q3, q4 = q2_v
    Q = np.array([[q4, -q3, -q2, q1],
                  [-q3, q4, q1, q2],
                [q2, -q1, q4, q3],
                [-q1, -q2, -q3, q4]])
    q = np.dot(Q, q1_v)
    q = q/np.linalg.norm(q)
    return q

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

## --------------------------------------
##  E K F   A L G O R I T H M
## --------------------------------------
def Position_EKF(dt, position, velocity,acc_meas, R_BODY2ENU, x_k_1, P_k_1, sigma_acc, hdop, vdot):
    # dt: time interval (use gnss time)
    # position: 3 by 1, East, North, Up from GPS
    # velocity: 3 by 1, East, North, Up from GPS
    # acc_meas: 3 by 1, Body frame with gravity
    # R_BODY2ENU: 3 by 3, rotation matrix from body frame to ENU frame
    # x_k: 6 by 1, state vector
    # P_k: 6 by 6, covariance matrix
    # sigma_acc: 1 by 1, standard deviation of accelerometer
    # hdop: 1 by 1, horizontal dilution of precision
    # vdot: 1 by 1, vertical dilution of precision


    # Q: 6 by 6, process noise covariance matrix diag(sigma_acc)^2
    # R: 6 by 6, measurement noise covariance matrix diag(hdop^2, hdop^2, vdot^2,2*hdop^2/dt, 2*hdop^2/dt^2, 2*vdot^2/dt^2)


    acc_meas = np.dot(R_BODY2ENU, acc_meas) + np.array([0, 0, 9.805])
    measurements = np.concatenate((position, velocity), axis=0)
    
    x_k = np.zeros(6)
    F = np.array([[1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]])
    dt2_2 = 0.5*dt*dt
    B = np.array([[dt2_2, 0, 0],
                [0, dt2_2, 0],
                [0, 0, dt2_2],
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt]])
    
    Q = np.diag([sigma_acc**2, sigma_acc**2, sigma_acc**2]) 
    R = np.diag([hdop**2, hdop**2, vdot**2, 0.1**2, 0.1**2, 0.1**2]) # 0.1 is the standard deviation of the velocity
    V = np.dot(np.dot(B, Q), np.transpose(B))

    ## Prediction
    x_k = np.dot(F, x_k_1) + np.dot(B, acc_meas)
    P_k = np.dot(np.dot(F, P_k_1), np.transpose(F)) + V

    ## Update
    H = np.eye(6)
    K = np.dot(np.dot(P_k, np.transpose(H)), np.linalg.inv(np.dot(np.dot(H, P_k), np.transpose(H)) + R))
    x_k = x_k + np.dot(K, (measurements - np.dot(H, x_k)))
    P_k = np.dot((np.eye(6) - np.dot(K, H)), P_k)

    return x_k, P_k

R_BODY2ENU = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
origin_ECEF = np.array([[0], [0], [0]])
gps_time_old = gps.timestamp_utc

latitude = gps.latitude
longitude = gps.longitude
altitude = gps.altitude_m + gps.height_geoid # height above ellipsoid = orthometric height + geoid separation
velocity = gps.speed_kmh / 3.6 # km/h to m/s
position_ENU = GPS2ENU(latitude, longitude, altitude, origin_ECEF)
sigma_acc = 0.3
sigma_ARW = 2 * np.pi/180
sigma_RRW = 3.1 * np.pi/180
sigma_mag = 1.4e-6

# EKF initialization
x_k = np.concatenate((position_ENU, velocity), axis=0)
P_k= 1e-9*np.eye(6)

# MEKF initialization
Pq_k = 1e-9*np.eye(6)
q_k = np.array([0, 0, 0, 1])

f = open('data.txt', 'w')
f.write('Time;Latitude;Longitude;Altitude;')
f.write('Velocity_X;Velocity_Y;Velocity_Z;')
f.write('HDilution;VDilution;')
f.write('Accel_X;Accel_Y;Accel_Z;')
f.write('Gyro_X;Gyro_Y;Gyro_Z;')
f.write('Mag_X;Mag_Y;Mag_Z;')
f.write('Position_E;Position_N;Position_U;')
f.write('Position_EKF_E;Position_EKF_N;Position_EKF_U;')
f.write('Velocity_EKF_E;Velocity_EKF_N;Velocity_EKF_U')
f.write('q1;q2;q3;q4')
f.write('\n')
f.close()

while True:

    ## --------------------------------------
    ##  R E A D   GPS
    ## --------------------------------------

    # Make sure to call gps.update() every loop iteration and at least twice
    # as fast as data comes from the GPS unit (usually every second).
    # This returns a bool that's true if it parsed new data (you can ignore it
    # though if you don't care and instead look at the has_fix property).
    gps.update()
    # Every second print out current location details if there's a fix.
    # Every second print out current location details if there's a fix.
    current = time.monotonic()
    if current - last_print >= 1.0:
        last_print = current
        if not gps.has_fix:
            # Try again if we don't have a fix yet.
            print("Waiting for fix...")
            continue
        # We have a fix! (gps.has_fix is true)
        # Print out details about the fix like location, date, etc.
        print("=" * 40)  # Print a separator line.
        print(
            "Fix timestamp: {}/{}/{} {:02}:{:02}:{:02}".format(
                gps.timestamp_utc.tm_mon,  # Grab parts of the time from the
                gps.timestamp_utc.tm_mday,  # struct_time object that holds
                gps.timestamp_utc.tm_year,  # the fix time.  Note you might
                gps.timestamp_utc.tm_hour,  # not get all data like year, day,
                gps.timestamp_utc.tm_min,  # month!
                gps.timestamp_utc.tm_sec,
            )
        )
        print("Latitude: {0:.6f} degrees".format(gps.latitude))
        print("Longitude: {0:.6f} degrees".format(gps.longitude))
        print(
            "Precise Latitude: {} degs, {:2.4f} mins".format(
                gps.latitude_degrees, gps.latitude_minutes
            )
        )
        print(
            "Precise Longitude: {} degs, {:2.4f} mins".format(
                gps.longitude_degrees, gps.longitude_minutes
            )
        )
        print("Fix quality: {}".format(gps.fix_quality))
        # Some attributes beyond latitude, longitude and timestamp are optional
        # and might not be present.  Check if they're None before trying to use!
        if gps.satellites is not None:
            print("# satellites: {}".format(gps.satellites))
        if gps.altitude_m is not None:
            print("Altitude: {} meters".format(gps.altitude_m))
        if gps.speed_knots is not None:
            print("Speed: {} knots".format(gps.speed_knots))
        if gps.speed_kmh is not None:
            print("Speed: {} km/h".format(gps.speed_kmh))
        if gps.track_angle_deg is not None:
            print("Track angle: {} degrees".format(gps.track_angle_deg))
        if gps.horizontal_dilution is not None:
            print("Horizontal dilution: {}".format(gps.horizontal_dilution))
        if gps.height_geoid is not None:
            print("Height geoid: {} meters".format(gps.height_geoid))

        ## --------------------------------------
        ## F R O M  G P S   T O   E N U
        ## --------------------------------------
        latitude = gps.latitude
        longitude = gps.longitude
        altitude = gps.altitude_m + gps.height_geoid # height above ellipsoid = orthometric height + geoid separation
        velocity = gps.speed_kmh / 3.6 # km/h to m/s
        position_ENU = GPS2ENU(latitude, longitude, altitude, origin_ECEF)
        print("Position in ENU: ", position_ENU)

    dt = gps.timestamp_utc.tm_sec - gps_time_old
    if dt < 0:
        dt += 60
    print("Time interval: ", dt)
    gps_time_old = gps.timestamp_utc.tm_sec

        
    ## --------------------------------------
    ##  R E A D   I M U
    ## --------------------------------------
    print("Acceleration:")
    accel_x, accel_y, accel_z = bno.acceleration  # pylint:disable=no-member
    print("X: %0.6f  Y: %0.6f Z: %0.6f  m/s^2" % (accel_x, accel_y, accel_z))
    print("")

    print("Gyro:")
    gyro_x, gyro_y, gyro_z = bno.gyro  # pylint:disable=no-member
    print("X: %0.6f  Y: %0.6f Z: %0.6f rads/s" % (gyro_x, gyro_y, gyro_z))
    print("")

    print("Magnetometer:")
    mag_x, mag_y, mag_z = bno.magnetic  # pylint:disable=no-member
    print("X: %0.6f  Y: %0.6f Z: %0.6f uT" % (mag_x, mag_y, mag_z))
    print("")
    
    q_k, Pq_k = Attitude_MEKF(dt, np.array([accel_x,accel_y,accel_z]), np.array([gyro_x,gyro_y,gyro_z]), np.array([mag_x,mag_y,mag_z]), sigma_acc, sigma_ARW, sigma_RRW, sigma_mag ,q_k, Pq_k)
    
    # R_BODY2ENU = Quaternion2Rotation(q_k)

    x_k, P_k = Position_EKF(dt, position_ENU, velocity, np.array([accel_x,accel_y,accel_z]) , R_BODY2ENU, x_k, P_k, sigma_acc, gps.hdop, gps.vdop)
    
    f = open('data.txt', 'a')
    f.write(str(gps_time_old)+';')
    f.write(str(latitude)+';')
    f.write(str(longitude)+';')
    f.write(str(altitude)+';')
    f.write(str(velocity[0])+';')
    f.write(str(velocity[1])+';')
    f.write(str(velocity[2])+';')
    f.write(str(gps.hdop)+';')
    f.write(str(gps.vdop)+';')
    f.write(str(accel_x)+';')
    f.write(str(accel_y)+';')
    f.write(str(accel_z)+';')
    f.write(str(gyro_x)+';')
    f.write(str(gyro_y)+';')
    f.write(str(gyro_z)+';')
    f.write(str(mag_x)+';')
    f.write(str(mag_y)+';')
    f.write(str(mag_z)+';')
    f.write(str(position_ENU[0])+';')
    f.write(str(position_ENU[1])+';')
    f.write(str(position_ENU[2])+';')
    f.write(str(x_k[0])+';')
    f.write(str(x_k[1])+';')
    f.write(str(x_k[2])+';')
    f.write(str(x_k[3])+';')
    f.write(str(x_k[4])+';')
    f.write(str(x_k[5])+';')
    f.write(str(q_k[0])+';')
    f.write(str(q_k[1])+';')
    f.write(str(q_k[2])+';')
    f.write(str(q_k[3])+';')
    f.write('\n')
    f.close()