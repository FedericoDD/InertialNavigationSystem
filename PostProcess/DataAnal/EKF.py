import numpy as np

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
        a, b = argv[4]
    else:
        a, b = 6378137, 6356752
    
    e2 = 1 - (b/a) ** 2           # squared eccentricity
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

    # TODO: R_ECEF2ENU should have been build from the latitude and longitude of the origin.
    point_ECEF, R_ECEF2ENU = GPS2ECEF(latitude, longitude, altitude)
    position_ENU = ECEF2ENU(origin_ECEF, point_ECEF, R_ECEF2ENU)
    # 3 by 1, North, East, Down
    return position_ENU

def getSpeed(speed, track_angle):
    # speed in km/h
    # track_angle in deg
    # return ENU velocity in m/s
    speed = speed/3.6 # km/h to m/s
    track_angle_rad = track_angle*np.pi/180 # degree to radian
    return np.array([speed*np.sin(track_angle_rad), speed*np.cos(track_angle_rad), 0])

def EKF(dt,lat,lon,alt,origin,speed_kmh, track_angle, acc_meas,  x_k_1, P_k_1, sigma_acc, hdop, vdop):
    # dt: time interval (use gnss time)
    # position: 3 by 1, East, North, Up from GPS
    # velocity: 3 by 1, East, North, Up from GPS
    # acc_meas: 3 by 1, Body frame with gravity
    # R_BODY2ENU: 3 by 3, rotation matrix from body frame to ENU frame
    # x_k: 6 by 1, state vector
    # P_k: 6 by 6, covariance matrix
    # sigma_acc: 1 by 1, standard deviation of accelerometer
    # hdop: 1 by 1, horizontal dilution of precision
    # vdop: 1 by 1, vertical dilution of precision


    # Q: 6 by 6, process noise covariance matrix diag(sigma_acc)^2
    # R: 6 by 6, measurement noise covariance matrix diag(hdop^2, hdop^2, vdop^2,2*hdop^2/dt, 2*hdop^2/dt^2, 2*vdop^2/dt^2)

    # From latitude and longitude to ENU
    position = GPS2ENU(lat, lon, alt, origin)
    velocity = getSpeed(speed_kmh, track_angle)

    measurements = np.concatenate((position, velocity), axis=0)
    
    x_k = np.zeros(6)
    F = np.array([[1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 1]])
    
    B = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt]])
    
    Q = np.diag([sigma_acc**2, sigma_acc**2, sigma_acc**2]) 
    R = np.diag([hdop**2, hdop**2, vdop**2, 0.15**2, 0.15**2, 0.15**2]) # 0.1 is the standard deviation of the velocity
    V = np.dot(np.dot(B, Q), B.transpose())

    ## Prediction
    x_k = np.dot(F, x_k_1) + np.dot(B, acc_meas)
    P_k = np.dot(np.dot(F, P_k_1), F.transpose()) + V

    ## Update
    H = np.eye(6)
    K = np.dot(np.dot(P_k, H.transpose()), np.linalg.inv(np.dot(np.dot(H, P_k), H.transpose()) + R))
    x_k = x_k + np.dot(K, (measurements - np.dot(H, x_k)))
    P_k = np.dot((np.eye(6) - np.dot(K, H)), P_k)

    return x_k, P_k
