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
    point_ECEF = np.array([[x], [y], [z]])
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
def Attitude_MEKF():
    pass

## --------------------------------------
##  E K F   A L G O R I T H M
## --------------------------------------
def Position_EKF():
    pass

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
        origin_ECEF = np.array([[0], [0], [0]])
        position_ENU = GPS2ENU(latitude, longitude, altitude, origin_ECEF)
        print("Position in ENU: ", position_ENU)
    
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