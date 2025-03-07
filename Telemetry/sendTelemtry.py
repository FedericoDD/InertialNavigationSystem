'''File executed after boot.py'''

# import ReadIMU

import time
import board
import busio

################ I M U : START ###########################
import adafruit_bno08x
from adafruit_bno08x.uart import BNO08X_UART
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
)


uart = busio.UART(board.IO17, board.IO18, baudrate=3000000, receiver_buffer_size=2048)

bno = BNO08X_UART(uart)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_MAGNETOMETER)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
################ I M U : END ###########################

################ G N S S : START ###########################
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


gps.send_command(b"PMTK314,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")

# Set update rate to once a second (1hz) which is what you typically want.
gps.send_command(b"PMTK220,1000")

################ G N S S : END ###########################

t_0 = time.monotonic()
last_print = t_0
while True:
    accel_x, accel_y, accel_z = bno.acceleration  # pylint:disable=no-member
    gyro_x, gyro_y, gyro_z = bno.gyro  # pylint:disable=no-member
    mag_x, mag_y, mag_z = bno.magnetic  # pylint:disable=no-member
    quat_i, quat_j, quat_k, quat_real = bno.quaternion  # pylint:disable=no-member
    
    gps.update()
    # Every second print out current location details if there's a fix.
    current = time.monotonic()
    if current - last_print >= 1.0:
        last_print = current
        if not gps.has_fix:
            # Try again if we don't have a fix yet.
            longitude = float('NaN')
            latitude = float('NaN')
            altitude = float('NaN')
            velocity = float('NaN')
            trackangle = float('NaN')
            continue
        
        longitude = gps.longitude_degrees
        latitude = gps.longitude_minutes
        altitude = gps.altitude_m + gps.height_geoid
        velocity = gps.speed_kmh
        trackangle = gps.track_angle_deg
        last_print = current

    dt=time.monotonic()-t_0
    print(f"{dt:.6f};{accel_x:.6f};{accel_y:.6f};{accel_z:.6f};{gyro_x:.6f};{gyro_y:.6f};{gyro_z:.6f};{mag_x:.6f};{mag_y:.6f};{mag_z:.6f};{quat_i:.6f};{quat_j:.6f};{quat_k:.6f};{quat_real:.6f};{latitude:.6f};{longitude:.6f};{altitude:.6f};{velocity:.6f};{trackangle:.6f}\r")
    

