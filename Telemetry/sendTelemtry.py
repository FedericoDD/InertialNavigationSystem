import supervisor

import time
import board
import busio
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
t_0 = time.monotonic()
while True:
    accel_x, accel_y, accel_z = bno.acceleration  # pylint:disable=no-member
    gyro_x, gyro_y, gyro_z = bno.gyro  # pylint:disable=no-member
    mag_x, mag_y, mag_z = bno.magnetic  # pylint:disable=no-member
    quat_i, quat_j, quat_k, quat_real = bno.quaternion  # pylint:disable=no-member
    
    #if supervisor.runtime.serial_bytes_available:
    #    value = input().strip()
    #    #print(f"Received: {value}\r")
    #    #print(f"{accel_x};{accel_y};{accel_z};{gyro_x};{gyro_y};{ gyro_z};{mag_x};{mag_y};{mag_z};{quat_i};{ quat_j};{ quat_k};{quat_real}\r")
    #    # reformat the previous line to have only 3 decimal for each measurement
    dt=time.monotonic()-t_0
    print(f"{dt:.6f};{accel_x:.6f};{accel_y:.6f};{accel_z:.6f};{gyro_x:.6f};{gyro_y:.6f};{gyro_z:.6f};{mag_x:.6f};{mag_y:.6f};{mag_z:.6f};{quat_i:.6f};{quat_j:.6f};{quat_k:.6f};{quat_real:.6f}\r")
    

