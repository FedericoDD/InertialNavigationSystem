import time
import serial
SERIAL_PORT = 'COM4'
ser = serial.Serial(SERIAL_PORT, 115200)  # open serial port

command = b'\r'
print(f"Start communication")

f = open('data.txt', 'w')
f.write('dTime;')
f.write('Accel_X;Accel_Y;Accel_Z;')
f.write('Gyro_X;Gyro_Y;Gyro_Z;')
f.write('Mag_X;Mag_Y;Mag_Z;')
f.write('q1;q2;q3;q4')
f.write('latitude;longitude;altitude;velocity;track_angle')
f.write('\n')
f.write('[s];')
f.write('[m/s^2];[m/s^2];[m/s^2];')
f.write('[rad/s];[rad/s];[rad/s];')
f.write('[uT];[uT];[uT];')
f.write('[-];[-];[-];[-]')
f.write('[deg];[deg];[m];[km/h];[deg]')
f.write('\n')
f.close()


while True:
    try:
        '''
        ser.write(command)     # write a string
        ended = False
        

        for _ in range(len(command)*4):
            a = ser.read() # Read the loopback chars and ignore
        '''
        reply = b''
        while True:
            a = ser.read()

            if a== b'\r':
                break

            else:
                reply += a

            #time.sleep(0.01)
        
        f = open('data.txt', 'a')
        try: 
            #print(reply.decode('utf'))
            f.write(reply.decode('utf'))
            f.close()
        except:    
            print('Error')
            f.close()
    except KeyboardInterrupt:
        ser.close()