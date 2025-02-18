# InertialNavigationSystem

This project implements an Inertial Navigation System (INS) using MicroPython, a GNSS module, and a 9-DOF IMU. The system combines data from the GNSS module and the IMU to provide accurate positioning and orientation information. It is designed for applications requiring precise navigation and tracking capabilities.

## Features
- **MicroPython**: Lightweight and efficient scripting language for microcontrollers.
- **GNSS Module**: Provides global positioning data.
- **9-DOF IMU**: Combines accelerometer, gyroscope, and magnetometer for comprehensive motion sensing.


## Getting Started
To get started with this project, follow the setup instructions and explore the example scripts provided in the repository.

## Circuit:

https://app.cirkitdesigner.com/project/bfacb2f2-38b0-4d9b-a405-62c1167092c1

# DATASHEET & THEORY

Algorithm:

+ Attitude: https://matthewhampsey.github.io/blog/2020/07/18/mekf

+ Position: https://github.com/FedericoDD/SunCubes/tree/main/UAV_position_observer

Adafruit Feather RP2040:

+ Arduino IDE: https://learn.adafruit.com/adafruit-feather-rp2040-pico/arduino-ide-setup

Adafruit 9-DOF Orientation IMU Fusion Breakout BNO085:

+ Library: https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085/arduino

+ Python library: https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085/python-circuitpython

+ Datasheet: https://www.ceva-ip.com/wp-content/uploads/BNO080_085-Datasheet.pdf

Adafruit Ultimate GPS FeatherWing:

+ Overview: https://learn.adafruit.com/adafruit-ultimate-gps-featherwing/overview

+ Python Library: https://learn.adafruit.com/adafruit-ultimate-gps-featherwing/circuitpython-library

+ Battery: https://www.adafruit.com/product/380