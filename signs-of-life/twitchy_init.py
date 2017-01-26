# === Twitchy Initialization Procedure ====
# CMPUT 607 W17 University of Alberta
# Revision history: 20 January 2017 by Pilarski
# Uses: Georgia Tech Research Corporation's "lib_robotis_hack.py" 
# (Library assumes use of Python 2, and needs to have pyserial2 installed)

# NOTE: this script is meant to be run in the command line,
# operation by operation, so you get experience
# with each of the commands conatained here.

# !!!! Have you read the AX12 datasheets? If not, do so now !!!!

# 1) Plug in power to the robot's barrell connector
# 2) Plug the USB2Dynamixel (set to TTL) into the USB port of your computer
# 3a) Plug the cable into the proximal servo; make sure only *one* servo is connected to the USB2Dynamixel
# 3b) Plug in servo cable to USB2Dynamixel
# 4) Initialize and try out the first servo as follows

from lib_robotis_hack import *

# At this point, check your /dev/ directory for the location of the USB to Serial device; if in doubt, unplug the USB2Dynamixel, look at the /dev/ directory, then plug it back in and see which device has been added. 

# For example, the upper right USB port on my computer gives the device: tty.usbserial-AI03QDFW, while the the bottom left gives /dev/tty.usbserial-AI0282TZ

# Create the USB to Serial channel
# Use the device you identified above, and baud of 1Mbps
D = USB2Dynamixel_Device(dev_name="/dev/ttyUSB0",baudrate=1000000)

# Identify servos on the bus (should be only ONE at this point)
# Should return: "FOUND A SERVO @ ID 1"
s_list = find_servos(D)

# Make a servo object for this servo
s1 = Robotis_Servo(D,s_list[1])

# Sample observations (signals) from servos
# i.e., make sure all is working properly

for i in range(5):
    s1.move_angle(-2.0)
    s1.move_angle(2.0)

sys.exit(0)
# Send commands to servo
s1.move_angle(0.0)
s1.move_to_encoder(512)
s1.disable_torque()
s1.enable_torque()

# Rename the servo
# All servos start with an ID of 1 right out of the box
# They need to have unique names or confusion will fill the serial bus
# Set this servo to ID "2"
s1.write_id(2)

### If you are not running command line, add a wait command here ###

# 5) Initialize the second servo 

# Plug in the next/remaining servo on your bus

# Rescan for servos
# Should return:
# "FOUND A SERVO @ ID 1"
# "FOUND A SERVO @ ID 2"
s_list = find_servos(D)

# Rename the second servo
# Set this servo to ID "3"
# Why, you ask? So that you can always plug in a new servo and have immediate access to it as ID "1" without overlapping with an existing servo on the bus
s1 = Robotis_Servo(D,s_list[0])
s1.write_id(3)

# Rescan for the new servo names
# Should return:
# "FOUND A SERVO @ ID 2"
# "FOUND A SERVO @ ID 3"
s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
s2 = Robotis_Servo(D,s_list[1])

# 6) Test out both servos to make sure your robot is working as expected
s1.read_angle()
s1.read_load()
s2.read_angle()
s2.read_load()

s1.move_angle(0.0); s2.move_angle(0.0)
s1.move_angle(0.5); s2.move_angle(-0.5)
s1.move_angle(-0.5); s2.move_angle(0.5)

# Congratulations! You have now completed the robot initialization process.

