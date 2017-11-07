import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import sys, signal

from vicon_bridge.msg import Markers
from std_msgs.msg import Float32MultiArray

import cPickle as pickle

import rospy
from geometry_msgs import *

from MoveArms.moveArms import *

import math

# global prev_angles
window_size=1
active_joints=4
prev_angles=np.zeros((window_size,active_joints))
stable_angles=np.zeros((window_size,active_joints))

viconData=[]

def signal_handler(signal, frame):
	print("bye!")
	#pickle.dump(viconData,open("vicon.dat","wb"))
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def getJointAngles(shoulder_right,shoulder_left,torso,elbow_right,wrist_right,elbow_left,wrist_left):
	global prev_angles
	global window_size
	#Create Vectors

	#For Shoulder Pitch
	left_shoulderPitch_v0 = np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z]) - np.array([torso.x,torso.y,torso.z])
	left_shoulderPitch_v1 = np.array([shoulder_left.x, shoulder_left.y, shoulder_left.z]) - np.array([elbow_left.x, elbow_left.y, elbow_left.z])
	
	left_dot_product = np.dot(left_shoulderPitch_v0,left_shoulderPitch_v1)
	left_mag_v0 = np.linalg.norm(left_shoulderPitch_v0)
	left_mag_v1 = np.linalg.norm(left_shoulderPitch_v1)

	left_shoulderPitch_angle = (180/math.pi)*np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))
	#angle_list.append(shoulderPitch_angle)

	#For Shoulder Yaw
	left_shoulderYaw_v0 = np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z])
	left_shoulderYaw_v1 = np.array([shoulder_left.x, shoulder_left.y, shoulder_left.z]) - np.array([elbow_left.x, elbow_left.y, elbow_left.z])

	left_dot_product = np.dot(left_shoulderYaw_v0,left_shoulderYaw_v1)
	left_mag_v0 = np.linalg.norm(left_shoulderYaw_v0)
	left_mag_v1 = np.linalg.norm(left_shoulderYaw_v1)

	left_shoulderYaw_angle = (180/math.pi)*np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))
	#angle_list.append(shoulderYaw_angle)

	#For Elbow Pitch
	left_elbowPitch_v0 = np.array([elbow_left.x,elbow_left.y,elbow_left.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z])
	left_elbowPitch_v1 = np.array([elbow_left.x, elbow_left.y, elbow_left.z]) - np.array([wrist_left.x, wrist_left.y, wrist_left.z])

	left_dot_product = np.dot(left_elbowPitch_v0,left_elbowPitch_v1)
	left_mag_v0 = np.linalg.norm(left_elbowPitch_v0)
	left_mag_v1 = np.linalg.norm(left_elbowPitch_v1)

	left_elbowPitch_angle = (180/math.pi)*np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))
	#angle_list.append(elbowPitch_angle)

	#For Shoulder Roll
	left_shoulderRoll_v0_1 = np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z])
	left_shoulderRoll_v1_1 = np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z]) - np.array([elbow_left.x,elbow_left.y,elbow_left.z])
	
	left_shoulderRoll_v0_2 = np.array([elbow_left.x, elbow_left.y, elbow_left.z]) - np.array([wrist_left.x, wrist_left.y, wrist_left.z])
	left_shoulderRoll_v1_2 = np.array([elbow_left.x, elbow_left.y, elbow_left.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z])
	
	left_EM = np.cross(left_shoulderRoll_v0_1, left_shoulderRoll_v1_1) #perpedicular to plane DEA
	left_AH = np.cross(left_shoulderRoll_v0_2, left_shoulderRoll_v1_2) #perpendicular to plane EAB

	left_dot_product = np.dot(left_EM, left_AH)
	left_mag_v0 = np.linalg.norm(left_EM)
	left_mag_v1 = np.linalg.norm(left_AH)

	left_shoulderRoll_angle = (180/math.pi)*np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))


	#For Shoulder Pitch
	right_shoulderPitch_v0 = np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z]) - np.array([torso.x,torso.y,torso.z])
	right_shoulderPitch_v1 = np.array([shoulder_right.x, shoulder_right.y, shoulder_right.z]) - np.array([elbow_right.x, elbow_right.y, elbow_right.z])
	
	right_dot_product = np.dot(right_shoulderPitch_v0,right_shoulderPitch_v1)
	right_mag_v0 = np.linalg.norm(right_shoulderPitch_v0)
	right_mag_v1 = np.linalg.norm(right_shoulderPitch_v1)

	right_shoulderPitch_angle = (180/math.pi)*np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))
	#angle_list.append(shoulderPitch_angle)

	#For Shoulder Yaw
	right_shoulderYaw_v0 = np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z])
	right_shoulderYaw_v1 = np.array([shoulder_right.x, shoulder_right.y, shoulder_right.z]) - np.array([elbow_left.x, elbow_left.y, elbow_left.z])

	right_dot_product = np.dot(right_shoulderYaw_v0,right_shoulderYaw_v1)
	right_mag_v0 = np.linalg.norm(right_shoulderYaw_v0)
	right_mag_v1 = np.linalg.norm(right_shoulderYaw_v1)

	right_shoulderYaw_angle = (180/math.pi)*np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))
	#angle_list.append(shoulderYaw_angle)

	#For Elbow Pitch
	right_elbowPitch_v0 = np.array([elbow_right.x,elbow_right.y,elbow_right.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z])
	right_elbowPitch_v1 = np.array([elbow_right.x, elbow_right.y, elbow_right.z]) - np.array([wrist_right.x, wrist_right.y, wrist_right.z])

	right_dot_product = np.dot(right_elbowPitch_v0,right_elbowPitch_v1)
	right_mag_v0 = np.linalg.norm(right_elbowPitch_v0)
	right_mag_v1 = np.linalg.norm(right_elbowPitch_v1)

	right_elbowPitch_angle = (180/math.pi)*np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))
	#angle_list.append(elbowPitch_angle)

	#For Shoulder Roll
	right_shoulderRoll_v0_1 = np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z])
	right_shoulderRoll_v1_1 = np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z]) - np.array([elbow_right.x,elbow_right.y,elbow_right.z])
	
	right_shoulderRoll_v0_2 = np.array([elbow_right.x, elbow_right.y, elbow_right.z]) - np.array([wrist_right.x, wrist_right.y, wrist_right.z])
	right_shoulderRoll_v1_2 = np.array([elbow_right.x, elbow_right.y, elbow_right.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z])
	
	right_EM = np.cross(right_shoulderRoll_v0_1, right_shoulderRoll_v1_1) #perpedicular to plane DEA
	right_AH = np.cross(right_shoulderRoll_v0_2, right_shoulderRoll_v1_2) #perpendicular to plane EAB

	right_dot_product = np.dot(right_EM, right_AH)
	right_mag_v0 = np.linalg.norm(right_EM)
	right_mag_v1 = np.linalg.norm(right_AH)

	right_shoulderRoll_angle = (180/math.pi)*np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))

	return [left_shoulderPitch_angle,left_shoulderYaw_angle,left_shoulderRoll_angle,left_elbowPitch_angle,
			right_shoulderPitch_angle,right_shoulderYaw_angle,right_shoulderRoll_angle,right_elbowPitch_angle]

def callback(data):

	#viconData.append(data)
	shoulder_right = data.markers[1].translation
	shoulder_left = data.markers[2].translation
	torso = data.markers[0].translation
	elbow_right = data.markers[4].translation
	wrist_right = data.markers[6].translation
	elbow_left = data.markers[12].translation
	wrist_left = data.markers[14].translation

	global prev_angles
	global window_size

	[left_shoulderPitch_angle,left_shoulderYaw_angle,left_shoulderRoll_angle,left_elbowPitch_angle
	,right_shoulderPitch_angle,right_shoulderYaw_angle,right_shoulderRoll_angle,right_elbowPitch_angle]=getJointAngles(shoulder_right,shoulder_left,torso,elbow_right,wrist_right,elbow_left,wrist_left)

	left_b_shoulderPitch = -np.pi*(left_shoulderPitch_angle-90)/180
	left_b_shoulderYaw = np.pi*(left_shoulderYaw_angle-30)/180
	left_b_shoulderRoll = np.pi*(-left_shoulderRoll_angle+90)/180
	left_b_elbowPitch = np.pi/2 - np.pi*(left_elbowPitch_angle-55)/180

	left_setShoulderPitch = left_b_shoulderPitch
	left_setShoulderYaw = left_b_shoulderYaw
	left_setShoulderRoll = left_b_shoulderRoll
	left_setElbowPitch = left_b_elbowPitch

	right_b_shoulderPitch = np.pi*(right_shoulderPitch_angle-90)/180
	right_b_shoulderYaw = np.pi*(right_shoulderYaw_angle-30)/(180*3)
	right_b_shoulderRoll = np.pi*(right_shoulderRoll_angle-90)/180
	right_b_elbowPitch = np.pi/2 - np.pi*(right_elbowPitch_angle-55)/180

	right_setShoulderPitch = right_b_shoulderPitch
	right_setShoulderYaw = right_b_shoulderYaw
	right_setShoulderRoll = right_b_shoulderRoll
	right_setElbowPitch = right_b_elbowPitch

	global m
	left_joint_names=m.left_limb.joint_names()
	m.setLeftArmJointAngles([left_joint_names[0],left_joint_names[1],left_joint_names[2],left_joint_names[3]],[left_setShoulderPitch, left_setShoulderYaw, left_setShoulderRoll, left_setElbowPitch]);
	print([left_setShoulderPitch, left_setShoulderYaw, left_setShoulderRoll, left_setElbowPitch])

	right_joint_names=m.right_limb.joint_names()
	m.setRightArmJointAngles([right_joint_names[0],right_joint_names[1],right_joint_names[2],right_joint_names[3]],[right_setShoulderPitch, right_setShoulderYaw, right_setShoulderRoll, right_setElbowPitch]);
	print([right_setShoulderPitch, right_setShoulderYaw, right_setShoulderRoll, right_setElbowPitch])

def main():
	rospy.init_node('listener', anonymous=True)
	global m
	m=MoveArms()
	print("Getting robot state... ")
	rs = baxter_interface.RobotEnable(CHECK_VERSION)
	init_state = rs.state().enabled
	rate = rospy.Rate(100) # 10hz
	def clean_shutdown():
		print("\nExiting example...")
		if not init_state:
			print("Disabling robot...")
			rs.disable()
	rospy.on_shutdown(clean_shutdown)
	print("Enabling robot... ")
	rs.enable()
	rospy.Subscriber("/vicon/markers", Markers, callback)

	while(True):
		pass

if __name__ == '__main__':
	main()

