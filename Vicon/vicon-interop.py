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

import numpy, scipy.io

firFilter=np.array([-0.0001,0.0000,0.0002,0.0006,0.0013,0.0022,0.0036,0.0051,0.0068,0.0085,0.0097,
0.0103,0.0099,0.0082,0.0051,0.0008,-0.0044,-0.0100,-0.0152,-0.0191,-0.0206,-0.0191,-0.0138,-0.0045,
0.0087,0.0250,0.0433,0.0623,0.0803,0.0955,0.1066,0.1125,0.1125,0.1066,0.0955,0.0803,0.0623,0.0433,
0.0250,0.0087,-0.0045,-0.0138,-0.0191,-0.0206,-0.0191,-0.0152,-0.0100,-0.0044,0.0008,0.0051,0.0082,
0.0099,0.0103,0.0097,0.0085,0.0068,0.0051,0.0036,0.0022,0.0013,0.0006,0.0002,0.0000,-0.0001])

# global prev_angles
window_size=1
active_joints=5

viconData=[]
jointAngles=[]
bufferAngles=[]

bufferLength=64
angleBuffer=np.zeros((active_joints*2,bufferLength))

def signal_handler(signal, frame):
	print("bye!")
	#pickle.dump(viconData,open("viconData.dat","wb"))
	pickle.dump(jointAngles,open("angle.dat","wb"))
	#scipy.io.savemat('angle.mat', mdict={'jointAngles': jointAngles})
	pickle.dump(bufferAngles,open("buffer.dat","wb"))
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def getJointAngles(torso,shoulder_right,shoulder_left,arm_right,elbow_right,wrist_right_1,wrist_right_2,palm_right,thumb_right,index_right,arm_left,elbow_left,wrist_left_1,wrist_left_2,palm_left,thumb_left,index_left):
	global prev_angles
	global window_size
	#Create Vectors

	ex = np.array([shoulder_left.x-shoulder_right.x,shoulder_left.y-shoulder_right.y,shoulder_left.z-shoulder_right.z])
	ex = ex/np.linalg.norm(ex)
	ey = np.array([0.5*(shoulder_left.x+shoulder_right.x)-torso.x,0.5*(shoulder_left.y+shoulder_right.y)-torso.y,0.5*(shoulder_left.z+shoulder_right.z)-torso.z])
	ey = ey/np.linalg.norm(ey)
	v1 = np.array([shoulder_left.x-shoulder_right.x,shoulder_left.y-shoulder_right.y,shoulder_left.z-shoulder_right.z])
	v2 = np.array([torso.x-shoulder_right.x,torso.y-shoulder_right.y,torso.z-shoulder_right.z])
	ez = np.cross(v1,v2)
	ez = ez/np.linalg.norm(ez)
	#For Shoulder Pitch
	left_shoulderPitch_v0 = np.array([elbow_left.x-shoulder_left.x,elbow_left.y-shoulder_left.y ,elbow_left.z-shoulder_left.z ])
	left_shoulderPitch_v1 = np.array([torso.x - 0.5*(shoulder_left.x+shoulder_right.x),torso.y - 0.5*(shoulder_left.y+shoulder_right.y),torso.z - 0.5*(shoulder_left.z+shoulder_right.z)])
	
	left_dot_product = np.dot(left_shoulderPitch_v0,left_shoulderPitch_v1)
	left_mag_v0 = np.linalg.norm(left_shoulderPitch_v0)
	left_mag_v1 = np.linalg.norm(left_shoulderPitch_v1)

	left_shoulderPitch_angle = np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))
	#angle_list.append(shoulderPitch_angle)

	#For Shoulder Yaw
	left_shoulderYaw_v0 = np.array([elbow_left.x,elbow_left.y,elbow_left.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z])
	left_shoulderYaw_v0 = left_shoulderYaw_v0 - np.dot(left_shoulderYaw_v0,ey)*ey
	left_shoulderYaw_v1 = ex
	left_dot_product = np.dot(left_shoulderYaw_v0,left_shoulderYaw_v1)
	left_mag_v0 = np.linalg.norm(left_shoulderYaw_v0)
	left_mag_v1 = np.linalg.norm(left_shoulderYaw_v1)

	left_shoulderYaw_angle = -np.pi+np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))
	#angle_list.append(shoulderYaw_angle)

	#For Elbow Pitch
	left_elbowPitch_v0 = np.array([elbow_left.x,elbow_left.y,elbow_left.z]) - np.array([arm_left.x,arm_left.y,arm_left.z])
	left_elbowPitch_v1 = np.array([0.5*(wrist_left_1.x + wrist_left_2.x), 0.5*(wrist_left_1.y + wrist_left_2.y), 0.5*(wrist_left_1.z + wrist_left_2.z)]) - np.array([elbow_left.x, elbow_left.y, elbow_left.z])

	left_dot_product = np.dot(left_elbowPitch_v0,left_elbowPitch_v1)
	left_mag_v0 = np.linalg.norm(left_elbowPitch_v0)
	left_mag_v1 = np.linalg.norm(left_elbowPitch_v1)

	left_elbowPitch_angle =np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))
	#angle_list.append(elbowPitch_angle)

	#For Shoulder Roll
	left_shoulderRoll_v0_1 = np.array([arm_left.x,arm_left.y,arm_left.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z])
	left_shoulderRoll_v1_1 = np.array([elbow_left.x,elbow_left.y,elbow_left.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z])
	
	left_shoulderRoll_v0_2 = np.array([wrist_left_1.x, wrist_left_1.y, wrist_left_1.z]) - np.array([elbow_left.x,elbow_left.y,elbow_left.z])
	left_shoulderRoll_v1_2 = np.array([wrist_left_2.x, wrist_left_2.y, wrist_left_2.z]) - np.array([elbow_left.x, elbow_left.y, elbow_left.z])
	
	left_EM = np.cross(left_shoulderRoll_v0_1, left_shoulderRoll_v1_1) #perpedicular to plane DEA
	left_AH = np.cross(left_shoulderRoll_v0_2, left_shoulderRoll_v1_2) #perpendicular to plane EAB

	left_dot_product = np.dot(left_EM, left_AH)
	left_mag_v0 = np.linalg.norm(left_EM)
	left_mag_v1 = np.linalg.norm(left_AH)

	left_shoulderRoll_angle = np.arccos(1.0*(left_dot_product)/(left_mag_v0*left_mag_v1))
    
	v1=np.cross(np.array([palm_left.x-wrist_left_1.x,palm_left.y-wrist_left_1.y,palm_left.z-wrist_left_1.z]),np.array([wrist_left_2.x-wrist_left_1.x,wrist_left_2.y-wrist_left_1.y,wrist_left_2.z-wrist_left_1.z]))
	v2=np.cross(np.array([wrist_left_2.x-wrist_left_1.x,wrist_left_2.y-wrist_left_1.y,wrist_left_2.z-wrist_left_1.z]),np.array([wrist_left_1.x-elbow_left.x,wrist_left_1.y-elbow_left.y,wrist_left_1.z-elbow_left.z]))
	
	left_wrist_pitch=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

	#For Shoulder Pitch
	right_shoulderPitch_v0 = np.array([elbow_right.x-shoulder_right.x,elbow_right.y-shoulder_right.y ,elbow_right.z-shoulder_right.z ])
	right_shoulderPitch_v1 = np.array([torso.x - 0.5*(shoulder_right.x+shoulder_right.x),torso.y - 0.5*(shoulder_right.y+shoulder_right.y),torso.z - 0.5*(shoulder_right.z+shoulder_right.z)])
	
	right_dot_product = np.dot(right_shoulderPitch_v0,right_shoulderPitch_v1)
	right_mag_v0 = np.linalg.norm(right_shoulderPitch_v0)
	right_mag_v1 = np.linalg.norm(right_shoulderPitch_v1)

	right_shoulderPitch_angle = np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))
	#angle_list.append(shoulderPitch_angle)

	#For Shoulder Yaw
	right_shoulderYaw_v0 = np.array([elbow_right.x,elbow_right.y,elbow_right.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z])
	right_shoulderYaw_v0 = right_shoulderYaw_v0 - np.dot(right_shoulderYaw_v0,ey)*ey
	right_shoulderYaw_v1 = ex
	right_dot_product = np.dot(right_shoulderYaw_v0,right_shoulderYaw_v1)
	right_mag_v0 = np.linalg.norm(right_shoulderYaw_v0)
	right_mag_v1 = np.linalg.norm(right_shoulderYaw_v1)

	right_shoulderYaw_angle = -np.pi+np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))
	#angle_list.append(shoulderYaw_angle)

	#For Elbow Pitch
	right_elbowPitch_v0 = np.array([elbow_right.x,elbow_right.y,elbow_right.z]) - np.array([arm_right.x,arm_right.y,arm_right.z])
	right_elbowPitch_v1 = np.array([0.5*(wrist_right_1.x + wrist_right_2.x), 0.5*(wrist_right_1.y + wrist_right_2.y), 0.5*(wrist_right_1.z + wrist_right_2.z)]) - np.array([elbow_right.x, elbow_right.y, elbow_right.z])


	right_dot_product = np.dot(right_elbowPitch_v0,right_elbowPitch_v1)
	right_mag_v0 = np.linalg.norm(right_elbowPitch_v0)
	right_mag_v1 = np.linalg.norm(right_elbowPitch_v1)

	right_elbowPitch_angle = np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))
	#angle_list.append(elbowPitch_angle)

	#For Shoulder Roll
	left_shoulderRoll_v0_2 = np.array([wrist_left_1.x, wrist_left_1.y, wrist_left_1.z]) - np.array([elbow_left.x,elbow_left.y,elbow_left.z])
	left_shoulderRoll_v1_2 = np.array([wrist_left_2.x, wrist_left_2.y, wrist_left_2.z]) - np.array([elbow_left.x, elbow_left.y, elbow_left.z])

	right_shoulderRoll_v0_1 = np.array([arm_right.x,arm_right.y,arm_right.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z])
	right_shoulderRoll_v1_1 = np.array([elbow_right.x,elbow_right.y,elbow_right.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z])
	
	right_shoulderRoll_v0_2 = np.array([wrist_right_1.x, wrist_right_1.y, wrist_right_1.z]) - np.array([elbow_right.x, elbow_right.y, elbow_right.z])
	right_shoulderRoll_v1_2 = np.array([wrist_right_2.x, wrist_right_2.y, wrist_right_2.z]) - np.array([elbow_right.x, elbow_right.y, elbow_right.z])
	
	right_EM = np.cross(right_shoulderRoll_v0_1, right_shoulderRoll_v1_1) #perpedicular to plane DEA
	right_AH = np.cross(right_shoulderRoll_v0_2, right_shoulderRoll_v1_2) #perpendicular to plane EAB

	right_dot_product = np.dot(right_EM, right_AH)
	right_mag_v0 = np.linalg.norm(right_EM)
	right_mag_v1 = np.linalg.norm(right_AH)

	right_shoulderRoll_angle = np.arccos(1.0*(right_dot_product)/(right_mag_v0*right_mag_v1))
	
	v1=np.cross(np.array([palm_right.x-wrist_right_1.x,palm_right.y-wrist_right_1.y,palm_right.z-wrist_right_1.z]),np.array([wrist_right_2.x-wrist_right_1.x,wrist_right_2.y-wrist_right_1.y,wrist_right_2.z-wrist_right_1.z]))
	v2=np.cross(np.array([wrist_right_2.x-elbow_right.x,wrist_right_2.y-elbow_right.y,wrist_right_2.z-elbow_right.z]),np.array([wrist_right_1.x-elbow_right.x,wrist_right_1.y-elbow_right.y,wrist_right_1.z-elbow_right.z]))
	
	right_wrist_pitch=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

	angleBuffer[0:,0:bufferLength-1]=angleBuffer[0:,1:bufferLength]
	angleBuffer[0:,bufferLength-1] = [left_shoulderPitch_angle,left_shoulderYaw_angle,left_shoulderRoll_angle,left_elbowPitch_angle,left_wrist_pitch,	right_shoulderPitch_angle,right_shoulderYaw_angle,right_shoulderRoll_angle,right_elbowPitch_angle,right_wrist_pitch]
	
	angles=np.dot(firFilter[::-1],np.transpose(angleBuffer))    
    
	jointAngles.append([left_shoulderPitch_angle,left_shoulderYaw_angle,left_shoulderRoll_angle,left_elbowPitch_angle,
	right_shoulderPitch_angle,right_shoulderYaw_angle,right_shoulderRoll_angle,right_elbowPitch_angle])
	
	bufferAngles.append(angles)

	return angles

def callback(data):

	viconData.append(data)
	torso = data.markers[0].translation
	shoulder_right = data.markers[1].translation
	shoulder_left = data.markers[2].translation
	arm_right = data.markers[3].translation
	elbow_right = data.markers[4].translation
	wrist_right_1 = data.markers[5].translation
	wrist_right_2 = data.markers[6].translation
	palm_right = data.markers[7].translation
	thumb_right = data.markers[8].translation
	index_right = data.markers[9].translation
	arm_left = data.markers[10].translation
	elbow_left = data.markers[11].translation
	wrist_left_1 = data.markers[12].translation
	wrist_left_2 = data.markers[13].translation
	palm_left = data.markers[14].translation
	thumb_left = data.markers[15].translation
	index_left = data.markers[16].translation

	global prev_angles
	global window_size

	[left_shoulderPitch_angle,left_shoulderYaw_angle,left_shoulderRoll_angle,left_elbowPitch_angle,left_wrist_pitch,right_shoulderPitch_angle,right_shoulderYaw_angle,right_shoulderRoll_angle,right_elbowPitch_angle,right_wrist_pitch]=getJointAngles(torso,shoulder_right,shoulder_left,arm_right,elbow_right,wrist_right_1,wrist_right_2,palm_right,thumb_right,index_right,arm_left,elbow_left,wrist_left_1,wrist_left_2,palm_left,thumb_left,index_left)

	left_b_shoulderPitch = (left_shoulderPitch_angle-np.pi/2)
	left_b_shoulderYaw = np.pi*((183/150)*60-123)/180-183*(left_shoulderYaw_angle)/150
	left_b_shoulderRoll = -np.pi/2-(left_shoulderRoll_angle)
	left_b_elbowPitch = (left_elbowPitch_angle)

	left_setShoulderPitch = left_b_shoulderPitch
	left_setShoulderYaw = left_b_shoulderYaw
	left_setShoulderRoll = left_b_shoulderRoll
	left_setElbowPitch = left_b_elbowPitch

	right_b_shoulderPitch = -(right_shoulderPitch_angle-np.pi/2)
	right_b_shoulderYaw = np.pi*((183/150)*60-123)/180-183*(right_shoulderYaw_angle)/150
	right_b_shoulderRoll = np.pi/2+(right_shoulderRoll_angle)
	right_b_elbowPitch = (right_elbowPitch_angle)

	right_setShoulderPitch = right_b_shoulderPitch
	right_setShoulderYaw = right_b_shoulderYaw
	right_setShoulderRoll = right_b_shoulderRoll
	right_setElbowPitch = right_b_elbowPitch

	global m
	left_joint_names=m.left_limb.joint_names()
	m.setLeftArmJointAngles([left_joint_names[0],left_joint_names[1],left_joint_names[2],left_joint_names[3],left_joint_names[5]],[left_setShoulderPitch, left_setShoulderYaw, left_setShoulderRoll, left_setElbowPitch, left_wrist_pitch]);
	print(180*np.array([left_setShoulderPitch, left_setShoulderYaw, left_setShoulderRoll, left_setElbowPitch, left_wrist_pitch])/np.pi)

	right_joint_names=m.right_limb.joint_names()
	m.setRightArmJointAngles([right_joint_names[0],right_joint_names[1],right_joint_names[2],right_joint_names[3],right_joint_names[5]],[right_setShoulderPitch, right_setShoulderYaw, right_setShoulderRoll, right_setElbowPitch, right_wrist_pitch]);
	print(180*np.array([right_setShoulderPitch, right_setShoulderYaw, right_setShoulderRoll, right_setElbowPitch, right_wrist_pitch])/np.pi)

def main():
	rospy.init_node('listener', anonymous=True)
	global m
	m=MoveArms()
	print("Getting robot state... ")
	'''rs = baxter_interface.RobotEnable(CHECK_VERSION)
	init_state = rs.state().enabled
	rate = rospy.Rate(100) # 10hz
	def clean_shutdown():
		print("\nExiting example...")
		if not init_state:
			print("Disabling robot...")
			rs.disable()
	rospy.on_shutdown(clean_shutdown)
	print("Enabling robot... ")
	rs.enable()'''
	rospy.Subscriber("/vicon/markers", Markers, callback)

	while(True):
		pass

if __name__ == '__main__':
	main()

