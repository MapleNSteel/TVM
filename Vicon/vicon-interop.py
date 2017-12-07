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
active_joints=6

viconData=[]
jointAngles=[]
bufferAngles=[]

bufferLength=64
angleBuffer=np.zeros((active_joints*2,bufferLength))
maxa = -1000
mina = 1000

def signal_handler(signal, frame):
	print("bye!")
	print("max angle is ", maxa)
	print("min angle is ", mina)
	#pickle.dump(viconData,open("viconData.dat","wb"))
	pickle.dump(jointAngles,open("angle.dat","wb"))
	#scipy.io.savemat('angle.mat', mdict={'jointAngles': jointAngles})
	pickle.dump(bufferAngles,open("buffer.dat","wb"))
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def getJointAngles(torso,shoulder_right,shoulder_left,arm_right,elbow_right,wrist_right_1,wrist_right_2,lowerarm_right,thumb_right,index_right,arm_left,elbow_left,wrist_left_1,wrist_left_2,lowerarm_left,thumb_left,index_left):
	global prev_angles
	global window_size
	#For Shoulder Pitch

	left_shoulderPitch_angle = np.arctan2(elbow_left.x-shoulder_left.x,-(elbow_left.y-shoulder_left.y))
	#angle_list.append(shoulderPitch_angle)

	#For Shoulder Roll
	
	if(np.cos(left_shoulderPitch_angle)!=0):
		s2=-(elbow_left.y-shoulder_left.y)/(np.linalg.norm([elbow_left.x-shoulder_left.x,elbow_left.y-shoulder_left.y,elbow_left.z-shoulder_left.z])*np.cos(left_shoulderPitch_angle))
	else:
		s2=(elbow_left.x-shoulder_left.x)/(np.sin(left_shoulderPitch_angle)*np.linalg.norm([elbow_left.x-shoulder_left.x,elbow_left.y-shoulder_left.y,elbow_left.z-shoulder_left.z]))
	left_shoulderYaw_angle = np.arctan2(s2,(elbow_left.z-shoulder_left.z)/np.linalg.norm([elbow_left.x-shoulder_left.x,elbow_left.y-shoulder_left.y,elbow_left.z-shoulder_left.z]))

	#For Shoulder Yaw

	left_shoulderRoll_angle=np.arctan2(-np.sin(left_shoulderPitch_angle)*np.cos(left_shoulderYaw_angle)*0.5*(wrist_left_1.x-shoulder_left.x + wrist_left_2.x-shoulder_left.x)
									  + np.cos(left_shoulderPitch_angle)*np.cos(left_shoulderYaw_angle)*0.5*(wrist_left_1.y-shoulder_left.y + wrist_left_2.y-shoulder_left.y)
									  + np.sin(left_shoulderYaw_angle)*0.5*(wrist_left_1.z-shoulder_left.z + wrist_left_2.z-shoulder_left.z),
									  np.cos(left_shoulderPitch_angle)*0.5*(wrist_left_1.x-shoulder_left.x + wrist_left_2.x-shoulder_left.x)+
									  np.sin(left_shoulderPitch_angle)*0.5*(wrist_left_1.y-shoulder_left.y + wrist_left_2.y-shoulder_left.y))

	if(left_shoulderRoll_angle<0):
		left_shoulderRoll_angle+=2*np.pi



	#angle_list.append(shoulderYaw_angle)

	#For Elbow Pitch
	L1 = np.linalg.norm(np.array([elbow_left.x,elbow_left.y,elbow_left.z]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z]))
	L2 = np.linalg.norm(np.array([0.5*(wrist_left_1.x + wrist_left_2.x), 0.5*(wrist_left_1.y + wrist_left_2.y), 0.5*(wrist_left_1.z + wrist_left_2.z)]) - np.array([elbow_left.x, elbow_left.y, elbow_left.z]))
	P  = np.linalg.norm(np.array([0.5*(wrist_left_1.x + wrist_left_2.x), 0.5*(wrist_left_1.y + wrist_left_2.y), 0.5*(wrist_left_1.z + wrist_left_2.z)]) - np.array([shoulder_left.x,shoulder_left.y,shoulder_left.z]))

	left_elbowPitch_angle =np.pi-np.arccos((L1*L1+L2*L2-P*P)/(2*L1*L2))
	#angle_list.append(elbowPitch_angle)
	A1=np.array([[np.cos(left_shoulderPitch_angle), -np.sin(left_shoulderPitch_angle), 0 , 0],[np.sin(left_shoulderPitch_angle), np.cos(left_shoulderPitch_angle), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
	A2=np.array([[1, 0, 0, 0],[0, np.cos(left_shoulderYaw_angle), -np.sin(left_shoulderYaw_angle), 0],[0, np.sin(left_shoulderYaw_angle), np.cos(left_shoulderYaw_angle), 0],[0, 0, 0, 1]])
	A3=np.array([[np.cos(left_shoulderRoll_angle), -np.sin(left_shoulderRoll_angle), 0, 0],[np.sin(left_shoulderRoll_angle), np.cos(left_shoulderRoll_angle), 0, 0],[0, 0, 1, L1],[0, 0, 0, 1]])

	rot=np.zeros((3,3))
	rot[:,0]=np.array([(wrist_left_1.x-wrist_left_2.x), (wrist_left_1.y-wrist_left_2.y), (wrist_left_1.z-wrist_left_2.z)])
	rot[:,0]/=np.linalg.norm(rot[:,0])
	rot[:,1]=np.array([lowerarm_left.x-0.5*(wrist_left_1.x+wrist_left_2.x), lowerarm_left.y-0.5*(wrist_left_1.y+wrist_left_2.y), lowerarm_left.z-0.5*(wrist_left_1.z+wrist_left_2.z)])
	rot[:,1]/=np.linalg.norm(rot[:,1])
	rot[:,2]=np.cross(rot[:,0], rot[:,1])

	r=A1[0:3,0:3].dot(A2[0:3,0:3]).dot(A3[0:3,0:3]).dot(rot)

	left_elbow_roll=np.arcsin(-r[1,2])
    
	v1=np.array([index_left.x-0.5*(wrist_left_1.x+wrist_left_2.x),index_left.y-0.5*(wrist_left_1.y+wrist_left_2.y),index_left.z-0.5*(wrist_left_1.z+wrist_left_2.z)])
	v2=np.array([0.5*(wrist_left_1.x+wrist_left_2.x)-elbow_left.x,0.5*(wrist_left_1.y+wrist_left_2.y)-elbow_left.y,0.5*(wrist_left_1.z+wrist_left_2.z)-elbow_left.z])
	
	left_wrist_pitch=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

	#For Shoulder Pitch

	right_shoulderPitch_angle = np.arctan2(elbow_right.x-shoulder_right.x,-(elbow_right.y-shoulder_right.y))
	#angle_list.append(shoulderPitch_angle)

	#For Shoulder Roll

	if(np.cos(right_shoulderPitch_angle)!=0):
		s2=-(elbow_right.y-shoulder_right.y)/(np.linalg.norm([elbow_right.x-shoulder_right.x,elbow_right.y-shoulder_right.y,elbow_right.z-shoulder_right.z])*np.cos(right_shoulderPitch_angle))
	else:
		s2=(elbow_right.x-shoulder_right.x)/(np.sin(right_shoulderPitch_angle)*np.linalg.norm([elbow_right.x-shoulder_right.x,elbow_right.y-shoulder_right.y,elbow_right.z-shoulder_right.z]))
	right_shoulderYaw_angle = np.arctan2(s2,(elbow_right.z-shoulder_right.z)/np.linalg.norm([elbow_right.x-shoulder_right.x,elbow_right.y-shoulder_right.y,elbow_right.z-shoulder_right.z]))

	#For Shoulder Yaw
	
	right_shoulderRoll_angle=np.arctan2(-np.sin(right_shoulderPitch_angle)*np.cos(right_shoulderYaw_angle)*0.5*(wrist_right_1.x-shoulder_right.x + wrist_right_2.x-shoulder_right.x)
									  + np.cos(right_shoulderPitch_angle)*np.cos(right_shoulderYaw_angle)*0.5*(wrist_right_1.y-shoulder_right.y + wrist_right_2.y-shoulder_right.y)
									  + np.sin(right_shoulderYaw_angle)*0.5*(wrist_right_1.z-shoulder_right.z + wrist_right_2.z-shoulder_right.z),
									  np.cos(right_shoulderPitch_angle)*0.5*(wrist_right_1.x-shoulder_right.x + wrist_right_2.x-shoulder_right.x)+
									  np.sin(right_shoulderPitch_angle)*0.5*(wrist_right_1.y-shoulder_right.y + wrist_right_2.y-shoulder_right.y))


	#angle_list.append(shoulderYaw_angle)

	#For Elbow Pitch
	L1 = np.linalg.norm(np.array([elbow_right.x,elbow_right.y,elbow_right.z]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z]))
	L2 = np.linalg.norm(np.array([0.5*(wrist_right_1.x + wrist_right_2.x), 0.5*(wrist_right_1.y + wrist_right_2.y), 0.5*(wrist_right_1.z + wrist_right_2.z)]) - np.array([elbow_right.x, elbow_right.y, elbow_right.z]))
	P  = np.linalg.norm(np.array([0.5*(wrist_right_1.x + wrist_right_2.x), 0.5*(wrist_right_1.y + wrist_right_2.y), 0.5*(wrist_right_1.z + wrist_right_2.z)]) - np.array([shoulder_right.x,shoulder_right.y,shoulder_right.z]))

	right_elbowPitch_angle =np.pi-np.arccos((L1*L1+L2*L2-P*P)/(2*L1*L2))
	#angle_list.append(elbowPitch_angle)

	A1=np.array([[np.cos(right_shoulderPitch_angle), -np.sin(right_shoulderPitch_angle), 0 , 0],[np.sin(right_shoulderPitch_angle), np.cos(right_shoulderPitch_angle), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
	A2=np.array([[1, 0, 0, 0],[0, np.cos(right_shoulderYaw_angle), -np.sin(right_shoulderYaw_angle), 0],[0, np.sin(right_shoulderYaw_angle), np.cos(right_shoulderYaw_angle), 0],[0, 0, 0, 1]])
	A3=np.array([[np.cos(right_shoulderRoll_angle), -np.sin(right_shoulderRoll_angle), 0, 0],[np.sin(right_shoulderRoll_angle), np.cos(right_shoulderRoll_angle), 0, 0],[0, 0, 1, L1],[0, 0, 0, 1]])

	rot=np.zeros((3,3))
	rot[:,0]=np.array([(wrist_right_1.x-wrist_right_2.x), (wrist_right_1.y-wrist_right_2.y), (wrist_right_1.z-wrist_right_2.z)])
	rot[:,0]/=np.linalg.norm(rot[:,0])
	rot[:,1]=np.array([lowerarm_right.x-0.5*(wrist_right_1.x+wrist_right_2.x), lowerarm_right.y-0.5*(wrist_right_1.y+wrist_right_2.y), lowerarm_right.z-0.5*(wrist_right_1.z+wrist_right_2.z)])
	rot[:,1]/=np.linalg.norm(rot[:,1])
	rot[:,2]=np.cross(rot[:,0], rot[:,1])

	r=A1[0:3,0:3].dot(A2[0:3,0:3]).dot(A3[0:3,0:3]).dot(rot)

	right_elbow_roll=np.arcsin(-r[1,2])
	
	v1=np.array([index_right.x-0.5*(wrist_right_1.x+wrist_right_2.x),index_right.y-0.5*(wrist_right_1.y+wrist_right_2.y),index_right.z-0.5*(wrist_right_1.z+wrist_right_2.z)])
	v2=np.array([0.5*(wrist_right_1.x+wrist_right_2.x)-elbow_right.x,0.5*(wrist_right_1.y+wrist_right_2.y)-elbow_right.y,0.5*(wrist_right_1.z+wrist_right_2.z)-elbow_right.z])
	
	right_wrist_pitch=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

	angleBuffer[0:,0:bufferLength-1]=angleBuffer[0:,1:bufferLength]
	angleBuffer[0:,bufferLength-1] = [left_shoulderPitch_angle,left_shoulderYaw_angle,left_shoulderRoll_angle,left_elbowPitch_angle,left_elbow_roll,left_wrist_pitch,right_shoulderPitch_angle,right_shoulderYaw_angle,right_shoulderRoll_angle,right_elbowPitch_angle,right_elbow_roll,right_wrist_pitch]
	
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
	lowerarm_right = data.markers[5].translation
	wrist_right_1 = data.markers[6].translation
	wrist_right_2 = data.markers[7].translation
	thumb_right = data.markers[8].translation
	index_right = data.markers[9].translation
	arm_left = data.markers[10].translation
	elbow_left = data.markers[11].translation
	lowerarm_left = data.markers[12].translation
	wrist_left_1 = data.markers[13].translation
	wrist_left_2 = data.markers[14].translation
	thumb_left = data.markers[15].translation
	index_left = data.markers[16].translation

	global prev_angles
	global window_size

	[left_shoulderPitch_angle,left_shoulderYaw_angle,left_shoulderRoll_angle,left_elbowPitch_angle,left_elbow_roll,left_wrist_pitch,right_shoulderPitch_angle,right_shoulderYaw_angle,right_shoulderRoll_angle,right_elbowPitch_angle,right_elbow_roll,right_wrist_pitch]=getJointAngles(torso,shoulder_right,shoulder_left,arm_right,elbow_right,wrist_right_1,wrist_right_2,lowerarm_right,thumb_right,index_right,arm_left,elbow_left,wrist_left_1,wrist_left_2,lowerarm_left,thumb_left,index_left)

	#left_b_shoulderPitch = (left_shoulderPitch_angle+np.pi/2)
	hs0_min = -0.08919059 # human s0 joint limit min 
	hs0_max = 1.754911514 # human s0 joint limit max
	rs0_min = -1.0 # robot s0 joint limit min
	rs0_max = 1.0 # robot s0 joint limit max
	left_b_shoulderPitch = getangle(hs0_min, hs0_max, rs0_min, rs0_max, left_shoulderPitch_angle)
	#left_b_shoulderYaw = left_shoulderYaw_angle-5*np.pi/6
	hs1_min = -0.0559188
	hs1_max = 2.88017836
	rs1_min = -2.147
	rs1_max = 1.047
	left_b_shoulderYaw = getangle(hs1_min, hs1_max, rs1_min, rs1_max, left_shoulderYaw_angle)
	#left_b_shoulderRoll = (left_shoulderRoll_angle)-np.pi-np.pi/4
	he0_min = -0.06421624
	he0_max = 3.55323329
	re0_min = -3.0541
	re0_max = 0
	left_b_shoulderRoll = getangle(he0_min, he0_max, re0_min, re0_max, left_shoulderRoll_angle)
	#left_b_elbowPitch = (left_elbowPitch_angle)
	he1_min = -0.07400418
	he1_max = 2.4775118
	re1_min = -0.05
	re1_max = 2.618
	left_b_elbowPitch = getangle(he1_min, he1_max, re1_min, re1_max, left_elbowPitch_angle)
	hw0_min = -1.448542
	hw0_max = 0.09527575
	rw0_min = -3.059
	rw0_max = 0
	left_b_elbowRoll = getangle(hw0_min, hw0_max, rw0_min, rw0_max, left_elbow_roll)
	hw1_min = -0.3133831
	hw1_max = 1.1209720
	rw1_min = -1.5707
	rw1_max = 2.094
	left_b_wristPitch = getangle(hw1_min, hw1_max, rw1_min, rw1_max, left_wrist_pitch)
	# set joint angles
	left_setShoulderPitch = left_b_shoulderPitch
	left_setShoulderYaw = left_b_shoulderYaw
	left_setShoulderRoll = left_b_shoulderRoll
	left_setElbowPitch = left_b_elbowPitch
	left_setElbowRoll = left_b_elbowRoll
	left_setWristPitch = left_b_wristPitch
	right_b_shoulderPitch = (right_shoulderPitch_angle-np.pi/2)
	right_b_shoulderYaw = right_shoulderYaw_angle-5*np.pi/6
	right_b_shoulderRoll = (right_shoulderRoll_angle)+np.pi/4	
	right_b_elbowPitch = (right_elbowPitch_angle)

	right_setShoulderPitch = right_b_shoulderPitch
	right_setShoulderYaw = right_b_shoulderYaw
	right_setShoulderRoll = right_b_shoulderRoll
	right_setElbowPitch = right_b_elbowPitch
	right_setElbowRoll = right_elbow_roll
	global m
	global maxa
	global mina
	left_joint_names=m.left_limb.joint_names()
	right_joint_names=m.right_limb.joint_names()
	print("Left joint angles are :", np.array([left_setShoulderPitch, left_setShoulderYaw, left_setShoulderRoll, left_setElbowPitch, left_setElbowRoll, left_wrist_pitch]))
	print("Right joint angles are :", np.array([right_setShoulderPitch, right_setShoulderYaw, right_setShoulderRoll, right_setElbowPitch, right_setElbowRoll, right_wrist_pitch]))
	a = left_wrist_pitch
	if a > maxa:
	    maxa = a
	if a < mina:
	    mina = a
	m.setLeftArmJointAngles([left_joint_names[0],left_joint_names[1],left_joint_names[2],left_joint_names[3],left_joint_names[4],left_joint_names[5]],[left_setShoulderPitch, left_setShoulderYaw, left_setShoulderRoll, left_setElbowPitch, left_setElbowRoll, left_setWristPitch])
	m.setRightArmJointAngles([right_joint_names[0],right_joint_names[1],right_joint_names[2],right_joint_names[3],right_joint_names[4],right_joint_names[5]],[right_setShoulderPitch, right_setShoulderYaw, right_setShoulderRoll, right_setElbowPitch, right_setElbowRoll, right_wrist_pitch])

def getangle(minhuman, maxhuman, minbaxter, maxbaxter, curr):
	if curr > maxhuman:
		curr = maxhuman
	if curr < minhuman:
		curr = minhuman
	k = (maxbaxter - minbaxter)/(maxhuman - minhuman)
	result = k * (curr - minhuman) + minbaxter
	if result > max(maxbaxter, minbaxter):
		result = max(maxbaxter, minbaxter)
	if result < min(minbaxter, maxbaxter):
		result = min(minbaxter, maxbaxter)
	return result

def main():
	rospy.init_node('listener', anonymous=True)
	#print("here")
	global m
	global maxa
	global mina
	maxa = -1000
	mina = 1000

	m=MoveArms()
	#print("Getting robot state... ")
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
	#print("here")
	rospy.Subscriber("/vicon/markers", Markers, callback)

	while(True):
		pass

if __name__ == '__main__':
	main()

