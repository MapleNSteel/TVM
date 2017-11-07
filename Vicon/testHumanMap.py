import numpy as np
import math
import sys, signal
from moveArms import *

positions=[[268.868499756,1041.41772461,1336.46142578],[454.274932861,933.246643066,1331.62414551],[61.1024284363,977.05480957,1340.4498291],[598.723571777,996.747131348,1138.19140625],[-65.6389694214,1065.03173828,1132.72827148],[466.533508301,1165.55529785,1040.90234375],[101.381401062,1196.74304199,1038.7208252]]
positions_next=[[268.868499756,1041.41772461,1336.46142578],[454.274932861,933.246643066,1331.62414551],[61.1024284363,977.05480957,1340.4498291],[598.723571777,996.747131348,1260.19140625],[-65.6389694214,1065.03173828,1262.72827148],[466.533508301,1165.55529785,1040.90234375],[101.381401062,1196.74304199,1038.7208252]]

m=MoveArms()

def signal_handler(signal, frame):
	print("bye!")
	sys.exit(0)

def callback(data):
	shoulder_right = data[1]
	shoulder_left = data[2]
	torso = data[0]
	elbow_right = data[3]
	wrist_right = data[5]
	elbow_left = data[4]
	wrist_left = data[6]

	shoulderPitch_left_v0 = np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]]) - np.array([torso[0],torso[1],torso[2]])
	shoulderPitch_left_v1 = np.array([shoulder_left[0], shoulder_left[1], shoulder_left[2]]) - np.array([elbow_left[0], elbow_left[1], elbow_left[2]])
	
	dot_product = np.dot(shoulderPitch_left_v0,shoulderPitch_left_v1)
	mag_v0 = np.linalg.norm(shoulderPitch_left_v0)
	mag_v1 = np.linalg.norm(shoulderPitch_left_v1)

	shoulderPitch_left_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))

	shoulderPitch_right_v0 = np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]]) - np.array([torso[0],torso[1],torso[2]])
	shoulderPitch_right_v1 = np.array([shoulder_right[0], shoulder_right[1], shoulder_right[2]]) - np.array([elbow_right[0], elbow_right[1], elbow_right[2]])
	
	dot_product = np.dot(shoulderPitch_right_v0,shoulderPitch_right_v1)
	mag_v0 = np.linalg.norm(shoulderPitch_right_v0)
	mag_v1 = np.linalg.norm(shoulderPitch_right_v1)

	shoulderPitch_right_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))


	shoulderYaw_left_v0 = np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]]) - np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]])
	shoulderYaw_left_v1 = np.array([shoulder_left[0], shoulder_left[1], shoulder_left[2]]) - np.array([elbow_left[0], elbow_left[1], elbow_left[2]])

	dot_product = np.dot(shoulderYaw_left_v0,shoulderYaw_left_v1)
	mag_v0 = np.linalg.norm(shoulderYaw_left_v0)
	mag_v1 = np.linalg.norm(shoulderYaw_left_v1)

	shoulderYaw_left_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))

	shoulderYaw_right_v0 = np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]]) - np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]])
	shoulderYaw_right_v1 = np.array([shoulder_right[0], shoulder_right[1], shoulder_right[2]]) - np.array([elbow_right[0], elbow_right[1], elbow_right[2]])

	dot_product = np.dot(shoulderYaw_right_v0,shoulderYaw_right_v1)
	mag_v0 = np.linalg.norm(shoulderYaw_right_v0)
	mag_v1 = np.linalg.norm(shoulderYaw_right_v1)

	shoulderYaw_right_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))


	elbowPitch_left_v0 = np.array([elbow_left[0],elbow_left[1],elbow_left[2]]) - np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]])
	elbowPitch_left_v1 = np.array([elbow_left[0], elbow_left[1], elbow_left[2]]) - np.array([wrist_left[0], wrist_left[1], wrist_left[2]])

	dot_product = np.dot(elbowPitch_left_v0,elbowPitch_left_v1)
	mag_v0 = np.linalg.norm(elbowPitch_left_v0)
	mag_v1 = np.linalg.norm(elbowPitch_left_v1)

	elbowPitch_left_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))
	
	elbowPitch_right_v0 = np.array([elbow_right[0],elbow_right[1],elbow_right[2]]) - np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]])
	elbowPitch_right_v1 = np.array([elbow_right[0], elbow_right[1], elbow_right[2]]) - np.array([wrist_right[0], wrist_right[1], wrist_right[2]])

	dot_product = np.dot(elbowPitch_right_v0,elbowPitch_right_v1)
	mag_v0 = np.linalg.norm(elbowPitch_right_v0)
	mag_v1 = np.linalg.norm(elbowPitch_right_v1)

	elbowPitch_right_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))
	

	shoulderRoll_left_v0_1 = np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]]) - np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]])
	shoulderRoll_left_v1_1 = np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]]) - np.array([elbow_left[0],elbow_left[1],elbow_left[2]])
	
	shoulderRoll_left_v0_2 = np.array([elbow_left[0], elbow_left[1], elbow_left[2]]) - np.array([wrist_left[0], wrist_left[1], wrist_left[2]])
	shoulderRoll_left_v1_2 = np.array([elbow_left[0], elbow_left[1], elbow_left[2]]) - np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]])
	
	EM = np.cross(shoulderRoll_left_v0_1, shoulderRoll_left_v1_1) #perpedicular to plane DEA
	AH = np.cross(shoulderRoll_left_v0_2, shoulderRoll_left_v1_2) #perpendicular to plane EAB

	dot_product = np.dot(EM, AH)
	mag_v0 = np.linalg.norm(EM)
	mag_v1 = np.linalg.norm(AH)

	shoulderRoll_left_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))

	shoulderRoll_right_v0_1 = np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]]) - np.array([shoulder_left[0],shoulder_left[1],shoulder_left[2]])
	shoulderRoll_right_v1_1 = np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]]) - np.array([elbow_right[0],elbow_right[1],elbow_right[2]])
	
	shoulderRoll_right_v0_2 = np.array([elbow_right[0], elbow_right[1], elbow_right[2]]) - np.array([wrist_right[0], wrist_right[1], wrist_right[2]])
	shoulderRoll_right_v1_2 = np.array([elbow_right[0], elbow_right[1], elbow_right[2]]) - np.array([shoulder_right[0],shoulder_right[1],shoulder_right[2]])
	
	EM = np.cross(shoulderRoll_right_v0_1, shoulderRoll_right_v1_1) #perpedicular to plane DEA
	AH = np.cross(shoulderRoll_right_v0_2, shoulderRoll_right_v1_2) #perpendicular to plane EAB

	dot_product = np.dot(EM, AH)
	mag_v0 = np.linalg.norm(EM)
	mag_v1 = np.linalg.norm(AH)

	shoulderRoll_right_angle = (180/math.pi)*np.arccos(1.0*(dot_product)/(mag_v0*mag_v1))
	


	print(torso,shoulder_right,shoulder_left,elbow_right,elbow_left,wrist_right,wrist_left)
	print(shoulderPitch_left_angle,shoulderYaw_left_angle,shoulderRoll_left_angle,elbowPitch_left_angle)
	print(shoulderPitch_right_angle,shoulderYaw_right_angle,shoulderRoll_left_angle,elbowPitch_right_angle)

	global m
	m.setLeftArmJointAngles([shoulderPitch_left_angle,shoulderYaw_left_angle,shoulderRoll_left_angle,elbowPitch_left_angle,0,0,0])
	m.setRightArmJointAngles([shoulderPitch_right_angle,shoulderYaw_right_angle,shoulderRoll_left_angle,elbowPitch_right_angle,0,0,0])

def main():

	while True:
		callback(positions)
		callback(positions_next)

if __name__ == '__main__':
	main()
