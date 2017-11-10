import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import sys, signal
import copy
from KalmanFilter import KF

import cPickle as pickle

import rospy
from sensor_msgs.msg import JointState

def signal_handler(signal, frame):
	print("bye!")
	#pickle.dump( jointStates, open( "jointStates.dat", "wb" ) )
	#pickle.dump( jointStatesEstimation, open( "jointStatesEstimation.dat", "wb" ) )

	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

jointStates=[]
num=0
numSteps=10

deltaTime=0.01

mean=np.array([  0.00000000e+00,  -2.80933241e-02,  -1.29741334e+00,   1.60955082e+00,
7.75902439e-01,   1.04444955e+00,   1.19985868e+00,   1.41754781e+00,
-1.67318954e-03,   1.29655278e+00,   1.60888890e+00,  -7.75708390e-01,
1.04309351e+00,  -1.19567091e+00,   1.41733075e+00,  -1.09324126e-01,
-1.25659871e+01, 0.00000000e+00,  -2.80933241e-02,  -1.29741334e+00,   1.60955082e+00,
7.75902439e-01,   1.04444955e+00,   1.19985868e+00,   1.41754781e+00,
-1.67318954e-03,   1.29655278e+00,   1.60888890e+00,  -7.75708390e-01,
1.04309351e+00,  -1.19567091e+00,   1.41733075e+00,  -1.09324126e-01,
-1.25659871e+01])

std=np.array([  3.08341586e-04,   3.08341586e-04,   5.89837056e-04,   4.19899047e-04,
3.76895794e-04,   4.11610059e-04,   3.95200127e-04,   3.50620328e-04,
1.28211518e-03,   6.10533989e-04,   1.60689064e-03,   3.87635875e-04,
4.06536564e-04,   4.50566057e-04,   4.07123819e-04,   4.19339504e-04,
1.75859327e-13, 3.08341586e-04,   3.08341586e-04,   5.89837056e-04,   4.19899047e-04,
3.76895794e-04,   4.11610059e-04,   3.95200127e-04,   3.50620328e-04,
1.28211518e-03,   6.10533989e-04,   1.60689064e-03,   3.87635875e-04,
4.06536564e-04,   4.50566057e-04,   4.07123819e-04,   4.19339504e-04,
1.75859327e-13])

F=np.eye(34)
F[0:17,17:]=np.eye(17)*deltaTime
B=np.eye(34)
H=np.eye(34)
P=np.zeros((34,34))

p_sigma=std
o_sigma=std

Q=np.diag(p_sigma**2)
R=np.diag(o_sigma**2)

jointStatesEstimation=[]

KF=KF.KalmanFilter(F,B,Q,H,R,P,mean)

def callback(data):
	#print("I heard "+str(np.array(np.concatenate((data.position,data.velocity)))))
	#jointStates.append(data)

	global num
	num+=1

	global KF
	KF.predict(np.zeros(34))
	[pos,P_temp]=KF.getPrediction()

	data_pred=copy.deepcopy(data)
	data_pred.position=pos[0:17]
	data_pred.velocity=pos[17:]

	#print("I think it's"+str(pos))

	#jointStatesEstimation.append(data_pred)#predicting next process state
	KF.update(np.concatenate((data.position,data.velocity)))#update Kalman


	pub = rospy.Publisher('/robot/kalman_joint_state',JointState, queue_size=1)
	pub.publish(data_pred)
	
def listener():

	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber("/robot/joint_states", JointState, callback)
	# spin() simply keeps python from exiting until this node is stopped

def main():

	listener()

	while(True):
		pass

if __name__ == '__main__':
	main()
