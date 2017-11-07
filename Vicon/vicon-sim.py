import sys, signal

from vicon_bridge.msg import Markers

import cPickle as pickle

import rospy

viconData=pickle.load(open("vicon.dat","rb"))

def signal_handler(signal, frame):
	print("bye!")
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

rospy.init_node('vicon')
rate = rospy.Rate(100)
pub = rospy.Publisher('/vicon/markers', Markers, queue_size=1)
while True:
	for data in viconData:
		pub.publish(data)
		rate.sleep()