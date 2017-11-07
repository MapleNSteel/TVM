import rospy
import numpy as np
# baxter_interface - Baxter Python API
import baxter_interface
from baxter_interface import CHECK_VERSION

import sys, signal

def signal_handler(signal, frame):
	print("bye!")

	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def symmetryMap(angles,arm):
	if(arm=='left'):
		return [-angles[0],angles[1],-angles[2],angles[3],-angles[4],angles[5],angles[6]]
	else:
		return [angles[0],angles[1],-angles[2],angles[3],-angles[4],angles[5],angles[6]]

class MoveArms():
	def __init__(self):
		# create an instance of baxter_interface's Limb class
		self.left_limb = baxter_interface.Limb('left')
		# get the left limb's current joint left_angles
		self.left_angles = self.left_limb.joint_angles()

		# create an instance of baxter_interface's Limb class
		self.right_limb = baxter_interface.Limb('right')
		# get the right limb's current joint right_angles
		self.right_angles = self.right_limb.joint_angles()
	# initialize our ROS node, registering it with the Master
	def setLeftArmJointAngles(self,joints,angles):
		# reassign new joint left_angles (all zeros) which we will later command to the limb
		joint_command = {joints[i]: angles[i] for i in range(0,len(joints))}
		# move the left arm to those joint right_angles
		self.left_limb.set_joint_positions(joint_command)

	def setRightArmJointAngles(self,joints,angles):
		# reassign new joint right_angles (all zeros) which we will later command to the limb
		joint_command = {joints[i]: angles[i] for i in range(0,len(joints))}
		# move the right arm to those joint right_angles
		self.right_limb.set_joint_positions(joint_command)
	def moveToLeftArmJointAngles(self,joints,angles):
		# reassign new joint right_angles (all zeros) which we will later command to the limb
		joint_command = {joints[i]: angles[i] for i in range(0,len(joints))}
		# move the right arm to those joint right_angles
		self.left_limb.set_joint_positions(joint_command)

	def moveToRightArmJointAngles(self,joints,angles):
		# reassign new joint right_angles (all zeros) which we will later command to the limb
		joint_command = {joints[i]: angles[i] for i in range(0,len(joints))}
		# move the right arm to those joint right_angles
		self.right_limb.set_joint_positions(joint_command)

	def setLeftArmJointVelocities(self,joints,angles):
		# reassign new joint right_angles (all zeros) which we will later command to the limb
		joint_command = {joints[i]: angles[i] for i in range(0,len(joints))}
		# move the left arm to those joint right_angles
		self.left_limb.set_joint_velocities(joint_command)

	def setRightArmJointVelocities(self,joints,angles):
		# reassign new joint right_angles (all zeros) which we will later command to the limb
		joint_command = {joints[i]: angles[i] for i in range(0,len(joints))}
		# move the left arm to those joint right_angles
		self.right_limb.set_joint_velocities(joint_command)

left_default = [0.7781117546548761, 1.0465583925348236, -1.2996652225359169, 1.6114468176736272, 1.1984224905354794, 1.419315723990979, -0.1112136071216925]
right_default = symmetryMap(left_default,'left')

def main():

	rospy.init_node('Hello_Baxter')
	print "left:"+str(left_default)
	print "right:"+str(right_default)

	m=MoveArms()

	m.moveToLeftArmJointAngles(m.left_limb.joint_names(),left_default)
	m.moveToRightArmJointAngles(m.right_limb.joint_names(),right_default)


if __name__ == '__main__':
	main()
