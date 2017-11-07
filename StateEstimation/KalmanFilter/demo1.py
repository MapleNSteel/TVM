import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
from KF import KalmanFilter

def main():

	deltaTime=0.001
	numSteps=500

	x_0=np.array([[0],[0]])
	F=np.array([[1, 0],[0, 1]])*0
	B=np.array([[1, 0],[0, 1]])
	H=np.array([[1, 0],[0, 1]])
	P=np.array([[0, 0],[0, 0]])
	
	p_sigma=1e-3
	o_sigma=1e-3

	Q=np.array([[1, 0],[0, 1]])*(p_sigma**2)
	R=np.array([[1, 0],[0, 1]])*(o_sigma**2)

	t=(np.arange(0,numSteps)*0.01)
	i=np.zeros((2,numSteps))
	i[0,0:]=np.sin(2*np.pi*1*t)*1

	p_n=np.reshape(np.random.multivariate_normal([0, 0],[[p_sigma,0],[0,p_sigma]],numSteps),(2,numSteps))
	o_n=np.reshape(np.random.multivariate_normal([0, 0],[[o_sigma,0],[0,o_sigma]],numSteps),(2,numSteps))

	x=np.zeros((2,numSteps))
	y=np.zeros((2,numSteps))
	x_pred=np.zeros((2,numSteps))

	KF=KalmanFilter(F,B,Q,H,R,P,x[0:,0])

	for t in range(1,numSteps):
		x[0:,t]=F.dot(x[0:,t-1])+B.dot(i[0:,t])+p_n[0:,t]
		KF.predict(i[0:,t])
		x_pred[0:,t],P=KF.getPrediction()
		#print(P)
		y[0:,t]=H.dot(x[0:,t])+o_n[0:,t]
		KF.update(y[0:,t])
	
	print(np.sum((y-x_pred)**2)/numSteps)
	print(np.sum((x-x_pred)**2)/numSteps)

	plt.plot(x[0,0:],'r',y[0,0:],'b', x_pred[0,0:],'g')
	plt.show()

if __name__ == '__main__':
	main()
