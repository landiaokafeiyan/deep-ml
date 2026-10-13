import math

def sigmoid(z: float) -> float:
	#Your code here
	exp_z=math.exp(-z)
	result=round(1/(exp_z+1),4)
	return result