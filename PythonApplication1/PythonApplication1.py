import csv
import numpy as np     # installed with matplotlib

def GetDATA(location):
	data = list(csv.reader(open(location, 'r')))
	dat = []

	for i in range(1, len(data)): # S. No. removed
		dat.append(data[i][1:])
	return np.array(dat).astype(np.float)

class NeuralNet:
	def __init__(self, list):
		self.weights = []
		self.bias = []
		for (index, val) in enumerate(list[0:len(list) - 1]):
			self.weights.append([[0 for col in range(list[index])] for row in range(list[index+1])])
			self.bias.append([0 for col in range(list[index + 1])])

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoid_prime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	def cost_derivative(self, output_activations, y):
		if y == 1:
			return [output_activations[0] - y, output_activations[1], output_activations[2]]
		elif y == 2:
			return [output_activations[0], output_activations[1] - y, output_activations[2]]
		else:
			return [output_activations[0], output_activations[1], output_activations[2] - y]

	def BackProp(self, x, y):
		# delta arrays
		del_w = [[[0 for col in range(len(self.weights[dim][0]))] for row in range(len(net.weights[dim]))] for dim in range(len(self.weights))]
		del_b = [[0 for row in range(len(self.bias[dim]))] for dim in range(len(self.bias))]

		# Forward Propogation
		a = []
		z = []
		a.append(x)
		z.append(x) # Non Usable Value
		
		## print("{0} Layer : {1}".format(0, a[-1]))
		for i in range(1, 5):  # for 1,2,3,4
			z.append(np.dot(self.weights[i - 1], a[i - 1]) + self.bias[i - 1])
			a.append(self.sigmoid(z[-1]))
			## print("{0} Layer : {1}".format(i, a[-1]))

		# Intermediate Steps
		delta = []
		print(z)
		print(a)
		delta.insert(0, np.multiply(self.cost_derivative(a[4], y), self.sigmoid_prime(z[4]))) ## 4
		delta.insert(0, np.multiply(np.dot(np.transpose(self.weights[3]), delta[0]), self.sigmoid_prime(z[3]))) ## 3
		print(delta)
		
		return (x, y)

net = NeuralNet([4,10,10,10,3])
data = GetDATA('iris.data')
net.BackProp(data[1][0 : 4], data[1][-1])