from joblib.numpy_pickle_utils import xrange
from numpy import *


class NeuralNet(object):
	def __init__(self):
		# Generate random numbers
		random.seed(1)


		self.synaptic_weights = 2 * random.random((10, 1)) - 1

	# The Sigmoid function
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# Train the neural network and adjust the weights each time.
	def train(self, inputs, outputs, training_iterations):
		for iteration in xrange(training_iterations):
			# Pass the training set through the network.
			output = self.learn(inputs)

			# Calculate the error
			error = outputs - output

			# Adjust the weights by a factor
			factor = dot(inputs.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += factor

		# The neural network thinks.

	def bitTransform(self, x):
		i = 2
		arr2 = []
		for i in range(10):
			if (1 << i) & x:
				arr2.append(1)
			else:
				arr2.append(0)
		return arr2

	def bitToNumber(self, arr):
	    x = 0
	    for i in range(len(arr)):
	        x += (arr[i] << i)
	    return x


	def learn(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

arr = [1, 2, 3]
if __name__ == "__main__":
	# Initialize
	neural_network = NeuralNet()
	x = 10
	arr = [61,60,63,63,63,61,62,60,60,60,59,59,60,62,60,57,60,60,61,62,61]
	for i in range(len(arr)):
	    arr[i] *= 10



	# The training set.
	#print(neural_network.bitTransform(x))
	i = 0
	inputs = []
	outputs = []
	i = 0
	fileObj = open("prediction.txt", "r")
	arr = fileObj.read().splitlines()[0].split(" ")
	for i in range(len(arr)):
	    arr[i] = float(arr[i])
	#for i in range(len(ara)):
	#    ara[i] = float(ara[i])

	print(str(arr))
	for i in range(len(arr)):
	    arr[i] = int(arr[i] * 10)
	i = 0




	while i < len(arr):
		inputs.append(neural_network.bitTransform(arr[i]))
		i += 1
	#print(inputs)

	for i in range(len(arr)):
		outputs.append(1)


	#print(arr)



	neural_network.train(array(inputs), array([outputs]).T, 10000)

	for i in range(10):
	    x = arr[len(arr) - random.randint(2, 5)]

	    ok = True
	    while ok == True:
	        newNumber = x + random.randint(-50, 100)
	        y = neural_network.learn(neural_network.bitTransform(newNumber))

	        if y[0] > 0.95:
	            ok = False
	            #print(1)
	            arr.append(newNumber)
	            inputs.append(neural_network.bitTransform(newNumber))
	    inputs.append(neural_network.bitTransform(x))
	    outputs.append(1)
	for i in range(len(arr)):
	    arr[i] /= 10




	# Test the neural network with a test example.
	#print(neural_network.learn(array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])))
	#print(neural_network.bitToNumber(inputs[0]))#print(neural_network.learn(array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])))

	print(arr)
	file = open("predictionOut.txt", "w+")
	for i in range(len(arr)):
		file.write(str(arr[i]) + " ")


