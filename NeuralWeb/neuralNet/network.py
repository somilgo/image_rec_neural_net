import numpy as np
from scipy import optimize
import copy
iteration = 0

class Neural_Network(object):
	#Intialize neural network object (requires length of square image)
	def __init__ (self, imageSize, hLayer=10, Lambda=0):
		self.inputLayerSize = imageSize**2
		self.outputLayerSize = 10
		#Set number of neurons in hidden layer to mean of input layer and output layer
		self.hiddenLayerSize = hLayer

		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		self.Lambda = Lambda

	#Sigmoid Activation function
	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	#Differentiated Sigmoid activation function
	def sigmoidPrime(self, z):
		return (np.exp(-z) / ((1.0+np.exp(-z))**2))

	def forward(self, x):
		self.z2 = np.dot(x, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	#Cross Entropy cost function
	def cost(self, x, y, outPut=False, test=False):
		self.yHat = self.forward(x)
		if outPut:
			print self.yHat
		J = (-1.0/len(x)) * sum(sum(y * np.log(self.yHat) + (1-y)*np.log(1-self.yHat)))
		#Regularizes cost function to prevent overfitting
		regularize = (self.Lambda/2.0/len(x)) * (sum(sum(self.W1**2)) + sum(sum(self.W2**2)))
		if test:
			regularize = 0
		return J + regularize

	#Compute derivative of cost function
	def costPrime(self, x, y):
		self.yHat = self.forward(x)

		#backError2 = (-y/float(len(x))/self.yHat) * self.sigmoidPrime(self.z3)
		backError2 = (y-self.yHat)/(-float(len(x)))
		dJdW2 = np.dot(self.a2.transpose(), backError2) + (self.Lambda*self.W2)/(len(x))

		backError1 = np.dot(backError2, self.W2.transpose()) * self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(x.transpose(), backError1) + (self.Lambda*sum(sum(self.W1)))/(len(x))
		return dJdW1, dJdW2

	#Squared differences cost function
	# def cost(self, x, y):
	# 	#Compute cost for given X,y, use weights already stored in class.
	# 	self.yHat = self.forward(x)
	# 	J = 0.5*sum(sum((y-self.yHat)**2))
	# 	return J
		
	# def costPrime(self, X, y):
	# 	#Compute derivative with respect to W and W2 for a given X and y:
	# 	self.yHat = self.forward(X)
		
	# 	delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
	# 	dJdW2 = np.dot(self.a2.T, delta3)
		
	# 	delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
	# 	dJdW1 = np.dot(X.T, delta2)  
		
	# 	return dJdW1, dJdW2

	def getParams(self):
		#Get W1 and W2 unrolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
	
	def setParams(self, params):
		#Set W1 and W2 using single paramater vector.
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
		
	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
	paramsInitial = N.getParams()
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	e = 1e-4

	for p in range(len(paramsInitial)):
		#Set perturbation vector
		perturb[p] = e
		N.setParams(paramsInitial + perturb)
		loss2 = N.cost(X, y)
		
		N.setParams(paramsInitial - perturb)
		loss1 = N.cost(X, y)

		#Compute Numerical Gradient
		numgrad[p] = (loss2 - loss1) / (2*e)

		#Return the value we changed to zero:
		perturb[p] = 0
		
	#Return Params to original value:
	N.setParams(paramsInitial)

	return numgrad 

class Trainer(object):
	def __init__(self, N):
		#Make Local reference to network:
		self.N = N
		
	def callbackF(self, params):
		#Set parameters and add cost data to lists for graphing
		self.N.setParams(params)
		self.J.append(self.N.cost(self.X, self.y, test=True))
		self.testJ.append(self.N.cost(self.testX, self.testY, test=True))
		
	def costFunctionWrapper(self, params, X, y):
		#Computes cost function and dJ/dW
		self.N.setParams(params)
		cost = self.N.cost(X, y)
		grad = self.N.computeGradients(X,y)		
		return cost, grad
		
	def train(self, trainX, trainY, testX, testY):
		#Make an internal variable for the callback function:
		self.X = trainX
		self.y = trainY
		
		self.testX = testX
		self.testY = testY

		#Make empty list to store training costs:
		self.J = []
		self.testJ = []
		
		params0 = self.N.getParams()

		options = {'maxiter': 100, 'disp' : True}

		#Minimize cost function using computed gradient method
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='CG',
								 args=(trainX, trainY), options=options, callback=self.callbackF)
		self.N.setParams(_res.x)
		self.optimizationResults = _res

		#Plot Cost vs Iterations graphs for testing and training data
		# plt.plot(self.J)
		# plt.plot(self.testJ)
		# plt.ylabel('Cost')
		# plt.xlabel('Iterations')
		# plt.legend(['Training', 'Test'], loc='upper left')
		# plt.show()

		#Returns average error for testing data
		return (sum(self.testJ))/(len(self.testJ))
