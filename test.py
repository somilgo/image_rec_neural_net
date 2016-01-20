import numpy as np
ite = 0
class Neural_Network(object):
	def __init__(self, Lambda=0):        
		#Define Hyperparameters
		self.inputLayerSize = 30
		self.outputLayerSize = 1
		self.hiddenLayerSize = 10
		self.Lambda=Lambda
		#Weights (parameters)
		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
		
	def forward(self, X):
		#Propogate inputs though network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat
		
	
	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoidPrime(self, z):
		return (np.exp(-z) / ((1.0+np.exp(-z))**2))

	
	def costFunction(self, X, y):
		#Compute cost for given X,y, use weights already stored in class.
		self.yHat = self.forward(X)
		J = 0.5*sum((y-self.yHat)**2)
		regularize = (self.Lambda/2.0/len(X)) * (sum(sum(self.W1**2)) + sum(sum(self.W2**2)))
		return J + regularize
		
	def costFunctionPrime(self, X, y):
		#Compute derivative with respect to W and W2 for a given X and y:
		self.yHat = self.forward(X)
		
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)
		
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)  
		
		return dJdW1, dJdW2
	
	#Helper Functions for interacting with other classes:
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
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
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
			loss2 = N.costFunction(X, y)
			
			N.setParams(paramsInitial - perturb)
			loss1 = N.costFunction(X, y)

			#Compute Numerical Gradient
			numgrad[p] = (loss2 - loss1) / (2*e)

			#Return the value we changed to zero:
			perturb[p] = 0
			
		#Return Params to original value:
		N.setParams(paramsInitial)

		return numgrad 

from scipy import optimize


class Trainer(object):
	def __init__(self, N):
		#Make Local reference to network:
		self.N = N
		
	def callbackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.y))   
		
	def costFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X,y)
		return cost, grad
		
	def train(self, X, y):
		#Make an internal variable for the callback function:
		self.X = X
		self.y = y

		#Make empty list to store costs:
		self.J = []
		
		params0 = self.N.getParams()

		options = {'maxiter': 200, 'disp' : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='CG', \
								 args=(X, y), options=options, callback=self.callbackF)

		self.N.setParams(_res.x)
		self.optimizationResults = _res