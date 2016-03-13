import numpy as np
import sys
import os
from random import shuffle
from network import *

#Collects and parses digit image data from images folder
def parseData():
	x = []
	y = []
	f = open('optdigits.tra')
	line = f.readline()
	while (line != ''):
		data = map(int, (line.replace("\n", "").split(",")))
		ytoadd = [0]*10
		ytoadd[data.pop()] = 1
		y.append(ytoadd)
		x.append(data)
		line = f.readline()
	f.close()
	f = open('optdigits.tes')
	line = f.readline()
	xtest = []
	ytest = []
	while (line != ''):
		data = map(int, (line.replace("\n", "").split(",")))
		ytoadd = [0]*10
		ytoadd[data.pop()] = 1
		ytest.append(ytoadd)
		xtest.append(data)
		line = f.readline()
	return np.array((x), dtype=float), np.array((y), dtype=float), np.array((xtest), dtype=float), np.array((ytest), dtype=float)

#Uses parsed data to train the neural network
def networkTrain(NN):
	T = Trainer(NN)
	data = parseData()
	
	trainx = data[0]
	trainy = data[1]
	testx = data[2]
	testy = data[3]

	#Full training data
	# trainx = data[0]
	# trainy = data[1]
	# print len(trainx)
	T.train(trainx, trainy, testx, testy)

	numberCorrect = 0
	total = 0
	#Print out percent of test data that is accurate
	for i in range(len(testx)):
		w = NN.forward(testx[i])
		d = testy[i]
		if list(w).index(max(w))==list(d).index(max(d)):
			numberCorrect+=1
		total+=1

	print numberCorrect
	print total
	print float(numberCorrect)/total * 100

nn = Neural_Network(iLayer=64, hLayer=20, oLayer=10)
networkTrain(nn)