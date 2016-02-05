import numpy as np
import sys
import os
from random import shuffle
from network import *

#Collects and parses digit image data from images folder
def parseData():
	x = []
	y = []
	dataDir = False
	dataDirs = []
	for i in os.walk("/home/somil/Documents/SRC-NN/NeuralWeb/neuralNet/static/imageData/"):
		if dataDir:
			dataDirs.append(i[0])
		dataDir = True
	for i in dataDirs:
		for root, dirs, files in os.walk(i):
			for f in files:
				if f.endswith(".txt"):
					cf = open(i+'/'+f, 'r')
					data = map(int, (cf.read().replace('[', '').replace(',', '').replace(']', '').split(' ')))
					x.append(data)
					result = int(i[len(i)-1])
					resultList = [0]*10
					resultList[result] = 1
					y.append(resultList)
	return np.array((x), dtype=float), np.array((y), dtype=float), x, y

#Uses parsed data to train the neural network
def networkTrain(NN):
	T = Trainer(NN)
	data = parseData()
	xdata = data[2]
	ydata = data[3]
	picker = range(len(xdata))
	#Randomizes order of data list indexes
	shuffle(picker)
	#Chooses 80% of the data as the training set
	trainset = int(len(xdata)*.8)
	pickersplit=[picker[x:x+trainset] for x in xrange(0, len(picker), trainset)]
	trainx = []
	testx = []
	trainy = []
	testy = []
	for p in pickersplit[0]:
		trainx.append(xdata[p])
		trainy.append(ydata[p])
	for t in pickersplit[1]:
		testx.append(xdata[t])
		testy.append(ydata[t])
	#Creates training sets and testing sets
	trainx = np.array(trainx, dtype=float)
	trainy = np.array(trainy, dtype=float)
	testx = np.array(testx, dtype=float)
	testy = np.array(testy, dtype=float)
	return T.train(trainx, trainy, testx, testy)

print os.getcwd()+"/images/"
print parseData()