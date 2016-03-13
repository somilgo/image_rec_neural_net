import os
import numpy as np
from random import shuffle

dataFile = open(os.getcwd()+"/wdbc.data")
x = []
y = []
for line in dataFile:
	datum = line.split(",")
	datum[len(datum)-1]=datum[len(datum)-1].replace("\n", "")
	datum.pop(0)
	if datum.pop(0) == 'M':
		y.append([1])
	else:
		y.append([0])
	x.append(datum)

from network import *
xdata = x
ydata = y
picker = range(len(xdata))
#Randomizes order of data list indexes
shuffle(picker)
#Chooses 80% of the data as the training set
trainset = int(len(xdata)*.85)
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
x = testx
y = testy
trainx = np.array(trainx, dtype=float)
trainy = np.array(trainy, dtype=float)
testx = np.array(testx, dtype=float)
testy = np.array(testy, dtype=float)
trainx = trainx/np.amax(trainx, axis=0)
trainy = trainy/np.amax(trainy, axis=0)
testx = testx/np.amax(testx, axis=0)
testy = testy/np.amax(testy, axis=0)

nn = Neural_Network(hLayer=10, iLayer= 30, oLayer=1)
t = Trainer(nn, iterations=30)
t.train(trainx, trainy, testx, testy)

numberCorrect = 0
total = 0

for i in range(len(testx)):
	w = nn.forward(testx[i])
	d = testy[i] 
	if (abs(w[0]-d[0])) < .5:
		numberCorrect+=1
	total+=1

print "Number Correct: " + str(numberCorrect)
print "Testing sample size: " + str(total)
print "Percent Accuracy: " + str(float(numberCorrect)/total * 100)