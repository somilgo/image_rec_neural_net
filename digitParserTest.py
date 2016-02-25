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
	for i in os.walk("/home/somil/Documents/SRC-NN/images"):
		if dataDir:
			dataDirs.append(i[0])
		dataDir = True
	for i in dataDirs:
		for root, dirs, files in os.walk(i):
			for f in files:
				if f.endswith(".txt"):
					cf = open(i+'/'+f, 'r')
					data = map(int, (cf.read().replace('[', '').replace(',', '').replace('-1', '0').replace(']', '').split(' ')))
					x.append(data)
					result = int(i[len(i)-1])
					resultList = [0]*10
					resultList[result] = 1
					y.append(resultList)
	return np.array((x), dtype=float), np.array((y), dtype=float), x, y

#Uses parsed data to train the neural network
def networkTrain(NN, fr, ttx, tty, tx, ty, tester = False):
	T = Trainer(NN)
	data = parseData()
	xdata = data[2]
	ydata = data[3]
	if tester:
		xdata = list(tx)
		ydata = list(ty)
	picker = range(len(xdata))
	#Randomizes order of data list indexes
	shuffle(picker)
	#Chooses 80% of the data as the training set
	trainset = int(len(xdata)*fr)
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

	T.train(trainx, trainy, testx, testy)

	numberCorrect = 0
	total = 0
	if tester:
		for i in range(len(ttx)):
			w = NN.forward(ttx[i])
			d = tty[i]
			if list(w).index(max(w))==list(d).index(max(d)):
				numberCorrect+=1
			total+=1

		# print numberCorrect
		# print total
		# print float(numberCorrect)/total * 100

		print float(numberCorrect)/total*100
	else: total = 1
	return float(numberCorrect)/total * 100, len(trainx), testx, testy,trainx, trainy


perclist = []
sampleamt = []
nn = Neural_Network(iLayer=256, oLayer=10, hLayer=28)
g = networkTrain(nn, .75, 3, 3, 3, 3)
ttx = g[2]
tty = g[3]
tx = g[4]
ty = g[5]

for f in range(1, 100):
	nn = Neural_Network(iLayer=256, oLayer=10, hLayer=28)
	fr = float(f)/100.0
	g = networkTrain(nn, fr, ttx, tty, tx, ty, tester=True)
	perclist.append(g[0])
	sampleamt.append(g[1])

plt.scatter(sampleamt, perclist)
plt.ylabel('Percent Accurate')
plt.xlabel('Training Sample Size')
plt.show()

print perclist
print sampleamt