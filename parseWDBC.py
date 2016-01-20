import os
import numpy as np

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

from test import *

testx = []
testy = []

testx.append(x.pop(20))
testy.append(y.pop(20))

xo = np.array(x, dtype=float)
yo = np.array(y, dtype=float)
x = xo/np.amax(xo, axis=0)
y = yo/np.amax(yo, axis=0)

nn = Neural_Network()
t = Trainer(nn)
t.train(x, y)

testx = np.array(testx, dtype=float)/np.amax(xo, axis=0)
testy = np.array(testy, dtype=float)/np.amax(yo, axis=0)

print nn.forward(testx)
print testy