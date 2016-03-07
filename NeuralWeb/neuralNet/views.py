from django.shortcuts import render
from django.http import HttpResponse
from .models import *
from .imageParser import *
from .digitParser import *
import numpy as np

def main(request):
	context = {"digits":range(10)}
	return render(request, "neuralNet/main.html", context)

def run_digit_network(request):
	NN = Neural_Network(16, Lambda=0.0, hLayer = 28)
	params = request.session['params']
	NN.setParams(np.asarray(params))
	rawInput = request.POST.get('imageData')
	base64 = rawInput.split(",")
	base64 = base64[1]
	pixels = getPixels(base64)
	collected_data = np.array(pixels, dtype=float)
	result = list(NN.forward(collected_data)[0])
	val = str(result.index(max(result)))
	prob = max(result)/sum(result)
	return HttpResponse(str(val)+","+str(prob))

def load_network(request):
	NN = Neural_Network(16, Lambda=0.0, hLayer = 28)
	cost = networkTrain(NN)
	request.session['params'] = list(NN.getParams())
	return HttpResponse("Done!")

def submit_digit(request):
	rawInput = request.POST.get('imageData')
	digit = int(request.POST.get('digit'))
	print digit
	base64 = rawInput.split(",")
	base64 = base64[1]
	pixels = (getPixels(base64))[0]
	pixMap = ""
	print pixels
	for i in pixels:
		pixMap += str(i)
		pixMap += ","
	pixMap = pixMap[:-1]
	y = ["0"]*10
	y[digit] = "1"
	digitval = ""
	for j in y:
		digitval+=str(j)
		digitval+=","
	digitval = digitval[:-1]
	print digitval, pixMap
	newDigit = DigitData.objects.create(digit=digitval, pixelMap=pixMap)
	newDigit.save()
	return HttpResponse("Done!")




#Only needed once to convert static digit data to database
# def createDigitData():
# 	localData = parseData()
# 	x = localData[2]
# 	y = localData[3]
# 	for x, y in zip(x,y):
# 		pixMap = ""
# 		for i in x:
# 			pixMap += str(i)
# 			pixMap += ","
# 		pixMap = pixMap[:-1]
# 		digitval = ""
# 		for j in y:
# 			digitval+=str(j)
# 			digitval+=","
# 		digitval = digitval[:-1]
# 		newDigit = DigitData.objects.create(digit=digitval, pixelMap=pixMap)
# 		newDigit.save()