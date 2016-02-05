from django.shortcuts import render
from django.http import HttpResponse
from .imageParser import *
from .digitParser import *
import numpy as np

def main(request):
	NN = Neural_Network(16, Lambda=0.0, hLayer = 28)
	cost = networkTrain(NN)
	request.session['params'] = list(NN.getParams())
	return render(request, "neuralNet/main.html")

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
	return HttpResponse(str(val)+","+str(max(result)))