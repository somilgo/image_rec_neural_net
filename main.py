import pygame
import sys
import os
from pygame.locals import *
from PIL import Image
from network import *
from random import shuffle

pygame.init()

#Pygame window options
position = [400, 25]
os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
FPS = 35
fpsClock = pygame.time.Clock()
width = 700
height = 400
surface = pygame.display.set_mode((width, height))
pygame.display.set_caption('Train Handwriting')

def clamp(value, maximum, minimum):
	if value > maximum:
		return maximum
	elif value < minimum:
		return minimum
	else:
		return value

def checkWithinRect(point, rad, rect, rectw, recth):
	closestX = clamp(point[0], rect[0], rect[0]+rectw)
	closestY = clamp(point[1], rect[1], rect[1]+recth)
	distanceX = point[0] - closestX
	distanceY = point[1] - closestY
	d_squared = distanceY**2 + distanceX**2
	if point[0] >= rect[0] and point[0] <= (rect[0]+rectw) and point[1] >= rect[1] and point[1] <= (rect[1]+recth):
		return True
	elif d_squared < rad**2:
		return True
	else:
		return False

def parseData():
	x = []
	y = []
	dataDir = False
	dataDirs = []
	for i in os.walk(os.getcwd()+"/images"):
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

def networkTrain(NN):
	T = Trainer(NN)
	data = parseData()
	xdata = data[2]
	ydata = data[3]
	# for i in xdata:
	# 	for index, item in enumerate(i):
	# 		if item == -1:
	# 			i[index] = 0
	picker = range(len(xdata))
	shuffle(picker)
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
	trainx = np.array(trainx, dtype=float)
	trainy = np.array(trainy, dtype=float)
	testx = np.array(testx, dtype=float)
	testy = np.array(testy, dtype=float)
	return T.train(trainx, trainy, testx, testy)

windowLoop = True
BLACK = (0,0,0)
WHITE = (255,255,255)

start_pos = []
end_pos = []
gridSize = 5
pos = 0
recH = 400
recW = 400

p = lambda x: x*(gridSize)
gridRects = []

for x in range(recW/gridSize):
	for y in range(recH/gridSize):
		gridRects.append([p(x), p(y), WHITE])

clock = pygame.time.Clock()
pressed = 0
start = (0,0)
lines = []

fontsm = pygame.font.SysFont("Arial", 40)
text = fontsm.render("Training Network...", True, (255, 255, 255))
surface.blit(text, (surface.get_rect().center[0]-(text.get_rect().center[0]),200))
pygame.display.flip()

canvas = pygame.Rect(0,0,recW,recH)
panel = pygame.Rect((401,0,300,400))
cansub = surface.subsurface(canvas)
pygame.draw.rect(surface, WHITE, canvas)
save = False
brush = pygame.image.load("brush.png")
brush = pygame.transform.scale(brush, (20,20))
surface.fill(BLACK, rect=panel)
# for i in gridRects:
# 		pygame.draw.rect(surface, i[2], pygame.Rect(i[0], i[1], gridSize, gridSize))


# minCost = 1e9
# minH = 0

# costs = {}
# for c in range(1,31):
# 	costs[c]=0


# for i in range(30):
# 	for h in range(1,31):
# 		NN = Neural_Network(16, Lambda=0, hLayer = h)
# 		cost = networkTrain(NN)
# 		costs[h]= cost+costs[h]
# for c in costs:
# 	costs[c] = costs[c]/30.0

# for c in costs:
# 	if costs[c]<minCost:
# 		minCost = costs[c]
# 		minH = c

# print costs
# print minH
# print minCost

NN = Neural_Network(16, Lambda=0.0, hLayer = 23)
cost = networkTrain(NN)
pressed = False


#Main loop
while windowLoop:

	if pygame.mouse.get_pressed()[0] == 1:
		pressed = True
		mosPos = pygame.mouse.get_pos()
		cansub.blit(brush, (mosPos[0]-10, mosPos[1]-10))
		for g in gridRects:
			if checkWithinRect(mosPos, 12, (g[0], g[1]), gridSize, gridSize):
				g[2] = BLACK 
		if start != (0,0):
			pygame.draw.line(cansub, BLACK, start, mosPos, 24)
		start = mosPos
	else:
		start = (0,0)
	mosPos = pygame.mouse.get_pos()
	if mosPos[0] < 400:
		mosCan = True
	else:
		mosCan = False
	if pygame.mouse.get_pressed()[0] == 0 and pressed and mosCan:
		pressed = False

		pygame.image.save(cansub, os.getcwd() + "/temp.jpg")
		test = Image.open(os.getcwd()+"/temp.jpg")
		imageSize = 400
		test = test.resize((imageSize,imageSize), Image.ANTIALIAS).convert('1')
		pix = list(test.getdata())
		if 0 in pix:

			left = 1e9
			right = -1
			upper = 1e9
			lower = -1
			colCount = 0
			rowCount = 0
			for p in pix:
				if colCount == imageSize:
					colCount = 0
					rowCount += 1
				if p == 0:
					if colCount > right:
						right = colCount
					if colCount < left:
						left = colCount
					if rowCount > lower:
						lower = rowCount
					if rowCount < upper:
						upper = rowCount
				colCount += 1
			test = test.crop((left, upper, right, lower))
			imageSize = 16
			test = test.resize((imageSize,imageSize), Image.ANTIALIAS).convert('1')
			pix = list(test.getdata())
			newpix = []
			for p in pix:
				if p == 255:
					newpix.append(-1)
				else:
					newpix.append(1)
			empt = []
			empt.append(newpix)
			surface.fill(BLACK, rect=panel)
			jawn = np.array(empt, dtype=float)
			result = list(NN.forward(jawn)[0])
			val = str(result.index(max(result)))
			font = pygame.font.SysFont("Arial", 70)
			fontsm = pygame.font.SysFont("Arial", 40)
			text = fontsm.render("This digit is a:", True, (255, 255, 255))
			surface.blit(text, (panel.center[0]-(text.get_rect().center[0]),20))
			text = font.render(val, True, (255,255,255))
			surface.blit(text, (panel.center[0]-(text.get_rect().center[0]),90))
			percent = max(result)*100
			percent = "%.2f" % percent
			fontsml = pygame.font.SysFont("Arial", 30)
			text = fontsml.render("With a certainty of:", True, (255, 255, 255))
			surface.blit(text, (panel.center[0]-(text.get_rect().center[0]),200))
			text = font.render(str(percent)+"%", True, (255,255,255))
			surface.blit(text, (panel.center[0]-(text.get_rect().center[0]),250))


	if save:
		panel = pygame.Rect((401,0,300,400))
		surface.fill(BLACK, rect=panel)
		r = lambda x: x*75 + 25
		count = 0
		buttons = {}
		for i in range(5):
			buttons[str(count)] = pygame.Rect(433, r(i), 100, 50)
			pygame.draw.rect(surface, WHITE, buttons[str(count)])
			count+=1
			buttons[str(count)] = pygame.Rect(566, r(i), 100, 50)
			pygame.draw.rect(surface, WHITE, buttons[str(count)])
			count+=1
		font = pygame.font.SysFont("Arial", 35)
		button = [473, 606]
		for i in range(10):
			text = font.render(str(i), True, (0, 0, 0))
			val = (i+1)%2
			if val:
				surface.blit(text, (button[0],5+r(int(i/2))))
			else:
				surface.blit(text, (button[1],5+r(int(i/2))))
		if pygame.mouse.get_pressed()[0] == 1:
			mosPos = pygame.mouse.get_pos()
			for b in buttons:
				if buttons[b].collidepoint(mosPos):
					maxnum = 0
					for f in os.listdir(os.getcwd()+"/images/" + b):
						if int(f[:-4]) > maxnum:
							maxnum = int(f[:-4])
					maxnum+=1
					pressed = False
					mosCan = False
					
					pygame.image.save(cansub, os.getcwd()+"/images/" + b + "/"+str(maxnum)+".jpg")
					image = Image.open(os.getcwd()+"/images/" + b + "/"+str(maxnum)+".jpg")
					imageSize = 400
					image = image.resize((imageSize,imageSize), Image.ANTIALIAS).convert('1')
					filename = os.getcwd()+"/images/" + b + "/"+str(maxnum)+".txt"
					f = open(filename, 'w')
					pix = list(image.getdata())
					left = 1e9
					right = -1
					upper = 1e9
					lower = -1
					colCount = 0
					rowCount = 0
					for p in pix:
						if colCount == imageSize:
							colCount = 0
							rowCount += 1
						if p == 0:
							if colCount > right:
								right = colCount
							if colCount < left:
								left = colCount
							if rowCount > lower:
								lower = rowCount
							if rowCount < upper:
								upper = rowCount
						colCount += 1
					image = image.crop((left, upper, right, lower))
					imageSize = 16
					image = image.resize((imageSize,imageSize), Image.ANTIALIAS).convert('1')
					pix = list(image.getdata())
					newpix = []
					for p in pix:
						if p == 255:
							newpix.append(-1)
						else:
							newpix.append(1)
					f.write(str(newpix))
					f.close()
					image.save(os.getcwd()+"/images/" + b + "/"+str(maxnum)+".jpg")
					save = False
					surface.fill(BLACK)
					cansub.fill(WHITE)

	#Use escape or "X" to quit the window
	for event in pygame.event.get():
                key = pygame.key.get_pressed()
                if key[K_ESCAPE]:
                    pygame.quit()
                    sys.exit()
                    break
                if key[K_s]:
                	if save:
                		save = False
                		surface.subsurface(pygame.Rect(400,0,300,400)).fill(BLACK)
                	else:
                		save = True
                if key[K_c]:
                	cansub.fill(WHITE)
                	surface.fill(BLACK, rect=panel)
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                    break 
	pygame.display.flip()
	clock.tick(FPS)