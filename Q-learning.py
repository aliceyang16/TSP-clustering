import numpy as np
import math
import matplotlib.pyplot as plt
import random
import csv
import math

#from numpy.random import seed
#from numpy.random import randint
#seed(0)
#print(randint(0, 10, 20))

class Agent:
	def __init__(self, state, toVisit):
		self.state = state
		self.visited = [state]
		self.toVisit = toVisit
		self.toVisit.remove(state)

	def randomMovement(self):
		nextCity = random.choice(self.toVisit)
		self.state = nextCity
		self.visited.append(nextCity)
		self.toVisit.remove(nextCity)

	def calculateDistanceTravelled(self, distanceMatrix):
		totalDistance = 0
		previous_state = self.visited[0] 
		for city in self.visited:
			current_sate = city
			totalDistance += distanceMatrix[previous_state, current_sate]
			previous_state = current_sate
		return totalDistance

def getTSPpoints(filename):
	label = []
	x = []
	y = []
	with open(filename) as input:
		for l in input:
			line = l.split(' ')
			label.append(int(line[0]))
			x.append(float(line[1]))
			y.append(float(line[2]))
	return label, x, y

def plotTSPMap(x, y, label):
	plt.figure()
	plt.scatter(x, y)

	for j in range(len(label)):
		x_current = x[j]
		y_current = y[j]
		for k in range(len(label)):
			if j is not k:
				x_values = [x_current, x[k]]
				y_values = [y_current, y[k]]
				plt.plot(x_values, y_values, '-')

	for i in range(len(label)):
		plt.annotate(label[i], # this is the text
	                 (x[i],y[i]), # this is the point to label
	                 textcoords="offset points", # how to position the text
	                 xytext=(0,10), # distance from text to points (x,y)
	                 ha='center') # horizontal alignment can be left, right or center
	plt.show()

def getCities(filename):
	cities = []
	with open(filename) as input:
		for l in input:
			line = l.split(' ')
			city = {
				"n": float(line[0]),
				"x": float(line[1]),
				"y": float(line[2])
			}
			cities.append(city)
	return cities

def getDistanceMatrix(cities):
	size = len(cities), len(cities)
	distanceMatrix = np.zeros(size)
	for i in range(len(cities)):
		for j in range(len(cities)):
			x1 = cities[i]["x"]
			y1 = cities[i]["y"]
			x2 = cities[j]["x"]
			y2 = cities[j]["y"]

			distance = round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
			distanceMatrix[i, j] = distance
	return distanceMatrix

def getCostMatrix(distanceMatrix):
	size = len(distanceMatrix), len(distanceMatrix)
	costMatrix = np.zeros(size)
	for i in range(len(distanceMatrix)):
		for j in range(len(distanceMatrix)):
			if i is not j:
				costMatrix[i, j] = 1/distanceMatrix[i, j]
	return costMatrix

def plotRoute(x, y, label, route, cities):
	plt.figure()
	plt.scatter(x, y)
	prev_x = cities[route[0]]["x"]
	prev_y = cities[route[0]]["y"]
	for i in range(1, len(route)):
		x_values = [prev_x, cities[route[i]]["x"]]
		y_values = [prev_y, cities[route[i]]["y"]]
		plt.plot(x_values, y_values, '-')
		prev_x = cities[route[i]]["x"]
		prev_y = cities[route[i]]["y"]

	for i in range(len(label)):
		plt.annotate(label[i], # this is the text
	                 (x[i],y[i]), # this is the point to label
	                 textcoords="offset points", # how to position the text
	                 xytext=(0,10), # distance from text to points (x,y)
	                 ha='center') # horizontal alignment can be left, right or center
	plt.show()

def writeRoutesToFile(routes):
	filename = 'routes.csv'
	with open(filename, 'w') as output:
		for travelled in routes:
			output.write(str(travelled["cost"]))
			output.write(',')
			for city in travelled["route"]:
				output.write(str(city))
				output.write(',')
			output.write('\n')
	output.close()

def getDistanceFrequency(routes, distanceMatrix):
	distance = []
	for i in range(10):
		prev_city = routes[i]["route"][0]
		for j in range(1, len(routes[i]["route"])):
			current_city = routes[i]["route"][j]
			distance.append(distanceMatrix[prev_city, current_city])

	x = [i for i in range(len(distance))]
	distance.sort()

	norm_x = [i / max(x) for i in x]
	norm_distance = [j / max(distance) for j in distance]

	areaUnderGraph = 0
	for i in range(len(norm_x)):
		area = norm_x[i] * norm_distance[i]
		areaUnderGraph += area
	print(areaUnderGraph)

	plt.figure()
	plt.plot(norm_x, norm_distance)
	plt.show()

def updateValueMatrix(valueMatrix, travelledRoute, reward):
	prev_city = travelledRoute[0]
	for i in range(1, len(travelledRoute)):
		current_city = travelledRoute[i]
		valueMatrix[prev_city, current_city] += reward
		prev_city = current_city

def getMinDistance(distanceMatrix):
	rowSize = len(distanceMatrix[0,:])
	minDistance = sum(distanceMatrix[0, :])
	for i in range(1, rowSize):
		distance = sum(distanceMatrix[i, :])
		if distance < minDistance:
			minDistance = distance
	return minDistance

# Determine how Travelling agent show choose city to travel to
# Create a list, S with all travelled cities and N for not yet travelled cities
# Calculate the cost of travelling
# Create Q matrix that is the size of the distance matrix

if __name__ == '__main__':
	filename = 'att6.txt'
	label, x, y = getTSPpoints(filename)
	plotTSPMap(x, y, label)
	cities = getCities(filename)
	distanceMatrix = getDistanceMatrix(cities)
	costMatrix = getCostMatrix(distanceMatrix)

	valueMatrix = np.zeros((len(x), len(y)))

	considered_states = [k for k in range(len(label))]
	routeCosts = []

	currentDistanceTravelled = getMinDistance(distanceMatrix)
	print(currentDistanceTravelled)
	for state in considered_states:
		reward = 0
		for i in range(720):
			states = [j for j in range(len(label))]
			agent = Agent(state, states)

			while (agent.toVisit):
				agent.randomMovement()
			distanceTravelled = agent.calculateDistanceTravelled(distanceMatrix)

			travledDetails = {
				"cost": distanceTravelled,
				"route": agent.visited
			}
			reversedTravelDetails = {
				"cost": distanceTravelled,
				"route": agent.visited[::-1]
			}
			if (travledDetails not in routeCosts) and (reversedTravelDetails not in routeCosts):
				routeCosts.append(travledDetails)

				if (distanceTravelled < currentDistanceTravelled):
					updateValueMatrix(valueMatrix, agent.visited, reward)
					currentDistanceTravelled = distanceTravelled
			reward += 0.1
	
	print(valueMatrix)

	routeCosts.sort(key=lambda x: x.get('cost'))
	#print(routeCosts)
	print(len(routeCosts))
	min_distnace = routeCosts[0].get('cost')
	plotRoute(x, y, label, routeCosts[0]["route"], cities)

	#writeRoutesToFile(routeCosts)
	getDistanceFrequency(routeCosts, distanceMatrix)
