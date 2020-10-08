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

	def getLevyDistribution(self):
		filename = 'LevyDistLUTable.csv'
		levyDistribution = []
		with open (filename) as input:
			for l in input:
				line = l.split(',')
				levyDistribution.append(float(line[1]))
		return levyDistribution

	def indexSearch(self, item, levyDistribution, min_index, max_index):
		if max_index <= min_index:
			return min_index
		else:
			mid_index = math.floor((min_index + max_index)/2)

			if levyDistribution[min_index] > item:
				return self.indexSearch(item, levyDistribution, min_index, mid_index - 1)
			elif levyDistribution[mid_index] < item:
				return self.indexSearch(item, levyDistribution, mid_index + 1, max_index)
			else:
				return mid_index

	def getNextCity(self, index, visit_cost):
		mini = 0
		maxi = 0
		i = 0
		cityFound = False
		while (not cityFound):
			if index == 10000:
				cityFound = True
				node_position = len(visit_cost) - 1
				break

			mini = maxi
			maxi += visit_cost[i]["probabilityCost"]

			if (index >= mini * 10000) and (index < maxi * 10000):
				cityFound = True
				node_position = i
				break
			i += 1
		return node_position

	def FPAmovement(self, costMatrix):
		remaining_cost = [costMatrix[self.state, city] for city in self.toVisit]
		sum_cost =  sum(remaining_cost)
		visit_cost = [
			{
				"city": self.toVisit[i],
				"probabilityCost": remaining_cost[i] / sum_cost
			} for i in range(len(self.toVisit))
		]
		visit_cost.sort(key=lambda x: x.get("probabilityCost"))

		p = random.uniform(0, 1) # p is the switching probability
		r = random.uniform(0, 1) # r is the random number that is compared to p to determine global or local search
		p = 0
		if r > p:
			# Global search
			levyDistribution = self.getLevyDistribution()
			upper = max(levyDistribution)
			lower = min(levyDistribution)
			w = (lower - upper) * random.random() + upper
			index = self.indexSearch(w, levyDistribution, 0, len(levyDistribution)-1)
			nextCity = self.getNextCity(index, visit_cost)

			if (nextCity in self.toVisit):
				self.state = nextCity
				self.visited.append(nextCity)
				self.toVisit.remove(nextCity)
			else:
				nextCity = visit_cost[0]["city"]
				self.state = nextCity
				self.visited.append(nextCity)
				self.toVisit.remove(nextCity)
		else:
			#Local search
			nextCity = visit_cost[0]["city"]
			self.state = nextCity
			self.visited.append(nextCity)
			self.toVisit.remove(nextCity)


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

def plotRouteTravelledDistance(route, distanceMatrix):
	distance = []
	current_city = route["route"][0]
	for i in range(1, len(route["route"])):
		next_city = route["route"][i]
		distance.append(distanceMatrix[current_city, next_city])
		current_city = next_city
	x = [i for i in range(len(distance))]
	distance.sort()

	norm_x = [i / max(x) for i in x]
	norm_distance = [j / max(distance) for j in distance]

	plt.figure()
	plt.plot(norm_x, norm_distance)
	plt.show()

# Determine how Travelling agent show choose city to travel to
# Create a list, S with all travelled cities and N for not yet travelled cities
# Calculate the cost of travelling
# Create Q matrix that is the size of the distance matrix
def bruteForceTSP(x, y, label, distanceMatrix, cities):
	#num_cols = int(len(y)/2)
	#num_rows = int(len(x)/2)
	#plt.figure(figsize=(2*num_cols, 2*num_rows))

	considered_states = [k for k in range(len(label))]
	routes = []
	for state in considered_states:
		stateRoutes = []
		for i in range(720):
			states = [j for j in range(len(label))]
			agent = Agent(state, states)

			while (agent.toVisit):
				agent.randomMovement()
			agent.visited.append(state)
			distanceTravelled = agent.calculateDistanceTravelled(distanceMatrix)
			travledDetails = {
				"cost": distanceTravelled,
				"route": agent.visited
			}
			reversedTravelDetails = {
				"cost": distanceTravelled,
				"route": agent.visited[::-1]
			}
			if (travledDetails not in stateRoutes) and (reversedTravelDetails not in stateRoutes):
				stateRoutes.append(travledDetails)
		stateRoutes.sort(key=lambda x: x.get('cost'))
		#plotRouteTravelledDistance(stateRoutes[0], distanceMatrix)
		#plt.subplot(num_rows, num_cols, state+1)
		#plotRoute(x, y, label, stateRoutes[0]["route"], cities)
		#print(stateRoutes[0]["route"])
		#print(stateRoutes[0]["cost"])
		routes.extend(stateRoutes)
	#plt.tight_layout()
	#plt.show()

	routes.sort(key=lambda x: x.get('cost'))
	#print(routes)
	print(len(routes))
	min_distnace = routes[0].get('cost')
	
	plotRoute(x, y, label, routes[0]["route"], cities)
	#writeRoutesToFile(routeCosts)
	getDistanceFrequency(routes, distanceMatrix)

def FPA_TSP(x, y, label, distanceMatrix, costMatrix):
	considered_states = [k for k in range(len(label))]
	routes = []
	for i in range(720):
		for state in considered_states:
			states = [j for j in range(len(label))]
			agent = Agent(state, states)

			while (agent.toVisit):
				agent.FPAmovement(costMatrix)
			agent.visited.append(state)
			distanceTravelled = agent.calculateDistanceTravelled(distanceMatrix)
			travledDetails = {
				"cost": distanceTravelled,
				"route": agent.visited
			}
			reversedTravelDetails = {
				"cost": distanceTravelled,
				"route": agent.visited[::-1]
			}
			if (travledDetails not in routes) and (reversedTravelDetails not in routes):
				routes.append(travledDetails)

	routes.sort(key=lambda x: x.get('cost'))
	print(routes)

	min_distnace = routes[0].get('cost')
	plotRoute(x, y, label, routes[0]["route"], cities)
	getDistanceFrequency(routes, distanceMatrix)

def plotRouteSolution(x, y, label, solution):
	plt.figure()
	plt.scatter(x, y)
	prev_x = x[solution[0]-1]
	prev_y = y[solution[0]-1]
	for i in range(1, len(solution)):
		x_current = x[solution[i]-1]
		y_current = y[solution[i]-1]
		x_values = [prev_x, x_current]
		y_values = [prev_y, y_current]
		plt.plot(x_values, y_values, '-')
		prev_x = x_current
		prev_y = y_current

	for i in range(len(label)):
		plt.annotate(label[i], # this is the text
	                 (x[i],y[i]), # this is the point to label
	                 textcoords="offset points", # how to position the text
	                 xytext=(0,10), # distance from text to points (x,y)
	                 ha='center') # horizontal alignment can be left, right or center
	plt.show()

if __name__ == '__main__':
	filename = 'lau15_xy.txt'
	label, x, y = getTSPpoints(filename)
	plotTSPMap(x, y, label)
	cities = getCities(filename)
	distanceMatrix = getDistanceMatrix(cities)
	costMatrix = getCostMatrix(distanceMatrix)

	solution = [1, 13, 2, 15, 9, 5, 7, 3, 12, 14, 10, 8, 6, 4, 11, 1]
	plotRouteSolution(x, y, label, solution)

	# bruteForceTSP(x, y, label, distanceMatrix, cities)
	#FPA_TSP(x, y, label, distanceMatrix, costMatrix)