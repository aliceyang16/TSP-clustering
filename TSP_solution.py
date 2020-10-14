import os
import numpy as np
import matplotlib.pyplot as plt 
import math

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

def readSolutionPoints(filename):
	solution = []
	with open(filename) as input:
		for line in input:
			solution.append(int(line))
	return solution

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

def calculateDistanceTravelled(visited, distanceMatrix):
		totalDistance = 0
		previous_state = visited[0] - 1
		for city in visited:
			current_sate = city - 1
			totalDistance += distanceMatrix[previous_state, current_sate]
			previous_state = current_sate
		return totalDistance

def plotRouteSolution(x, y, label, solution):
	plt.figure()
	plt.scatter(x, y)
	prev_x = x[solution[0] - 1]
	prev_y = y[solution[0] - 1]
	for i in range(1, len(solution)):
		x_current = x[solution[i] - 1]
		y_current = y[solution[i] - 1]
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
	problem_filename = 'Problems/berlin52.txt'
	solution_filename = 'Optimal_Solutions/berlin52.txt'

	label, x, y = getTSPpoints(problem_filename)
	solution = readSolutionPoints(solution_filename)
	cities = getCities(problem_filename)
	distanceMatrix = getDistanceMatrix(cities)
	travelled_distance = calculateDistanceTravelled(solution, distanceMatrix)

	plotRouteSolution(x, y, label, solution)
	print(travelled_distance)