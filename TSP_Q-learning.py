# Travelling Salesman Problem Q-learning
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import pylab as pl

import scipy.cluster.hierarchy as sch

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
				"n": int(line[0]),
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
			distanceMatrix[i][j] = distance
	return distanceMatrix

def getRewardMatrix(distanceMatrix):
	rewardMatrix = np.zeros(distanceMatrix.shape)
	for row in range(distanceMatrix.shape[0]):
		for col in range(distanceMatrix.shape[1]):
			if row is not col:
				rewardMatrix[row][col] = 1 / distanceMatrix[row][col]
	return rewardMatrix

def calculateRouteCost(visited, rewardMatrix):
		totalDistance = 0
		previous_state = visited[0] 
		for city in visited:
			current_sate = city
			totalDistance += rewardMatrix[previous_state][current_sate]
			previous_state = current_sate
		return totalDistance

def getAvailableAction(rewardMatrix, state, visited):
	options = rewardMatrix[state, :]
	availableAction = np.where(options > 0)[0]
	availableAction = [x for x in availableAction if x not in visited]
	return availableAction

# def getAvailableAction(rewardMatrix, state):
# 	options = rewardMatrix[state, :]
# 	availableAction = np.where(options > 0)[0]
# 	return availableAction

def getNextAction(options):
	nextAction = int(np.random.choice(options, 1))
	return nextAction

def update(Q, rewardMatrix, currentState, nextState, gamma):
	maxIndex = np.where(Q[nextState, :] == np.max(Q[nextState, :]))[0]
	if maxIndex.shape[0] > 1:
		maxIndex = int(np.random.choice(maxIndex, size = 1))
	else:
		maxIndex = int(maxIndex)
	maxValue = Q[nextState, maxIndex]
	Q[currentState][nextState] = rewardMatrix[currentState, nextState] + gamma * maxValue
	Q[nextState][currentState] = rewardMatrix[currentState, nextState] + gamma * maxValue

	if (np.max(Q) > 0):
		return (np.sum(Q / np.max(Q) * 100))
	else:
		return (0)

def updateRoute(Q, rewardMatrix, gamma, route, cost):
	# Reward the route in the Q matrix that has the
	routeCost = calculateRouteCost(route, rewardMatrix)
	if routeCost > cost:
		currentState = route[0]
		for i in range(1, len(route)):
			nextState = route[i]
			update(Q, rewardMatrix, currentState, nextState, gamma)
		return routeCost

if __name__ == '__main__':
	filename = 'Problems/lau15_xy.txt'
	label, x, y = getTSPpoints(filename)
	plotTSPMap(x, y, label)
	cities = getCities(filename)
	distanceMatrix = getDistanceMatrix(cities)
	rewardMatrix = getRewardMatrix(distanceMatrix)

	Q = np.zeros(rewardMatrix.shape)
	gamma = 0.75
	# initial_state = 1
	# visited = [initial_state]

	# availableAction = getAvailableAction(rewardMatrix, initial_state, visited)
	# nextAction = getNextAction(availableAction)
	# visited.append(nextAction)
	# update(Q, rewardMatrix, initial_state, nextAction, gamma)

	scores = []
	for i in range(1000):
		currentState = np.random.randint(0, int(Q.shape[0]))
		visited = [currentState]
		while len(visited) is not len(cities):
			availableAction = getAvailableAction(rewardMatrix, currentState, visited)
			nextState = getNextAction(availableAction)
			visited.append(nextState)
			score = update(Q, rewardMatrix, currentState, nextState, gamma)
			scores.append(score)
	print(Q)
	# Testing 
	current_state = 0
	steps = [current_state]
	print('TESTING')

	while len(steps) is not len(cities):
		potential = Q[current_state, ]
		for visited in steps:
			potential[visited] = 0
		next_step_index = np.where(Q[current_state, ] == np.max(potential))[0] 
		if next_step_index.shape[0] > 1: 
			next_step_index = int(np.random.choice(next_step_index, size = 1)) 
		else: 
			next_step_index = int(next_step_index) 
		steps.append(next_step_index) 
		current_state = next_step_index 

	print("Most efficient path:") 
	print(steps) 
	pl.plot(scores) 
	pl.xlabel('No of iterations') 
	pl.ylabel('Reward gained') 
	pl.show()