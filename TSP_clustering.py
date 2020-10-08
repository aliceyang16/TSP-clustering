import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

import scipy.cluster.hierarchy as sch

def writeToCSV(filename):
	fields = ['n', 'x', 'y']
	allData = []
	with open(filename+'.csv', 'w', newline='') as file:
		csvwriter = csv.writer(file)
		csvwriter.writerow(fields)
		with open(filename+'.txt') as input:
			for l in input:
				line = l.split(' ')
				if line[0] is not '#':
					data = [float(line[0]), float(line[1]), float(line[2])]
					allData.append(data)
		input.close()
		csvwriter.writerows(allData)
	file.close()

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

def getLeafMinDistance(leaf, new_leaf, distanceMatrix):
	distances = []
	index = 0
	for city in leaf:
		currentDistance = {
			"index": index,
			"city": city,
			"distance": distanceMatrix[city, new_leaf]
		}
		distances.append(currentDistance)
		index += 1
	distances.sort(key=lambda x: x.get('distance'))
	return distances[0]["index"]

def unique(routes):
	unique_routes = []
	unique_routes_set = set(routes)

	for route in unique_routes_set:
		unique_routes.append(route)
	return unique_routes

def calculateDistanceTravelled(visited, distanceMatrix):
		totalDistance = 0
		previous_state = visited[0] 
		for city in visited:
			current_sate = city
			totalDistance += distanceMatrix[previous_state, current_sate]
			previous_state = current_sate
		return totalDistance

def getHierarchialSolution(route, distanceMatrix, label):
	visited_leaf = route[0]
	route.remove(visited_leaf)

	while len(visited_leaf) is not len(label):
		distances = []
		leaf_node_first = visited_leaf[0]
		leaf_node_second = visited_leaf[len(visited_leaf) - 1]
		for current_leaf in route:
			start_node = current_leaf[0]
			end_node = current_leaf[len(current_leaf) - 1]

			# start_distance_first = {
			# 	"city": start_node,
			# 	"paired_cities":current_leaf[1:],
			# 	"cities": current_leaf,
			# 	"distance": distanceMatrix[leaf_node_first][start_node],
			# 	"entry" : True
			# }

			start_distance_second = {
				"city": start_node,
				"paired_cities": current_leaf[1:],
				"cities": current_leaf,
				"distance": distanceMatrix[leaf_node_second][start_node],
				"entry": False
			}
			# distances.append(start_distance_first)
			distances.append(start_distance_second)

			# end_distance_first = {
			# 	"city": end_node,
			# 	"paired_cities": list(reversed(current_leaf[:len(current_leaf)-1])),
			# 	"cities": current_leaf,
			# 	"distance": distanceMatrix[leaf_node_first][end_node],
			# 	"entry": True
			# }

			end_distance_second = {
				"city": end_node,
				"paired_cities": list(reversed(current_leaf[:len(current_leaf)-1])),
				"cities": current_leaf,
				"distance": distanceMatrix[leaf_node_second][end_node],
				"entry": False
			}
			# distances.append(end_distance_first)
			distances.append(end_distance_second)

		distances.sort(key=lambda x: x.get('distance'))
		#print(distances)
		current_visited = [distances[0]["city"]]
		current_visited.extend(distances[0]["paired_cities"])

		visited_leaf.extend(current_visited)

		route.remove(distances[0]["cities"])

	visited_leaf.append(visited_leaf[0])
	return visited_leaf

def plotRouteSolution(x, y, label, solution):
	plt.figure()
	plt.scatter(x, y)
	prev_x = x[solution[0]]
	prev_y = y[solution[0]]
	for i in range(1, len(solution)):
		x_current = x[solution[i]]
		y_current = y[solution[i]]
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

def plotClusterRoute(x, y, label, routes):
	plt.figure()
	plt.scatter(x, y)
	for cluster in routes:
		prev_x = x[cluster[0]]
		prev_y = y[cluster[0]]
		for i in range(1, len(cluster)):
			x_current = x[cluster[i]]
			y_current = y[cluster[i]]
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


def clusterCheck(first_value, second_value, index, Z, label, true_index_list):
	if first_value < len(label) and second_value >= len(label):
		index = second_value % len(label)
		cluster = Z[index]
		return clusterCheck(int(cluster[0]), int(cluster[1]), index, Z, label, true_index_list)
	elif first_value >= len(label) and second_value < len(label):
		index = first_value % len(label)
		cluster = Z[index]
		return clusterCheck(int(cluster[0]), int(cluster[1]), index, Z, label, true_index_list)
	else:
		if first_value >= len(label) and second_value >= len(label):
			index = first_value % len(label)
			cluster = Z[index]
			return clusterCheck(int(cluster[0]), int(cluster[1]), index, Z, label, true_index_list)
		else:
			true_index = true_index_list.index(index)
			return true_index


if __name__ == '__main__':
	filename = 'Problems/tsp225.txt'
	label, x, y = getTSPpoints(filename)
	plotTSPMap(x, y, label)
	cities = getCities(filename)
	distanceMatrix = getDistanceMatrix(cities)
	#print(distanceMatrix)

	filename = os.path.splitext(filename)[0]
	dataset = pd.read_csv(filename+'.csv')
	X = dataset.iloc[:, [1,2]].values

	original_route = []
	true_index_list = []
	counter = 0
	Z = sch.linkage(X, 'ward')
	for cluster in Z:
		if cluster[0] < len(label) and cluster[1] < len(label):
			leaf = [int(cluster[0]), int(cluster[1])]
			original_route.append(leaf)
			true_index_list.append(counter)
		else:
			if cluster[0] < len(label):
				index = clusterCheck(int(cluster[0]), int(cluster[1]), 0, Z, label, true_index_list)
				position = getLeafMinDistance(original_route[index], int(cluster[0]), distanceMatrix)
				original_route[index].insert(position, int(cluster[0]))

			if cluster[1] < len(label):
				index = clusterCheck(int(cluster[0]), int(cluster[1]), 0, Z, label, true_index_list)
				position = getLeafMinDistance(original_route[index], int(cluster[1]), distanceMatrix)
				original_route[index].insert(position, int(cluster[1]))
				# original_route.append(original_route[index])
		counter += 1
	
	#original_route = unique(original_route)
	#print(original_route)
	#plotClusterRoute(x, y, label, original_route)
	route = original_route.copy()
	# reverse_route = [x for x in original_route[::-1]]

	visited_leaf = getHierarchialSolution(route, distanceMatrix, label)
	visited_leaf_distance = calculateDistanceTravelled(visited_leaf, distanceMatrix)
	print(visited_leaf_distance)
	plotRouteSolution(x, y, label, visited_leaf)

	# reverse_visited_leaf = getHierarchialSolution(reverse_route, distanceMatrix, label)
	# reverse_visited_leaf_distance = calculateDistanceTravelled(reverse_visited_leaf, distanceMatrix)
	# print(reverse_visited_leaf)
	# plotRouteSolution(x, y, label, reverse_visited_leaf)