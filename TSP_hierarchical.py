import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
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
			distanceMatrix[i][j] = distance
	return distanceMatrix

def calculateDistanceTravelled(visited, distanceMatrix):
		totalDistance = 0
		previous_state = visited[0] 
		for city in visited:
			current_sate = city
			totalDistance += distanceMatrix[previous_state][current_sate]
			previous_state = current_sate
		return totalDistance

def getLeafMinDistance(leaf, new_leaf, distanceMatrix):
	distances = []
	index = 0
	for city in leaf:
		currentDistance = {
			"index": index,
			"city": city,
			"distance": distanceMatrix[city][new_leaf]
		}
		distances.append(currentDistance)
		index += 1
	distances.sort(key=lambda x: x.get('distance'))
	return distances[0]["index"]

def combineLeaves(leaf1, leaf2, distanceMatrix):
	leaf1_start = leaf1[0]
	leaf1_end = leaf1[len(leaf1) - 1]
	leaf2_start = leaf2[0]
	leaf2_end = leaf2[len(leaf2) - 1]

	distance1 = distanceMatrix[leaf1_start][leaf2_start]
	distance2 = distanceMatrix[leaf1_start][leaf2_end]
	distance3 = distanceMatrix[leaf1_end][leaf2_start]
	distance4 = distanceMatrix[leaf1_end][leaf2_end]

	distances = [distance1, distance2, distance3, distance4]
	minDistance = min(distances)
	index = distances.index(minDistance)

	if index == 0:
		new_leaf_route = leaf2[::-1].copy()
		new_leaf_route.extend(leaf1)
	elif index == 1:
		new_leaf_route = leaf2.copy()
		leaf1 = leaf1[::-1].copy()
		new_leaf_route.extend(leaf1)
	elif index == 2:
		new_leaf_route = leaf1.copy()
		new_leaf_route.extend(leaf2)
	else:
		new_leaf_route = leaf1.copy()
		leaf2 = leaf2[::-1].copy()
		new_leaf_route.extend(leaf2)

	return new_leaf_route

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

def pairClusters(route_id, original_route, hierarchy, distanceMatrix):
	grown_leaves = []
	for cluster in hierarchy:
		if int(cluster[0]) in route_id:
			if int(cluster[1]) in route_id:
				leaf1 = original_route[route_id.index(int(cluster[0]))]
				leaf2 = original_route[route_id.index(int(cluster[1]))]
				new_leaf = combineLeaves(leaf1, leaf2, distanceMatrix)
				original_route.append(new_leaf)
				grown_leaves.append(leaf1)
				original_route.remove(leaf2)
				route_id.remove(int(cluster[1]))

	for leaf in grown_leaves:
		original_route.remove(leaf)

def getHierarchialSolution(route, distanceMatrix, label):
	visited_leaf = route[0]
	route.remove(visited_leaf)
	leaf_node_first = visited_leaf[0]
	leaf_node_second = visited_leaf[len(visited_leaf) - 1]
	while len(visited_leaf) is not len(label):
		distances = []
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
		leaf_node_first = current_visited[0]
		leaf_node_second = current_visited[len(current_visited) - 1]

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

if __name__ == '__main__':
	filename = 'Problems/berlin52.txt'
	label, x, y = getTSPpoints(filename)
	plotTSPMap(x, y, label)
	cities = getCities(filename)
	distanceMatrix = getDistanceMatrix(cities)

	# Hierarchical clustering
	filename = os.path.splitext(filename)[0]
	dataset = pd.read_csv(filename+'.csv')
	X = dataset.iloc[:, [1,2]].values
	Z = sch.linkage(X, 'ward')

	original_route = []
	true_index_list = []
	route_id = []
	counter = 0  
	for cluster in Z:
		if cluster[0] < len(label) and cluster[1] < len(label):
			leaf = [int(cluster[0]), int(cluster[1])]
			original_route.append(leaf)
			route_id.append(len(Z) + counter + 1)
			true_index_list.append(counter)
		else:
			if cluster[0] < len(label):
				index = clusterCheck(int(cluster[0]), int(cluster[1]), 0, Z, label, true_index_list)
				position = getLeafMinDistance(original_route[index], int(cluster[0]), distanceMatrix)
				original_route[index].insert(position, int(cluster[0]))
				IDindex = route_id.index(int(cluster[1]))
				route_id[IDindex] = len(Z) + counter + 1

			if cluster[1] < len(label):
				index = clusterCheck(int(cluster[0]), int(cluster[1]), 0, Z, label, true_index_list)
				position = getLeafMinDistance(original_route[index], int(cluster[1]), distanceMatrix)
				original_route[index].insert(position, int(cluster[1]))
				IDindex = route_id.index(int(cluster[0]))
				route_id[IDindex] = len(Z) + counter + 1
				# original_route.append(original_route[index])
		counter += 1

	plotClusterRoute(x, y, label, original_route )
	#print(original_route)
	pairClusters(route_id, original_route, Z, distanceMatrix)
	route = original_route.copy()
	visited_leaf = getHierarchialSolution(route, distanceMatrix, label)
	visited_leaf_distance = calculateDistanceTravelled(visited_leaf, distanceMatrix)
	#print(visited_leaf)
	print(visited_leaf_distance)
	plotRouteSolution(x, y, label, visited_leaf)