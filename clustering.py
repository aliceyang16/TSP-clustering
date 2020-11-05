import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

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

def plotDendrogram(X):
	dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
	plt.title('Dendrogram')
	plt.xlabel('Cities')
	plt.ylabel('Euclidean distances')
	plt.show()

def plotCluster(X):
	cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
	cluster.fit_predict(X)
	plt.figure(figsize=(10, 7))
	plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
	plt.show()

if __name__ == '__main__':
	filename = 'Problems/att48'
	writeToCSV(filename)

	dataset = pd.read_csv(filename+'.csv')
	X = dataset.iloc[:, [1,2]].values
	plotDendrogram(X)

	Z = sch.linkage(X, 'ward')