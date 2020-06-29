from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version = 1)

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

#calculate the distance between data points
def distance(xi, u):
	return (xi - u).transpose().dot(xi - u)

#calculate MSE
def MSE(data, centroids, clusters):
	sum = 0
	for i in range(len(data)):
		sum += distance(data[i], centroids[clusters[i]])
	return sum / len(data)

#compute cluster membership
def cluster_membership(data, centroids):
	clusters = []
	#calculate the distance between data points and centroids
	for i in range(len(data)):
		distances = []
		for centroid in centroids:
			distances.append(distance(data[i], centroid))
		#assign the data point to the closest centroid
		clusters.append(distances.index(min(distances)))
	return clusters

def my_kmeans(data, K, M):
	data = np.asarray(data)
	centroids = []
	clusters = []
	MSEs = []
	clustering_MSEs = []
	for i in range(M):
		#randomly initialize the centroids
		indices = random.sample(range(2000), 10)
		new_centroids = []
		for index in indices:
			new_centroids.append(data[index])
		new_clusters = cluster_membership(data, new_centroids)
		old_centroids = []
		old_clusters = []
		clustering_MSE = []
		while (True):
			clustering_MSE.append(MSE(data, new_centroids, new_clusters))
			old_centroids = new_centroids
			old_clusters = new_clusters
			#recompute cluster membership
			new_clusters = cluster_membership(data, new_centroids)
			#recompute the centroids
			means = []
			for i in range(K):
				sum = 0
				count = 0
				for j in range(len(new_clusters)):
					if (new_clusters[j] == i):
						sum += data[j]
						count += 1
				means.append(sum / count)
			new_centroids = means
			#if the distance between the new centroid and the old centroid differs by some threshold, the algorithm converges
			if (abs(MSE(data, new_centroids, new_clusters) - MSE(data, old_centroids, old_clusters)) < 10 ** (-5)):
				centroids.append(new_centroids)
				clusters.append(new_clusters)
				MSEs.append(MSE(data, new_centroids, new_clusters))
				clustering_MSEs.append(clustering_MSE)
				break
	x = list(range(1, M + 1))
	y = MSEs
	plt.plot(x, y)
	plt.ylabel("MSE")
	plt.show()
	#the optimal clustering has the minimum MSE
	optimal_clustering_MSEs = clustering_MSEs[MSEs.index(min(MSEs))]
	x = list(range(1, len(optimal_clustering_MSEs) + 1))
	y = optimal_clustering_MSEs
	plt.plot(x, y)
	plt.ylabel("MSE")
	plt.show()
	return (centroids, clusters, MSEs)

X = pd.DataFrame(mnist["data"])[:2000]
print(my_kmeans(X, 10, 15))