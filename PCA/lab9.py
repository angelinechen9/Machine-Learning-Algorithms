from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version = 1)

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def pca(X, Y):
	#center the features
	X = X - X.mean()
	#calculate the covariance matrix
	covariance = X.transpose().dot(X) / 2000
	#calculate the eigenvalues and the eigenvectors of the covariance matrix
	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	#project the data to the 2D component space
	eigenvalue2 = eigenvalues[0:2]
	eigenvector2 = eigenvectors[:, 0:2]
	#project the data to the 3D component space
	eigenvalue3 = eigenvalues[0:3]
	eigenvector3 = eigenvectors[:, 0:3]
	#calculate the variance for 2 components
	variance2 = np.sum(eigenvalue2) / np.sum(eigenvalues)
	print("variance for 2 components")
	print(variance2.astype(np.float128))
	#calculate the variance for 3 components
	variance3 = np.sum(eigenvalue3) / np.sum(eigenvalues)
	print("variance for 3 components")
	print(variance3.astype(np.float128))
	#graph the data in the new dimension space
	projection2 = X.dot(eigenvector2)
	x = projection2[0].astype(np.float128)
	y = projection2[1].astype(np.float128)
	plt.scatter(x, y, c = Y)
	plt.show()
	projection3 = X.dot(eigenvector3)
	x = projection3[0].astype(np.float128)
	y = projection3[1].astype(np.float128)
	z = projection3[2].astype(np.float128)
	fig = plt.figure()
	ax = plt.axes(projection = "3d")
	ax.scatter(x, y, z, c = Y)
	plt.show()

X = pd.DataFrame(mnist["data"])[:2000]
Y = pd.DataFrame(mnist["target"])[:2000]
pca(X, Y)