import pandas as pd
import numpy as np
import math
import time

#calculate the mean for each class
def mean(x):
	mean = np.zeros(4)
	x = np.asarray(x)
	for i in range(40):
		mean += x[i]
	mean /= 40
	return mean

#calculate the covariance for each class
def covariance(x, mean):
	covariance = np.zeros((4, 4))
	x = np.asarray(x)
	for i in range(40):
		covariance += (x[i].reshape(4, 1) - mean.reshape(4, 1)).transpose() * (x[i].reshape(4, 1) - mean.reshape(4, 1))
	covariance /= 40
	return covariance

#calculate the covariance diagonal matrix for each class
def covariance_diagonal_matrix(x, mean):
	return np.diag(np.var(x))

#calculate the probability using mean and covariance
def multivariate_gaussian(xi, k, mean, covariance):
	xi = np.asarray(xi).reshape(4, 1)
	mean = np.asarray(mean).reshape(4, 1)
	covariance = np.asarray(covariance)
	return (1 / (math.sqrt((2 * np.pi) ** k * np.linalg.det(covariance)))) * math.exp(-(1 / 2) * ((xi - mean).transpose().dot(np.linalg.inv(covariance))).dot(xi - mean))

#classify the data point
def lda_probability(x, setosa_mean, versicolor_mean, virginica_mean, average_covariance):
	setosa_probability = multivariate_gaussian(x, 4, setosa_mean, average_covariance)
	versicolor_probability = multivariate_gaussian(x, 4, versicolor_mean, average_covariance)
	virginica_probability = multivariate_gaussian(x, 4, virginica_mean, average_covariance)
	probabilities = [setosa_probability, versicolor_probability, virginica_probability]
	return probabilities.index(max(probabilities))

#classify the data point
def qda_probability(x, setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance):
	setosa_probability = multivariate_gaussian(x, 4, setosa_mean, setosa_covariance)
	versicolor_probability = multivariate_gaussian(x, 4, versicolor_mean, versicolor_covariance)
	virginica_probability = multivariate_gaussian(x, 4, virginica_mean, virginica_covariance)
	probabilities = [setosa_probability, versicolor_probability, virginica_probability]
	return probabilities.index(max(probabilities))

#calculate the error
def lda_classifier(x_setosa, x_versicolor, x_virginica, setosa_mean, versicolor_mean, virginica_mean, average_covariance):
	count = 0
	for i in range(len(x_setosa)):
		if (lda_probability(x_setosa.to_numpy().tolist()[i], setosa_mean, versicolor_mean, virginica_mean, average_covariance) == 0):
			count += 1
	for i in range(len(x_versicolor)):
		if (lda_probability(x_versicolor.to_numpy().tolist()[i], setosa_mean, versicolor_mean, virginica_mean, average_covariance) == 1):
			count += 1
	for i in range(len(x_virginica)):
		if (lda_probability(x_virginica.to_numpy().tolist()[i], setosa_mean, versicolor_mean, virginica_mean, average_covariance) == 2):
			count += 1
	accuracy = count / (len(x_setosa) + len(x_versicolor) + len(x_virginica))
	print("accuracy: " + str(accuracy * 100) + "%")
	error = 1 - accuracy
	print("error: " + str(error * 100) + "%")

#check if the values are linearly separable
def linearly_separable(x_train_setosa, x_train_versicolor, x_train_virginica, setosa_mean, versicolor_mean, virginica_mean, average_covariance):
	print("setosa and versicolor")
	count = 0
	for i in range(len(x_train_setosa)):
		setosa_probability = multivariate_gaussian(x_train_setosa.to_numpy().tolist()[i], 4, setosa_mean, average_covariance)
		versicolor_probability = multivariate_gaussian(x_train_setosa.to_numpy().tolist()[i], 4, versicolor_mean, average_covariance)
		if (setosa_probability > versicolor_probability):
			count += 1
	for i in range(len(x_train_versicolor)):
		setosa_probability = multivariate_gaussian(x_train_versicolor.to_numpy().tolist()[i], 4, setosa_mean, average_covariance)
		versicolor_probability = multivariate_gaussian(x_train_versicolor.to_numpy().tolist()[i], 4, versicolor_mean, average_covariance)
		if (versicolor_probability > setosa_probability):
			count += 1
	accuracy = count / (len(x_train_setosa) + len(x_train_versicolor))
	print("accuracy: " + str(accuracy * 100) + "%")
	error = 1 - accuracy
	print("error: " + str(error * 100) + "%")
	#if the train error rate is equal to 0, the values are linearly separable
	if (error == 0):
		print("The values are linearly separable.")
	else:
		print("The values are not linearly separable.")
	print("setosa and virginica")
	count = 0
	for i in range(len(x_train_setosa)):
		setosa_probability = multivariate_gaussian(x_train_setosa.to_numpy().tolist()[i], 4, setosa_mean, average_covariance)
		virginica_probability = multivariate_gaussian(x_train_setosa.to_numpy().tolist()[i], 4, virginica_mean, average_covariance)
		if (setosa_probability > virginica_probability):
			count += 1
	for i in range(len(x_train_virginica)):
		setosa_probability = multivariate_gaussian(x_train_virginica.to_numpy().tolist()[i], 4, setosa_mean, average_covariance)
		virginica_probability = multivariate_gaussian(x_train_virginica.to_numpy().tolist()[i], 4, virginica_mean, average_covariance)
		if (virginica_probability > setosa_probability):
			count += 1
	accuracy = count / (len(x_train_setosa) + len(x_train_virginica))
	print("accuracy: " + str(accuracy * 100) + "%")
	error = 1 - accuracy
	print("error: " + str(error * 100) + "%")
	#if the train error rate is equal to 0, the values are linearly separable
	if (error == 0):
		print("The values are linearly separable.")
	else:
		print("The values are not linearly separable.")
	print("versicolor and virginica")
	count = 0
	for i in range(len(x_train_versicolor)):
		versicolor_probability = multivariate_gaussian(x_train_versicolor.to_numpy().tolist()[i], 4, versicolor_mean, average_covariance)
		virginica_probability = multivariate_gaussian(x_train_versicolor.to_numpy().tolist()[i], 4, virginica_mean, average_covariance)
		if (versicolor_probability > virginica_probability):
			count += 1
	for i in range(len(x_train_virginica)):
		versicolor_probability = multivariate_gaussian(x_train_virginica.to_numpy().tolist()[i], 4, versicolor_mean, average_covariance)
		virginica_probability = multivariate_gaussian(x_train_virginica.to_numpy().tolist()[i], 4, virginica_mean, average_covariance)
		if (virginica_probability > versicolor_probability):
			count += 1
	accuracy = count / (len(x_train_versicolor) + len(x_train_virginica))
	print("accuracy: " + str(accuracy * 100) + "%")
	error = 1 - accuracy
	print("error: " + str(error * 100) + "%")
	#if the train error rate is equal to 0, the values are linearly separable
	if (error == 0):
		print("The values are linearly separable.")
	else:
		print("The values are not linearly separable.")

#calculate the error
def qda_classifier(x_setosa, x_versicolor, x_virginica, setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance):
	count = 0
	for i in range(len(x_setosa)):
		if (qda_probability(x_setosa.to_numpy().tolist()[i], setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance) == 0):
			count += 1
	for i in range(len(x_versicolor)):
		if (qda_probability(x_versicolor.to_numpy().tolist()[i], setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance) == 1):
			count += 1
	for i in range(len(x_virginica)):
		if (qda_probability(x_virginica.to_numpy().tolist()[i], setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance) == 2):
			count += 1
	accuracy = count / (len(x_setosa) + len(x_versicolor) + len(x_virginica))
	print("accuracy: " + str(accuracy * 100) + "%")
	error = 1 - accuracy
	print("error: " + str(error * 100) + "%")

def lda_model(x_train_setosa, x_train_versicolor, x_train_virginica, x_test_setosa, x_test_versicolor, x_test_virginica):
	#train the model
	setosa_mean = mean(x_train_setosa)
	versicolor_mean = mean(x_train_versicolor)
	virginica_mean = mean(x_train_virginica)
	start = time.time()
	setosa_covariance = covariance(x_train_setosa, setosa_mean)
	versicolor_covariance = covariance(x_train_versicolor, versicolor_mean)
	virginica_covariance = covariance(x_train_virginica, virginica_mean)
	end = time.time()
	training_time = end - start
	print("training time: " + str(training_time))
	#use the average covariance
	average_covariance = (setosa_covariance + versicolor_covariance + virginica_covariance) / 3
	#test the model
	print("train")
	lda_classifier(x_train_setosa, x_train_versicolor, x_train_virginica, setosa_mean, versicolor_mean, virginica_mean, average_covariance)
	print("test")
	lda_classifier(x_test_setosa, x_test_versicolor, x_test_virginica, setosa_mean, versicolor_mean, virginica_mean, average_covariance)
	#check if the values are linearly separable
	linearly_separable(x_train_setosa, x_train_versicolor, x_train_virginica, setosa_mean, versicolor_mean, virginica_mean, average_covariance)
	#use the covariance diagonal matrix
	print("diagonal matrix")
	#train the model
	start = time.time()
	setosa_covariance = covariance_diagonal_matrix(x_train_setosa, setosa_mean)
	versicolor_covariance = covariance_diagonal_matrix(x_train_versicolor, versicolor_mean)
	virginica_covariance = covariance_diagonal_matrix(x_train_virginica, virginica_mean)
	end = time.time()
	training_time = end - start
	print("training time: " + str(training_time))
	#use the average covariance
	average_covariance = (setosa_covariance + versicolor_covariance + virginica_covariance) / 3
	#test the model
	print("train")
	lda_classifier(x_train_setosa, x_train_versicolor, x_train_virginica, setosa_mean, versicolor_mean, virginica_mean, average_covariance)
	print("test")
	lda_classifier(x_test_setosa, x_test_versicolor, x_test_virginica, setosa_mean, versicolor_mean, virginica_mean, average_covariance)

def qda_model(x_train_setosa, x_train_versicolor, x_train_virginica, x_test_setosa, x_test_versicolor, x_test_virginica):
	#train the model
	setosa_mean = mean(x_train_setosa)
	versicolor_mean = mean(x_train_versicolor)
	virginica_mean = mean(x_train_virginica)
	start = time.time()
	setosa_covariance = covariance(x_train_setosa, setosa_mean)
	versicolor_covariance = covariance(x_train_versicolor, versicolor_mean)
	virginica_covariance = covariance(x_train_virginica, virginica_mean)
	end = time.time()
	training_time = end - start
	print("training time: " + str(training_time))
	#test the model
	print("train")
	qda_classifier(x_train_setosa, x_train_versicolor, x_train_virginica, setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance)
	print("test")
	qda_classifier(x_test_setosa, x_test_versicolor, x_test_virginica, setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance)
	#use the covariance diagonal matrix
	print("diagonal matrix")
	#train the model
	start = time.time()
	setosa_covariance = covariance_diagonal_matrix(x_train_setosa, setosa_mean)
	versicolor_covariance = covariance_diagonal_matrix(x_train_versicolor, versicolor_mean)
	virginica_covariance = covariance_diagonal_matrix(x_train_virginica, virginica_mean)
	end = time.time()
	training_time = end - start
	print("training time: " + str(training_time))
	#test the model
	print("train")
	qda_classifier(x_train_setosa, x_train_versicolor, x_train_virginica, setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance)
	print("test")
	qda_classifier(x_test_setosa, x_test_versicolor, x_test_virginica, setosa_mean, versicolor_mean, virginica_mean, setosa_covariance, versicolor_covariance, virginica_covariance)

#read the data
url = "http://www.cse.scu.edu/~yfang/coen140/iris.data"
df = pd.read_csv(url, header = None)
#split the data into training and testing data
x_train_setosa = df.iloc[0:40, 0:4]
x_train_versicolor = df.iloc[50:90, 0:4]
x_train_virginica = df.iloc[100:140, 0:4]
x_test_setosa = df.iloc[40:50, 0:4]
x_test_versicolor = df.iloc[90:100, 0:4]
x_test_virginica = df.iloc[140:150, 0:4]
y_train = df.iloc[np.r_[0:40, 50:90, 100:140], 4]
y_test = df.iloc[np.r_[40:50, 90:100, 140:150], 4]
print("LDA")
lda_model(x_train_setosa, x_train_versicolor, x_train_virginica, x_test_setosa, x_test_versicolor, x_test_virginica)
print("QDA")
qda_model(x_train_setosa, x_train_versicolor, x_train_virginica, x_test_setosa, x_test_versicolor, x_test_virginica)