import pandas as pd
import numpy as np
import math

#calculate the output of the sigmoid function
def sigmoid_function(r):
	return 1 / (1 + math.exp(-r))

v_sigmoid_function = np.vectorize(sigmoid_function)

#use gradient descent to calculate the value of w for logistic regression
def gradient_descent(x, y):
	w0 = np.random.normal(0, 1, 57).reshape(57, 1)
	w1 = w0 + 10 ** (-5) * x.transpose().dot(y - v_sigmoid_function(x.dot(w0)).flatten()).reshape(57, 1)
	while (np.linalg.norm(w0 - w1) > 10 ** (-5)):
		w0 = w1
		w1 = w1 + 10 ** (-5) * x.transpose().dot(y - v_sigmoid_function(x.dot(w1)).flatten()).reshape(57, 1)
	return w1

def logistic_regression(x_train, y_train, x_test, y_test):
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	#train the model
	w = gradient_descent(x_train, y_train)
	#test the model
	count = 0
	for i in range(len(x_train)):
		r = w.transpose().dot(x_train[i])
		if (sigmoid_function(r) < .5):
			if (y_train[i] == 0):
				count += 1
		else:
			if (y_train[i] == 1):
				count += 1
	print("train")
	accuracy = count / len(x_train)
	print("accuracy: " + str(accuracy * 100) + "%")
	error = 1 - accuracy
	print("error: " + str(error * 100) + "%")
	count = 0
	for i in range(len(x_test)):
		r = w.transpose().dot(x_test[i])
		if (sigmoid_function(r) < .5):
			if (y_test[i] == 0):
				count += 1
		else:
			if (y_test[i] == 1):
				count += 1
	print("test")
	accuracy = count / len(x_test)
	print("accuracy: " + str(accuracy * 100) + "%")
	error = 1 - accuracy
	print("error: " + str(error * 100) + "%")

df = pd.read_csv("spam-train.csv")
x_train = df.iloc[:, 1:57]
#normalize the features
x_train = (x_train - x_train.mean()) / x_train.std()
x_train["dummy"] = 1
y_train = df.iloc[:, 57]
df = pd.read_csv("spam-test.csv")
x_test = df.iloc[:, 1:57]
#normalize the features
x_test = (x_test - x_test.mean()) / x_test.std()
x_test["dummy"] = 1
y_test = df.iloc[:, 57]
logistic_regression(x_train, y_train, x_test, y_test)