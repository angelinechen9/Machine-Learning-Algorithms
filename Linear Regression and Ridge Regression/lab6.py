import pandas as pd
import numpy as np
import math

#calculate RMSE
def RMSE(y_pred, y, n):
	sum = 0
	for i in range(n):
		sum += (y_pred[i] - y[i]) ** 2
	return math.sqrt(sum / n)

#use the value of lambda to calculate RMSE
def ridge_regression_cross_validation(x_train, y_train, x_test, y_test, l):
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	#train the model
	w = np.linalg.inv(x_train.transpose().dot(x_train) + l * np.identity(96)).dot(x_train.transpose().dot(y_train))
	w = w.reshape(96, 1)
	#test the model
	y_test_pred = w.transpose().dot(x_test.transpose())
	y_test_pred = y_test_pred.tolist()[0]
	return RMSE(y_test_pred, y_test, 319)

#use 5-fold cross validation to optimize the value of lambda
def cross_validation(x_train, y_train):
	#the values of lambda
	ls = [400 * (1 / 2) ** (n - 1) for n in range(1, 11)]
	RMSE = []
	for i in range(10):
		l = ls[i]
		sum = 0
		#split the training data into 5 parts
		x_train_cross_validation = x_train.iloc[np.r_[0:319, 319:638, 638:957, 957:1276], :]
		y_train_cross_validation = y_train.iloc[np.r_[0:319, 319:638, 638:957, 957:1276]]
		x_test_cross_validation = x_train.iloc[1276:1595, :]
		y_test_cross_validation = y_train.iloc[1276:1595]
		sum += ridge_regression_cross_validation(x_train_cross_validation, y_train_cross_validation, x_test_cross_validation, y_test_cross_validation, l)
		x_train_cross_validation = x_train.iloc[np.r_[0:319, 319:638, 638:957, 1276:1595], :]
		y_train_cross_validation = y_train.iloc[np.r_[0:319, 319:638, 638:957, 1276:1595]]
		x_test_cross_validation = x_train.iloc[957:1276, :]
		y_test_cross_validation = y_train.iloc[957:1276]
		sum += ridge_regression_cross_validation(x_train_cross_validation, y_train_cross_validation, x_test_cross_validation, y_test_cross_validation, l)
		x_train_cross_validation = x_train.iloc[np.r_[0:319, 319:638, 957:1276, 1276:1595], :]
		y_train_cross_validation = y_train.iloc[np.r_[0:319, 319:638, 957:1276, 1276:1595]]
		x_test_cross_validation = x_train.iloc[638:957, :]
		y_test_cross_validation = y_train.iloc[638:957]
		sum += ridge_regression_cross_validation(x_train_cross_validation, y_train_cross_validation, x_test_cross_validation, y_test_cross_validation, l)
		x_train_cross_validation = x_train.iloc[np.r_[0:319, 638:957, 957:1276, 1276:1595], :]
		y_train_cross_validation = y_train.iloc[np.r_[0:319, 638:957, 957:1276, 1276:1595]]
		x_test_cross_validation = x_train.iloc[319:638, :]
		y_test_cross_validation = y_train.iloc[319:638]
		sum += ridge_regression_cross_validation(x_train_cross_validation, y_train_cross_validation, x_test_cross_validation, y_test_cross_validation, l)
		x_train_cross_validation = x_train.iloc[np.r_[319:638, 638:957, 957:1276, 1276:1595], :]
		y_train_cross_validation = y_train.iloc[np.r_[319:638, 638:957, 957:1276, 1276:1595]]
		x_test_cross_validation = x_train.iloc[0:319, :]
		y_test_cross_validation = y_train.iloc[0:319]
		sum += ridge_regression_cross_validation(x_train_cross_validation, y_train_cross_validation, x_test_cross_validation, y_test_cross_validation, l)
		RMSE.append(sum / 5)
	#find the value of lambda that minimizes RMSE
	return ls[RMSE.index(min(RMSE))]

#use gradient descent to calculate the value of w for linear regression
def linear_regression_gradient_descent(x, y):
	w0 = np.random.normal(0, 1, 96).reshape(96, 1)
	w1 = w0 + 10 ** (-5) * x.transpose().dot(y - x.dot(w0))
	while (np.linalg.norm(w0 - w1) > 10 ** (-5)):
		w0 = w1
		w1 = w1 + 10 ** (-5) * x.transpose().dot(y - x.dot(w1))
	return w1

#use gradient descent to calculate the value of w for ridge regression
def ridge_regression_gradient_descent(x, y, l):
	w0 = np.random.normal(0, 1, 96).reshape(96, 1)
	w1 = w0 + 10 ** (-5) * (x.transpose().dot(y - x.dot(w0)) - l * w0)
	while (np.linalg.norm(w0 - w1) > 10 ** (-5)):
		w0 = w1
		w1 = w1 + 10 ** (-5) * (x.transpose().dot(y - x.dot(w1)) - l * w1)
	return w1

def problem1(x_train, y_train, x_test, y_test):
	l = cross_validation(x_train, y_train)
	print(l)
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train).reshape(1595, 1)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test).reshape(399, 1)
	#train the model
	w = np.linalg.inv(x_train.transpose().dot(x_train) + l * np.identity(96)).dot(x_train.transpose().dot(y_train))
	#test the model
	y_train_pred = w.transpose().dot(x_train.transpose())
	y_train_pred = y_train_pred.tolist()[0]
	print("train")
	print(RMSE(y_train_pred, y_train, 1595))
	y_test_pred = w.transpose().dot(x_test.transpose())
	y_test_pred = y_test_pred.tolist()[0]

def problem2(x_train, y_train, x_test, y_test):
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train).reshape(1595, 1)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test).reshape(399, 1)
	#train the model
	w = linear_regression_gradient_descent(x_train, y_train)
	#test the model
	y_train_pred = w.transpose().dot(x_train.transpose())
	y_train_pred = y_train_pred.tolist()[0]
	print("train")
	print(RMSE(y_train_pred, y_train, 1595))
	y_test_pred = w.transpose().dot(x_test.transpose())
	y_test_pred = y_test_pred.tolist()[0]
	print("test")
	print(RMSE(y_train_pred, y_train, 399))

def problem3(x_train, y_train, x_test, y_test):
	l = cross_validation(x_train, y_train)
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train).reshape(1595, 1)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test).reshape(399, 1)
	#train the model
	w = ridge_regression_gradient_descent(x_train, y_train, l)
	#test the model
	y_train_pred = w.transpose().dot(x_train.transpose())
	y_train_pred = y_train_pred.tolist()[0]
	print("train")
	print(RMSE(y_train_pred, y_train, 1595))
	y_test_pred = w.transpose().dot(x_test.transpose())
	y_test_pred = y_test_pred.tolist()[0]
	print("test")
	print(RMSE(y_train_pred, y_train, 399))

url = "http://www.cse.scu.edu/~yfang/coen140/crime-train.txt"
df = pd.read_csv(url, sep = "\t")
x_train = df.iloc[:, 1:97]
x_train["dummy"] = 1
y_train = df.iloc[:, 0]
url = "http://www.cse.scu.edu/~yfang/coen140/crime-test.txt"
df = pd.read_csv(url, sep = "\t")
x_test = df.iloc[:, 1:97]
x_test["dummy"] = 1
y_test = df.iloc[:, 0]
print("ridge regression using the closed form solution")
problem1(x_train, y_train, x_test, y_test)
print("linear regression")
problem2(x_train, y_train, x_test, y_test)
print("ridge regression using the gradient descent algorithm")
problem3(x_train, y_train, x_test, y_test)