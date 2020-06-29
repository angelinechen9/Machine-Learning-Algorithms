import pandas as pd
import numpy as np
import math

#calculate RMSE
def RMSE(y_pred, y, n):
	sum = 0
	for i in range(n):
		sum += (y_pred[i] - y[i]) ** 2
	return math.sqrt(sum / n)

def problem1(x_train, y_train, x_test, y_test):
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train).reshape(1595, 1)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test).reshape(399, 1)
	#train the model
	w = np.linalg.inv(x_train.transpose().dot(x_train)).dot(x_train.transpose().dot(y_train))
	#test the model
	y_train_pred = w.transpose().dot(x_train.transpose())
	y_train_pred = y_train_pred.tolist()[0]
	print("train")
	print(RMSE(y_train_pred, y_train, 1595))
	y_test_pred = w.transpose().dot(x_test.transpose())
	y_test_pred = y_test_pred.tolist()[0]
	print("test")
	print(RMSE(y_train_pred, y_train, 399))

def problem2(x_train, y_train, x_test, y_test):
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train).reshape(1595, 1)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test).reshape(399, 1)
	#train the model
	w = np.linalg.inv(x_train.transpose().dot(x_train) + 100 * np.identity(96)).dot(x_train.transpose().dot(y_train))
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
print("linear regression")
problem1(x_train, y_train, x_test, y_test)
print("ridge regression")
problem2(x_train, y_train, x_test, y_test)