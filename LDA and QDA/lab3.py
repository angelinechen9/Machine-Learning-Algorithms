import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def lda_classifier(x_train, x_test, y_train, y_test):
	lda = LinearDiscriminantAnalysis(solver = "svd", store_covariance = True)
	#train the model
	lda_model = lda.fit(x_train.astype(np.float64), y_train)
	#test the model
	y_train_pred = lda_model.predict(x_train)
	y_test_pred = lda_model.predict(x_test)
	#calculate the train error
	train_count = 0
	for i in range(len(y_train)):
		if (y_train_pred[i] == y_train.values[i]):
			train_count += 1
	train_accuracy = train_count / len(y_train)
	print("train accuracy: " + str(train_accuracy * 100) + "%")
	train_error = 1 - train_accuracy
	print("train error: " + str(train_error * 100) + "%")
	#calculate the test error
	test_count = 0
	for i in range(len(y_test)):
		if (y_test_pred[i] == y_test.values[i]):
			test_count += 1
	test_accuracy = test_count / len(y_test)
	print("test accuracy: " + str(test_accuracy * 100) + "%")
	test_error = 1 - test_accuracy
	print("test error: " + str(test_error * 100) + "%")

def qda_classifier(x_train, x_test, y_train, y_test):
	qda = QuadraticDiscriminantAnalysis(store_covariance = True)
	#train the model
	qda_model = qda.fit(x_train.astype(np.float64), y_train)
	#test the model
	y_train_pred = qda_model.predict(x_train)
	y_test_pred = qda_model.predict(x_test)
	#calculate the train error
	train_count = 0
	for i in range(len(y_train)):
		if (y_train_pred[i] == y_train.values[i]):
			train_count += 1
	train_accuracy = train_count / len(y_train)
	print("train accuracy: " + str(train_accuracy * 100) + "%")
	train_error = 1 - train_accuracy
	print("train error: " + str(train_error * 100) + "%")
	#calculate the test error
	test_count = 0
	for i in range(len(y_test)):
		if (y_test_pred[i] == y_test.values[i]):
			test_count += 1
	test_accuracy = test_count / len(y_test)
	print("test accuracy: " + str(test_accuracy * 100) + "%")
	test_error = 1 - test_accuracy
	print("test error: " + str(test_error * 100) + "%")

#read the data
url = "http://www.cse.scu.edu/~yfang/coen140/iris.data"
df = pd.read_csv(url, header = None)
#split the data into training and testing data
x_train = df.iloc[np.r_[0:40, 50:90, 100:140], 0:4]
x_test = df.iloc[np.r_[40:50, 90:100, 140:150], 0:4]
y_train = df.iloc[np.r_[0:40, 50:90, 100:140], 4]
y_test = df.iloc[np.r_[40:50, 90:100, 140:150], 4]
#run the models on all of the features
print("sepal length, sepal width, petal length, petal width")
print("LDA")
lda_classifier(x_train, x_test, y_train, y_test)
print("QDA")
qda_classifier(x_train, x_test, y_train, y_test)
print("\n")
#split the data into training and testing data
x_train = df.iloc[np.r_[0:40, 50:90, 100:140], [1, 2, 3]]
x_test = df.iloc[np.r_[40:50, 90:100, 140:150], [1, 2, 3]]
y_train = df.iloc[np.r_[0:40, 50:90, 100:140], 4]
y_test = df.iloc[np.r_[40:50, 90:100, 140:150], 4]
#run the models on subsets of the features
print("sepal width, petal length, petal width")
print("LDA")
lda_classifier(x_train, x_test, y_train, y_test)
print("QDA")
qda_classifier(x_train, x_test, y_train, y_test)
print("\n")
#split the data into training and testing data
x_train = df.iloc[np.r_[0:40, 50:90, 100:140], [0, 2, 3]]
x_test = df.iloc[np.r_[40:50, 90:100, 140:150], [0, 2, 3]]
y_train = df.iloc[np.r_[0:40, 50:90, 100:140], 4]
y_test = df.iloc[np.r_[40:50, 90:100, 140:150], 4]
#run the models on subsets of the features
print("sepal length, petal length, petal width")
print("LDA")
lda_classifier(x_train, x_test, y_train, y_test)
print("QDA")
qda_classifier(x_train, x_test, y_train, y_test)
print("\n")
#split the data into training and testing data
x_train = df.iloc[np.r_[0:40, 50:90, 100:140], [0, 1, 3]]
x_test = df.iloc[np.r_[40:50, 90:100, 140:150], [0, 1, 3]]
y_train = df.iloc[np.r_[0:40, 50:90, 100:140], 4]
y_test = df.iloc[np.r_[40:50, 90:100, 140:150], 4]
#run the models on subsets of the features
print("sepal length, sepal width, petal width")
print("LDA")
lda_classifier(x_train, x_test, y_train, y_test)
print("QDA")
qda_classifier(x_train, x_test, y_train, y_test)
print("\n")
#split the data into training and testing data
x_train = df.iloc[np.r_[0:40, 50:90, 100:140], [0, 1, 2]]
x_test = df.iloc[np.r_[40:50, 90:100, 140:150], [0, 1, 2]]
y_train = df.iloc[np.r_[0:40, 50:90, 100:140], 4]
y_test = df.iloc[np.r_[40:50, 90:100, 140:150], 4]
#run the models on subsets of the features
print("sepal length, sepal width, petal length")
print("LDA")
lda_classifier(x_train, x_test, y_train, y_test)
print("QDA")
qda_classifier(x_train, x_test, y_train, y_test)