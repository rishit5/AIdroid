from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np


iris = load_iris()

#testdata[0] =  iris.data[0]
#testdata[1] = iris.data[50]
#test.data[2] = iris.data[100]

test_data = iris.data[[0, 23, 67]]

train_data = np.delete(iris.data, [0, 50, 100], axis = 0)
train_feature = np.delete(iris.target, [0, 50, 100])

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_feature)

print clf.predict(test_data)
