from sklearn.datasets import load_boston
from sklearn import linear_model
import numpy as np

boston = load_boston()

test_dataset = boston.data[[10, 100, 300]]

#train_data = np.delete(boston.data, [10, 100, 300], axis = 0)
#train_target = np.delete(boston.target, [10, 100, 300])

#print(test_dataset)
#print(boston.feature_names)
#print(train_data[:10])
#print(train_target[:10])


clf = linear_model.LinearRegression()
clf.fit(boston.data, boston.target)
predictions = clf.predict(test_dataset)

print boston.target[[10, 100, 300]]
#print(train_target[[10, 100, 300]])
