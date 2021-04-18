from keras.datasets import mnist
import numpy as np

from helper import convert_to_one_hot, model

i = 2
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_y_oh = convert_to_one_hot(train_y, 10).T
test_y_oh = convert_to_one_hot(test_y, 10).T

train_X = np.array(train_X)
test_X = np.array(test_X)

train_X_shape = train_X.shape
test_X_shape = test_X.shape
print(train_X_shape[0])

train_X = train_X.reshape([60000, 28, 28, 1])
test_X = test_X.reshape([test_X_shape[0], 28, 28, 1])
# test_X = np.array(test_X).reshape([28,28,1])
_, _, parameters = model(train_X, train_y_oh, test_X, test_y_oh, 0.001, 20)
