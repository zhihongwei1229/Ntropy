from keras.datasets import mnist
import numpy as np
from helper import convert_to_one_hot, model


(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_x = np.array(train_X)
test_x = np.array(test_X)
train_x_left = np.array([[row[0: 14] for row in sample] for sample in train_x])
train_x_right = np.array([[row[14:] for row in sample] for sample in train_x])
test_x_left = np.array([[row[0: 14] for row in sample] for sample in test_x])
test_x_right = np.array([[row[14:] for row in sample] for sample in test_x])
new_train_X = np.concatenate((train_x_left, train_x_right), axis=0)
new_test_X = np.concatenate((test_x_left, test_x_right), axis=0)

# print(train_x_left.shape)
train_padding = np.zeros((train_x_left.shape[0], 28, 14))
test_padding = np.zeros((test_x_left.shape[0], 28, 14))
improved_train_x_left = np.concatenate((train_x_left, train_padding), axis=2)
improved_train_x_right = np.concatenate((train_padding, train_x_right), axis=2)
improved_test_x_left = np.concatenate((test_x_left, test_padding), axis=2)
improved_test_x_right = np.concatenate((test_padding, test_x_right), axis=2)
improved_train_X = np.concatenate((improved_train_x_left, improved_train_x_right), axis=0)
improved_test_X = np.concatenate((improved_test_x_left, improved_test_x_right), axis=0)


train_y_oh = convert_to_one_hot(train_y, 10).T
train_y_oh = np.concatenate((train_y_oh, train_y_oh), axis=0)
test_y_oh = convert_to_one_hot(test_y, 10).T
test_y_oh = np.concatenate((test_y_oh, test_y_oh), axis=0)

train_X_shape = new_train_X.shape
test_X_shape = new_test_X.shape

new_train_X = new_train_X.reshape([train_X_shape[0], 28, 14, 1])
new_test_X = new_test_X.reshape([test_X_shape[0], 28, 14, 1])
_, _, parameters = model(new_train_X, train_y_oh, new_test_X, test_y_oh, 0.001, 30)

improved_train_X_shape = improved_train_X.shape
improved_test_X_shape = improved_test_X.shape
improved_train_X = improved_train_X.reshape([improved_train_X_shape[0], 28, 28, 1])
improved_test_X = improved_test_X.reshape([improved_test_X_shape[0], 28, 28, 1])
_, _, parameters = model(improved_train_X, train_y_oh, improved_test_X, test_y_oh, 0.001, 30)

