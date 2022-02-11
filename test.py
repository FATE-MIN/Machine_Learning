import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28))
y_train = y_train.reshape((-1, 28, 28))

