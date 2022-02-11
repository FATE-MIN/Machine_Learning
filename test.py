import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv2d = tf.keras.layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), kernel_size=(3, 3), filters=32, padding='same', activation='relu', strides=(2, 2))
        self.pool2d = self.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    def call(self, inputs):
        y = self.conv2d(inputs)
        y = self.pool2d(y)
        y = self.flatten(y)
        y = self.dense1(y)
        outputs = self.dense2(y)
        return outputs
    

