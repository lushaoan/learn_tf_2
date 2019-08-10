#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.29'
__copyright__ = 'Copyright 2019, PI'


import tensorflow as tf
from tensorflow.python import keras


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        self.add_loss
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
print(linear_layer.weights)
print(linear_layer.losses)