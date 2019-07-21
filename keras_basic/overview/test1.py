#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.17'
__copyright__ = 'Copyright 2019, PI'


import tensorflow as tf
# import tensorflow.python.keras as keras
import tensorflow.keras as keras


inputs = tf.keras.Input(shape=(784,), name='img')
h1 = keras.layers.Dense(32, activation='relu')(inputs)
h2 = keras.layers.Dense(32, activation='relu')(h1)
outputs = keras.layers.Dense(10, activation='softmax')(h2)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist model')

model.summary()
keras.utils.plot_model(model, 'mnist_model.png')
keras.utils.plot_model(model, 'model_info.png', show_shapes=True)