#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.25'
__copyright__ = 'Copyright 2019, PI'


import tensorflow as tf
from tensorflow.python import keras


dataset_path = keras.utils.get_file('auto-mpg.data',
                                   'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')