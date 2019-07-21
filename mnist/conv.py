#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.22'
__copyright__ = 'Copyright 2019, PI'


import tensorflow as tf
import numpy as np
from tensorflow.python import keras
import cv2


def mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    


if __name__ == '__main__':
    mnist_dataset()