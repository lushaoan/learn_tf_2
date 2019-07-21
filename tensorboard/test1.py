#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.14'
__copyright__ = 'Copyright 2019, PI'


import tensorflow as tf
import cv2


summary_writer = tf.summary.create_file_writer('./')
with summary_writer.as_default():
    img = cv2.imread('6.png', flags=cv2.IMREAD_GRAYSCALE)
    img = tf.reshape(img, [1, 648, 1152, 1])
    tf.summary.image('img', img, 100)
    tf.summary.scalar('data', 123, 23)