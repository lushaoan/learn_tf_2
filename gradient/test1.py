#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.15'
__copyright__ = 'Copyright 2019, PI'


import tensorflow as tf


w1 = tf.Variable(2.0)
w2 = tf.Variable(3.0)

def weighted_sum(x1, x2):
    return w1 *x1 + w2 * x2

# s = weighted_sum(5., 7.)
# print(s.numpy()) # 31

with tf.GradientTape() as tape:
    s = weighted_sum(5., 7.)

    [w1_grad] = tape.gradient(s, [w1])
    print(w1_grad.numpy())

print('*'*20)

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)      #tensor: a Tensor or list of Tensors.
    y = x * x
dy_dx = g.gradient(y, x)#6
print(dy_dx)

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x
    dy_dx = gg.gradient(y, x)     # Will compute to 6.0
d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
print(d2y_dx2)

print('*'*20)

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = g.gradient(y, x)  # 6.0
print(dz_dx)
del g