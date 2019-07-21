from __future__ import absolute_import, division

import tensorflow as tf
from tensorflow.python import keras

x = [[2.]]
m = tf.matmul(x, x)
print(m)

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1

# fizzbuzz(15)

# keras.datasets.reuters.load_data()
# (mnist_images, mnist_labels), _ = keras.datasets.mnist.load_data()
# dataset = tf.data.Dataset.from_tensor_slices(
#   (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
#    tf.cast(mnist_labels,tf.int64)))
# dataset = dataset.shuffle(1000).batch(32)
