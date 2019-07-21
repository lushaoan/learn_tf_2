'''
fashion mnist
'''
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import metrics
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.keras.applications.resnet50 import ResNet50

print(keras.__version__)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
# cv2.imshow('img', train_images[0])
# cv2.waitKey(0)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=128, activation=keras.activations.relu),
    keras.layers.Dense(units=10, activation=keras.activations.softmax)
])
print(model.get_config())
