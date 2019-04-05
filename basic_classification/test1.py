import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import metrics
import numpy as np
import matplotlib.pyplot as plt
import cv2


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(type(train_images))
print(len(test_images))
print(test_labels[0])
cv2.imshow('img', test_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()