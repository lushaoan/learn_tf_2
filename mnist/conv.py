#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.22'
__copyright__ = 'Copyright 2019, PI'


import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.ops import summary_ops_v2
import cv2
import math
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K


def prepare_data(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int64)

    ##### using np.astype is wrong
    # image = image.astype(np.float32) / 255.0
    # label = label.astype(np.int64)

    return image, label

def mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))
    train_labels = tf.one_hot(train_labels, depth=10)
    test_labels = tf.one_hot(test_labels, depth=10)
    train_images, train_labels = prepare_data(image=train_images, label=train_labels)
    test_images, test_labels = prepare_data(image=test_images, label=test_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images[:-10000], train_labels[:-10000]))
    val_dataset = tf.data.Dataset.from_tensor_slices((train_images[-10000:], train_labels[-10000:]))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    train_dataset = train_dataset.shuffle(50000).batch(64)
    val_dataset = val_dataset.shuffle(10000).batch(64)

    return train_dataset, val_dataset, test_dataset


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        inputs = keras.Input(shape=(28,28,1), name='mnist_input_image')
        h1 = keras.layers.Conv2D(input_shape=((28, 28, 1)), filters=4, kernel_size=3, padding='same',
                                 activation=keras.activations.relu)(inputs)
        h2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(h1)
        h3 = keras.layers.Conv2D(filters=4, kernel_size=5, padding='same', activation=keras.activations.relu)(h2)
        h4 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(h3)
        h5 = keras.layers.Flatten()(h4)
        h6 = keras.layers.Dense(units=32, activation=keras.activations.relu)(h5)
        h7 = keras.layers.Dropout(rate=0.4)(h6)
        outputs = keras.layers.Dense(units=10)(h7)

        self.model = keras.Model(inputs, outputs, name='mnist_net')

    def call(self, inputs, training=None, mask=None):
        outputs = self.model(inputs)
        return outputs


def getModel():
    inputs = keras.Input(shape=(28,28,1), name='mnist_input_image')
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=keras.activations.relu,
                            kernel_regularizer=keras.regularizers.l2)(inputs)
    x = keras.layers.Conv2DTranspose(filters=10, kernel_size=3, padding='same', strides=3)(x)
    x = keras.layers.BatchNormalization(name='batch')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=32, activation=keras.activations.relu)(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    outputs = keras.layers.Dense(units=10, activation=keras.activations.softmax)(x)

    model = keras.Model(inputs, outputs, name='mnist_net')
    # model = keras.Model(inputs, x, name='mnist_net')
    return model


def lr_scheduler(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate


def MyCorssentropy(y_true, y_pred, e):
    # print(y_true, y_pred)
    # loss1 = K.categorical_crossentropy(y_true, y_pred)
    # loss2 = K.categorical_crossentropy (K.ones_like(y_pred)/10, y_pred)
    # return (1-e)*loss1 + e*loss2
    y_c = tf.cast(y_true, tf.float32)
    temp = tf.math.reduce_mean(y_c - y_pred)
    temp = temp * e
    return temp


if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = mnist_dataset()
    model = getModel()
    # model.summary()
    # keras.utils.plot_model(model=model, to_file='model.png', show_layer_names=True, show_shapes=True)
    # print('*'*20)
    # model = MyModel()
    # model.build(input_shape=(None, 28, 28, 1))
    # model.model.summary()
    # keras.utils.plot_model(model=model.model, to_file='model.png', show_layer_names=True, show_shapes=True)

    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # writer = tf.summary.create_file_writer('./')

    for epoch in range(0, 1):
        optimizer.learning_rate = 0.01*epoch
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                print(logits[0])
                loss_value = loss(y_true=y_batch_train, y_pred=logits)
                # loss_value = MyCorssentropy(y_true=y_batch_train, y_pred=logits, e=100000000)
            grads = tape.gradient(target=loss_value, sources=model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            accuracy(y_batch_train, logits)
            # optimizer.learning_rate = 0.01*epoch
            if step % 200 == 0:
                print('training loss', loss_value, optimizer.learning_rate)
                # with writer.as_default():
                #     tf.summary.scalar('loss', loss_value, step=step)
                #     tf.summary.image(name='image', data=x_batch_train, step=step)
        print('metric', accuracy.result())
        accuracy.reset_states()

    model.save('my_model.h5')


if __name__ == '__main2__':
    model = tf.keras.models.load_model('my_model.h5')
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    img = train_images[2]
    img = img.reshape((28,28,1))
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    print(res)
    # intermediate_output = keras.Model(inputs=model.input, outputs=model.get_layer(index=5).output)
    # res = intermediate_output.predict(img)
    #
    # plt.imshow(res[0][:,:,0])
    # plt.show()
