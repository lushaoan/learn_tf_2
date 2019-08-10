#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.07.26'
__copyright__ = 'Copyright 2019, PI'


import numpy as np
import tensorflow as tf
from tensorflow.python import keras
# import tensorflow.python.keras.backend.K as K
# from tensorflow.keras.layers import *
# from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K


def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    # if data_format == 'default':
    #     data_format = keras.backend.K.image_data_format()
    # if data_format == 'channels_first':
    #     original_shape = keras.backend.K.int_shape(X)
    #     if target_height and target_width:
    #         new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
    #     else:
    #         new_shape = tf.shape(X)[2:]
    #         new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
    #     X = permute_dimensions(X, [0, 2, 3, 1])
    #     X = tf.image.resize_bilinear(X, new_shape)
    #     X = permute_dimensions(X, [0, 3, 1, 2])
    #     if target_height and target_width:
    #         X.set_shape((None, None, target_height, target_width))
    #     else:
    #         X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
    #     return X
    # elif data_format == 'channels_last':
    original_shape = K.int_shape(X)
    if target_height and target_width:
        new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
    else:
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
    X = tf.image.resize(X, new_shape)

    if target_height and target_width:
        X.set_shape((None, target_height, target_width, None))
    else:
        X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
    return X
    # else:
    #     raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling2D(keras.layers.Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        # if data_format == 'default':
        #     data_format = keras.backend.K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        # assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        # self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_first':
        #     width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
        #     height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
        #     if self.target_size is not None:
        #         width = self.target_size[0]
        #         height = self.target_size[1]
        #     return (input_shape[0],
        #             input_shape[1],
        #             width,
        #             height)
        # elif self.data_format == 'channels_last':
        width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
        height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
        if self.target_size is not None:
            width = self.target_size[0]
            height = self.target_size[1]
        return (input_shape[0],
                width,
                height,
                input_shape[3])
        # else:
        #     raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def AtrousFCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    # if batch_shape:
    #     img_input = keras.layers.Input(batch_shape=batch_shape)
    #     image_size = batch_shape[1:3]
    # else:
    #     img_input = keras.layers.Input(shape=input_shape)
    #     image_size = input_shape[0:2]
    # Block 1
    img_input = keras.Input(shape=(320,320,3), batch_size=2)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=keras.regularizers.l2(weight_decay))(img_input)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    pool4_res = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=keras.regularizers.l2(weight_decay))(pool4_res)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = keras.layers.Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2),
                      name='fc1', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.Dropout(0.5)(x)
    #classifying layer
    x = keras.layers.Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=keras.regularizers.l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=(20,20))(x)

    x = keras.layers.concatenate([pool4_res, x])
    x = BilinearUpSampling2D(target_size=(320,320))(x)

    model = tf.keras.Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

# model = AtrousFCN_Vgg16_16s()
# model.summary()
# tf.keras.utils.plot_model(model=model, to_file='model.png', show_layer_names=True, show_shapes=True)
# bce = keras.losses.CategoricalCrossentropy()
# bce = keras.losses.BinaryCrossentropy()
# # bce = keras.losses.SparseCategoricalCrossentropy()
# # a = tf.constant([2])
# # b = tf.constant([[0.3,0.3,0.4]])
# loss = bce(y_true=[1.,0],
#            y_pred=[0.,1])
# print(loss.numpy)

# a = [[0., 0., 1.], [0.3, 0.4, 0.3],[0.1,0.2,0.7]]
# b = [[0.3,0.3,0.4],[ 0.,  1., 0.], [ 1., 0., 0.]]

# scc = keras.losses.SparseCategoricalCrossentropy()
# loss = scc()
# print(np.math.log(0, 1))
# print(np.math.log(1, 0))
# print(np.math.log(0.4, 2))
