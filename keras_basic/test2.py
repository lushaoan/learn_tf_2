#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2019.05.05'
__copyright__ = 'Copyright 2019, PI'


from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from tensorflow.python.keras import Model


def simple_inception_model():
    input_img = Input(shape=(256, 256, 3))

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)


def simple_resnet():
    # 输入张量为 3 通道 256x256 图像
    x = Input(shape=(256, 256, 3))
    # 3 输出通道（与输入通道相同）的 3x3 卷积核
    y = Conv2D(3, (3, 3), padding='same')(x)
    # 返回 x + y
    z = keras.layers.add([x, y])


"""
共享视觉模型
该模型在两个输入上重复使用同一个图像处理模块，以判断两个 MNIST 数字是否为相同的数字。
"""
def shared_model():
    # 首先，定义视觉模型
    digit_input = Input(shape=(27, 27, 1))
    x = Conv2D(64, (3, 3))(digit_input)
    x = Conv2D(64, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    out = Flatten()(x)

    vision_model = Model(digit_input, out)

    # 然后，定义区分数字的模型
    digit_a = Input(shape=(27, 27, 1))
    digit_b = Input(shape=(27, 27, 1))

    # 视觉模型将被共享，包括权重和其他所有
    out_a = vision_model(digit_a)
    out_b = vision_model(digit_b)

    concatenated = keras.layers.concatenate([out_a, out_b])
    out = Dense(1, activation='sigmoid')(concatenated)

    classification_model = Model([digit_a, digit_b], out)


"""
如何「冻结」网络层？

「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。

您可以将 trainable 参数（布尔值）传递给一个层的构造器，以将该层设置为不可训练的：

frozen_layer = Dense(32, trainable=False)

另外，可以在实例化之后将网络层的 trainable 属性设置为 True 或 False。为了使之生效，在修改 trainable 属性之后，需要在模型上调用 compile()。这是一个例子：
"""
def fronzen():
    x = Input(shape=(32,))
    layer = Dense(32)
    layer.trainable = False
    y = layer(x)

    frozen_model = Model(x, y)
    # 在下面的模型中，训练期间不会更新层的权重
    frozen_model.compile(optimizer='rmsprop', loss='mse')

    layer.trainable = True
    trainable_model = Model(x, y)
    # 使用这个模型，训练期间 `layer` 的权重将被更新
    # (这也会影响上面的模型，因为它使用了同一个网络层实例)
    trainable_model.compile(optimizer='rmsprop', loss='mse')

    frozen_model.fit(data, labels)  # 这不会更新 `layer` 的权重
    trainable_model.fit(data, labels)  # 这会更新 `layer` 的权重

