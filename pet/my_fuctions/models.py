# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name='last_layer')(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "sigmoid"
    outputs = layers.Dense(num_classes, activation=activation)(x)
    return keras.Model(inputs, outputs)

def make_model4(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=predictions)

def make_model4_1(input_shape, num_classes):
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    #x = keras.layers.Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)



def make_model5(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def make_model6(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def make_model7(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.NASNetLarge(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)


def make_model8(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def make_model9(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def make_model10(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def make_model11(input_shape, num_classes) :
    #input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #input_tensor = Input(shape=input_shape)
    input_tensor = keras.Input(shape=input_shape)
    base_model = keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet", input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

