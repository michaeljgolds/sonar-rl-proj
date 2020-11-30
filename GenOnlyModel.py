from __future__ import print_function, division
from tkinter import *
from tkinter import ttk
import tkinter as tk

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
tf.compat.v1.disable_eager_execution()

#from keras.datasets import mnist
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Concatenate, Add, Input, Dense, Reshape, Flatten, Dropout, Embedding, Lambda, UpSampling2D, Conv2D, Conv1D, UpSampling1D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error, categorical_crossentropy,sparse_categorical_crossentropy

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

import scipy.io

import math
batch_n = 48



# This is the generator from the GAN stripped down for efficiency. 
# It can make the network and load the weights from a weights file.

class ARGAN():
    def __init__(self):
        self.latent_dim = 50
        
        self.generator = self.build_generator()

    def build_generator(self):

       
        leaf_type = Input(shape=(1,))
        leaf_size = Input(shape=(1,))
        leaf_azim = Input(shape=(1,))
        leaf_elev = Input(shape=(1,))
        noise = Input(shape=(self.latent_dim,))
        li = Embedding(4,15)(leaf_type)
        li = Dense(15,activation="relu")(li)
        li = BatchNormalization(momentum=0.8)(li)
        li = Dense(15,activation="relu")(li)
        li = BatchNormalization(momentum=0.8)(li)
        li = Dense(15,activation="relu")(li)
        li = BatchNormalization(momentum=0.8)(li)
        li = Reshape((15,1))(li)
        
        li2 = Dense(10,activation="relu")(leaf_size)
        li2 = BatchNormalization(momentum=0.8)(li2)
        li2 = Dense(10,activation="relu")(li2)
        li2 = BatchNormalization(momentum=0.8)(li2)
        li2 = Reshape((10,1))(li2)
        li3 = Dense(15,activation="relu")(leaf_azim)
        li3 = BatchNormalization(momentum=0.8)(li3)
        li3 = Dense(15,activation="relu")(li3)
        li3 = BatchNormalization(momentum=0.8)(li3)
        li3 = Dense(15,activation="relu")(li3)
        li3 = BatchNormalization(momentum=0.8)(li3)
        li3 = Dense(15,activation="relu")(li3)
        li3 = BatchNormalization(momentum=0.8)(li3)
        li3 = Reshape((15,1))(li3)
        li4 = Dense(10,activation="relu")(leaf_elev)
        li4 = BatchNormalization(momentum=0.8)(li4)
        li4 = Dense(10,activation="relu")(li4)
        li4 = BatchNormalization(momentum=0.8)(li4)
        li4 = Reshape((10,1))(li4)
        noise_int = Dense(50)(noise)
        noise_int = Reshape((50,1))(noise_int)
        merge = Concatenate(axis=1)([li,noise_int])
        merge = Concatenate(axis=1)([merge,li2])
        merge = Concatenate(axis=1)([merge,li3])
        merge = Concatenate(axis=1)([merge,li4])
        merge = Flatten()(merge)
        
        # gen_temp=Dense(100)(merge)
        # gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        # gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        # gen_temp = Dense(100)(gen_temp)
        # gen = Add()([merge,gen_temp])
        # gen = LeakyReLU(alpha=0.2)(gen)
        # gen = BatchNormalization(momentum=0.8)(gen)
        
        # gen_temp=Dense(100)(merge)
        # # gen_temp=Dense(100)(gen)
        # gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        # gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        # gen_temp = Dense(100)(gen_temp)
        # gen = Add()([merge,gen_temp])
        # # gen = Add()([gen,gen_temp])
        # gen = LeakyReLU(alpha=0.2)(gen)
        # gen = BatchNormalization(momentum=0.8)(gen)
        
        gen_temp=Dense(100)(merge)
        # gen_temp=Dense(100)(gen)
        gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        gen_temp = Dense(100)(gen_temp)
        gen = Add()([merge,gen_temp])
        # gen = Add()([gen,gen_temp])
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)
        
        gen_temp = Dense(200)(gen)
        gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        gen_temp = Dense(200)(gen_temp)
        gen = Reshape((100,1))(gen)
        gen = UpSampling1D()(gen)
        gen = Reshape((1,200))(gen)
        gen = Add()([gen,gen_temp])
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)
        
        
        gen_temp = Dense(200)(gen)
        gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        gen_temp = Dense(200)(gen_temp)
        gen = Add()([gen,gen_temp])
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)
        
        gen_temp = Dense(200)(gen)
        gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        gen_temp = Dense(200)(gen_temp)
        gen = Add()([gen,gen_temp])
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)
        
        gen_temp = Dense(400)(gen)
        gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        gen_temp = Dense(400)(gen_temp)
        gen = Reshape((200,1))(gen)
        gen = UpSampling1D()(gen)
        gen = Reshape((1,400))(gen)
        gen = Add()([gen,gen_temp])
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)
        
        gen_temp = Dense(400)(gen)
        gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        gen_temp = Dense(400)(gen_temp)
        #gen = UpSampling1D()(gen)
        gen = Add()([gen,gen_temp])
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)
        
        gen_temp = Dense(400)(gen)
        gen_temp = LeakyReLU(alpha=0.2)(gen_temp)
        gen_temp = BatchNormalization(momentum=0.8)(gen_temp)
        gen_temp = Dense(400)(gen_temp)
        #gen = UpSampling1D()(gen)
        gen = Add()([gen,gen_temp])
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = BatchNormalization(momentum=0.8)(gen)
        
        gen = Dense(400,activation='tanh')(gen)
        
        out_layer = Reshape((400,1))(gen)
        model = Model(inputs=[noise,leaf_type,leaf_size,leaf_azim,leaf_elev], outputs=out_layer)
        model.summary()
        
        return model
 
    def load_weights(self,epoch_num=''):
        self.generator.load_weights('./weights/my_generator'+epoch_num)

    def close(self):
        import gc
        del self.generator
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()