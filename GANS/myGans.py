from __future__ import absolute_import


from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from generative import build_generator
from discriminator import build_discriminator
from dataPrep import data_prep
from GANS import GAN
import matplotlib.pyplot as plt

import sys

import numpy as np

### Get Data ####
X_train = data_prep()

#### Instantiate GAN  #####
myGan = GAN()
myGan.train(epochs=30000, batch_size=32, sample_interval=200,data = X_train)

