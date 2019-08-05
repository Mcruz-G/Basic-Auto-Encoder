from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Data_process import data_prep
from Visualize import view

# MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()
X_train = data_prep(X_train)
X_test = data_prep(X_test)
X_train_noisy = X_train + np.random.normal(loc=0.0, scale=0.5, size=X_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = X_test + np.random.normal(loc=0.0, scale=0.5, size=X_test.shape)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

##### Neural Network Architecture #####

inputImg= Input(shape=(784,))
h1 = Dense(units = 128, activation = 'relu')(inputImg)
h2 = Dense(units = 64, activation = 'relu')(h1)
latentLayer = Dense(units = 32, activation = 'relu')(h2)
decodeH1 = Dense(units=64, activation='relu')(latentLayer)
decodeH2 = Dense(units = 128, activation = 'relu')(decodeH1)
decodedImg = Dense(units = 784, activation = 'sigmoid')(decodeH2)

##### Neural Network Architecture #####

# Instantiate both autoencoder and encoder
autoEncoder = Model(inputImg, decodedImg)
encoder = Model(inputImg, latentLayer)

# Compile models

autoEncoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit autoencoder

autoEncoder.fit(X_train_noisy, X_train_noisy,epochs=20,batch_size=256,shuffle=True,validation_data=(X_test_noisy, X_test_noisy))

# Create memory
encodedImg= encoder.predict(X_test_noisy)
predictedImg = autoEncoder.predict(X_test_noisy)

##### Visualize original - encoded - decoded images


view(X_test_noisy,encodedImg,predictedImg)