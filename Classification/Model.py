import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

from keras.layers import LeakyReLU

def fault_classifier():

    model = Sequential()
  
    model.add(Dense(128, activation='relu',input_shape=(21,),
                    kernel_regularizer=regularizers.l2(0.001),bias_regularizer=regularizers.l1(0.01)))
    
    model.add(Dense(256,  kernel_regularizer=regularizers.l2(0.001),bias_regularizer=regularizers.l1(0.01)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l1(0.01)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001),bias_regularizer=regularizers.l1(0.01)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Dense(7, activation='softmax'))
    
    model.compile(keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                    loss = "categorical_crossentropy", metrics = ['acc'])
    
    return model