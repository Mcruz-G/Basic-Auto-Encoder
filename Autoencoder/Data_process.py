## Function to normalize data features
## returns normalized and flatten data of x_dim*y_dim dimension

import numpy as np

def data_prep(data):

    data = data.astype('float32')/255
    data = data.reshape(len(data), np.prod(data.shape[1:]))

    return data
