import numpy as np
import scipy.io as sio

# if not hasattr(np, 'unicode_'):
#     np.unicode_ = np.str_
def load_mat(path):

    data = sio.loadmat(path)

    X = np.squeeze(data['X'])

    Y = np.squeeze(data['Y'])


    return X,Y


