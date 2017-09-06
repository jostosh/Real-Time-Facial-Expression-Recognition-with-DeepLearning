import numpy as np
from keras.utils.np_utils import to_categorical


class DataSet:

    def __init__(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if abs(X.max() - 255.0) < 25.0:
            X /= 255.0
        self.X = X
        if len(y.shape) == 1:
            y = to_categorical(y)
        self.y = y
        self.num_classes = y.max() + 1