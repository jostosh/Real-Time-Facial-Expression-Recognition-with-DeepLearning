from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import tensorflow as tf
from keras.optimizers import Adam

def bde_model(shape, lr=0.0005, num_classes=7, weights_path=None):

    act = tf.nn.elu
    model = Sequential()

    model.add(Convolution2D(32, kernel_size=3, strides=1, activation=act, input_shape=shape, padding='same'))
    model.add(Convolution2D(32, kernel_size=3, strides=1, activation=act, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, kernel_size=3, strides=1, activation=act, padding='same'))
    model.add(Convolution2D(64, kernel_size=3, strides=1, activation=act, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation=act))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=act))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def bde_adience(shape, lr=0.0005, num_classes=7, weights_path=None):

    act = tf.nn.elu
    model = Sequential()

    model.add(Convolution2D(32, kernel_size=5, strides=3, activation=act, input_shape=shape, padding='same'))
    model.add(Convolution2D(32, kernel_size=5, strides=3, activation=act, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, kernel_size=3, strides=1, activation=act, padding='same'))
    model.add(Convolution2D(64, kernel_size=3, strides=1, activation=act, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, kernel_size=3, strides=1, activation=act, padding='same'))
    model.add(Convolution2D(128, kernel_size=3, strides=1, activation=act, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation=act))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=act))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
