import argparse

import cv2
import numpy as np

from keras.callbacks import LambdaCallback, EarlyStopping, TerminateOnNaN, TensorBoard, ReduceLROnPlateau

import feature_utility as fu
import myVGG
import pickle
from model.models import bde_model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras
from scipy.stats import rankdata

parser = argparse.ArgumentParser(description=("Model training process."))
# parser.add_argument('data_path', help=("The path of training data set"))
parser.add_argument('--weights_path', default='my_model_weights.h5')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--class_weights', dest='class_weights', action='store_true')
parser.add_argument('--lr', default=0.0005)
parser.add_argument('--model', default='bde', choices=['bde', 'vgg'])


args = parser.parse_args()

class_weights_default = {
    0: 1.980382,
    1: 4.183682,
    2: 1.947025,
    3: 1.384374,
    4: 1.775864,
    5: 2.193581,
    6: 1.756148
}



def main():
    if args.model == 'bde':
        model = bde_model((48, 48, 1), lr=args.lr) #myVGG.VGG_16()
        shape = (48, 48, 1)
    else:
        shape = (1, 48, 48)
        model = myVGG.VGG_16()
    batch_size = 128
    epochs = 500
    num_classes = len(class_weights_default)

    X_train, y_train, X_test, y_test = pickle.load(open('/home/demo/anchormen/emotion-rec/data/preprocessed.p', 'rb'))



    X_train = np.reshape(X_train, (X_train.shape[0],) + shape) / 255.
    X_test = np.reshape(X_test, (X_test.shape[0],) + shape) / 255.

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if args.test:
        model.load_weights(args.weights_path)
        model.evaluate(X_test, y_test)
        return


    train_generator = ImageDataGenerator(
        rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,
        shear_range=0.2
    )

    #input_path = args.data_path
    #print("training data path : " + input_path)
    #X_train, y_train = fu.extract_features(input_path)

    print("Training started")

    callbacks = []
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    #batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))
    #epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print("epoch:", epoch))
    callbacks.append(earlystop_callback)
    #callbacks.append(batch_print_callback)
    #callbacks.append(epoch_print_callback)

    callbacks = [
        TerminateOnNaN(),
        TensorBoard('./tblogs', batch_size=batch_size),
        ReduceLROnPlateau(patience=50)
    ]

    model.fit_generator(
        train_generator.flow(X_train, y_train), len(X_train) // batch_size + 1, epochs=epochs, callbacks=callbacks,
        validation_data=(X_test, y_test), class_weight=class_weights_default if args.class_weights else None
    )

    model.save_weights(args.weights_path)
    scores = model.evaluate(X_train, y_train, verbose=0)
    print ("Train loss : %.3f" % scores[0])
    print ("Train accuracy : %.3f" % scores[1])
    print ("Training finished")

if __name__ == "__main__":
    main()
