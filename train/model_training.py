import argparse
import numpy as np

from keras.callbacks import TerminateOnNaN, TensorBoard, ReduceLROnPlateau

import pickle
from train.models import bde_model, bde_adience
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


parser = argparse.ArgumentParser(description=("Model training process."))
# parser.add_argument('data_path', help=("The path of training data set"))
parser.add_argument('--weights_path', default='my_model_weights.h5')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--model', default='bde', choices=['bde', 'vgg', 'adience'])
parser.add_argument('--data', default='preprocessed.p')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_classes', default=7, type=int)


args = parser.parse_args()

def get_model(num_classes):
    if args.model == 'adience':
        return bde_adience((227, 227, 3), num_classes=num_classes)
    return bde_model((48, 48, 1), args.lr, num_classes=num_classes)




def main():

    if args.model == 'bde':
        shape = (48, 48, 1)
    elif args.model == 'adience':
        shape = (227, 227, 3)
    batch_size = args.batch_size
    epochs = args.epochs

    X_train, y_train, X_test, y_test = pickle.load(
        open('/home/demo/anchormen/emotion-rec/data/{}'.format(args.data), 'rb')
    )

    if len(y_train.shape) == 2:
        num_classes = y_train.shape[1]
    else:
        num_classes = args.num_classes

    X_train = np.reshape(X_train, (X_train.shape[0],) + shape) / 255.
    X_test = np.reshape(X_test, (X_test.shape[0],) + shape) / 255.

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)


    model = get_model(num_classes)

    if args.test:
        model.load_weights(args.weights_path)
        model.evaluate(X_test, y_test)
        return


    train_generator = ImageDataGenerator(
        rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,
        shear_range=0.2
    )

    print("Training started")

    callbacks = [
        TerminateOnNaN(),
        TensorBoard('./tblogs', batch_size=batch_size),
        ReduceLROnPlateau(patience=50)
    ]

    model.fit_generator(
        train_generator.flow(X_train, y_train), len(X_train) // batch_size + 1, epochs=epochs, callbacks=callbacks,
        validation_data=(X_test, y_test)
    )

    #train.save_weights(args.weights_path)

    model.save(args.weights_path)

    scores = model.evaluate(X_train, y_train, verbose=0)
    print ("Train loss : %.3f" % scores[0])
    print ("Train accuracy : %.3f" % scores[1])
    print ("Training finished")

if __name__ == "__main__":
    main()
