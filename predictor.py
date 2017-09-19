from keras.models import Model
import numpy as np
import cv2


N_OVERSAMPLED = 10


class Predictor:

    def __init__(self, class_idx_to_name: dict, model: Model, size: tuple, grayscale: bool):
        self.class_idx_to_name = class_idx_to_name
        self.model = model
        self.n_classes = len(class_idx_to_name)
        self.size = size
        self.grayscale = grayscale

    def predict(self, bbs, im, oversample=True):
        """ Returns class names of most likely class and corresponding confidences """
        x = self._preprocess(im, bbs, oversample=oversample)
        predictions = self.model.predict(x)
        if oversample:
            n_faces = len(x) // N_OVERSAMPLED
            predictions = np.reshape(predictions, (n_faces, N_OVERSAMPLED, len(self.class_idx_to_name))).mean(axis=1)
        indices = np.argmax(predictions, axis=1)
        return [self.class_idx_to_name[i] for i in indices], np.max(predictions, axis=1)

    def _preprocess(self, im, bbs, oversample):
        """ Preprocessing consists of oversampling, resizing and color conversion """
        if oversample:
            x = np.concatenate([self.oversample(im, bb) for bb in bbs])
        else:
            x = np.stack([self._clipped_crop(im, bb[1], bb[3], bb[0], bb[2]) for bb in bbs])

        x = np.stack([cv2.resize(im, self.size) for im in x])
        if self.grayscale:
            x = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in x])
            x = np.expand_dims(x, axis=3)
        return x

    def oversample(self, frame, bb, offset_fac=20):
        """
        Samples a crop by also including mirrored versions and corner crops with a 5 % offset (factor 1/20)
        This means the output will be of length 10 * len(bb)
        """
        x0, y0, x1, y1 = bb
        h = bb[3] - bb[1]
        w = bb[2] - bb[0]
        dx = w // offset_fac
        dy = h // offset_fac
        crops = np.stack([
            self._clipped_crop(frame, y0 - dy, y1 - dy, x0 - dx, x1 - dx),  # top left
            self._clipped_crop(frame, y0 + dy, y1 + dy, x0 - dx, x1 - dx),  # bottom left
            self._clipped_crop(frame, y0 - dy, y1 - dy, x0 + dx, x1 + dx),  # top right
            self._clipped_crop(frame, y0 + dy, y1 + dy, x0 + dx, x1 + dx),  # bottom right
            self._clipped_crop(frame, y0, y1, x0, x1)                       # center
        ])
        return np.concatenate([crops, crops[:, :, ::-1]])  # mirror

    @staticmethod
    def _clipped_crop(frame, y0, y1, x0, x1):
        """ Creates a clipped crop, where we make sure we do not exceed the boundaries of the frame """
        crop_w = x1 - x0
        crop_h = y1 - y0
        h = frame.shape[0]
        w = frame.shape[1]

        # Check for y boundary
        y0 = max(y0, 0)
        y1 = y0 + crop_h
        y1 = min(y1, h)
        y0 = y1 - crop_h

        # Check for x boundary
        x0 = max(x0, 0)
        x1 = x0 + crop_w
        x1 = min(x1, w)
        x0 = x1 - crop_w

        return frame[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]