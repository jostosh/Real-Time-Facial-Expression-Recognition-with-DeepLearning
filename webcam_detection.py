import argparse
import sys, os
sys.path.append("../")

import cv2
import numpy as np

import webcam.face_detection_utilities as fdu

import dlib
from keras.models import load_model
import colorlover as cl
from train.labeldefs import *
import itertools


def nothing(x):
    pass


colors = cl.to_numeric(cl.scales['12']['qual']['Set3'])


FACE_SHAPE = (48, 48)


class MindMirror:

    def __init__(self, wname='Big Data Expo Demo', mode='dlib'):
        self.age, self.gender, self.emotion = self.load_models()

        self.face_detector = dlib.get_frontal_face_detector()
        #self.shape_predictor = dlib.shape_predictor('model/shape_predictor.dat')

        self.camera = self.getCameraStreaming()

        self.mode = mode
        self.wname = wname
        self.setup_window()

    def run(self):
        nfaces = 0

        emotions_result = []
        age_result = []
        gender_result = []

        for count in itertools.count():
            flag, frame = self.camera.read()
            frame = frame[:, ::-1, :]
            output = np.copy(frame)

            # 68
            #
            bb_fac = np.sqrt(cv2.getTrackbarPos('boxsize', self.wname) / 50.0)
            #bb_fac = min(max(0.5, bb_fac), 2.0)
            #yoffset = cv2.getTrackbarPos('yoffset', self.wname) / 100.0

            try:
                bounding_boxes = fdu.get_bounding_boxes(frame, mode=self.mode, fac=np.sqrt(68 / 50), yoffset=0.04)
                bounding_boxes = sorted(bounding_boxes, key=lambda bb: bb[0])

                if len(bounding_boxes) != nfaces and count % 5 == 0:

                    adience_pp_crops = np.stack(
                        [self.adience_preprocess(frame, bb) for bb in bounding_boxes]
                    )
                    emotion_pp_crops = np.stack(
                        [fdu.preprocess(frame, bb, face_shape=FACE_SHAPE) for bb in bounding_boxes]
                    )

                    emotions_result = self.emotion.predict(emotion_pp_crops)
                    age_result = self.age.predict(adience_pp_crops)
                    gender_result = self.gender.predict(adience_pp_crops)

                self.display_label(output, emotions_result, colors, bounding_boxes, 0, 1, emotion_l_to_c)
                self.display_label(output, age_result, colors, bounding_boxes, 2, 1, age_l_to_c, prefix='Age: ')
                self.display_label(output, gender_result, colors, bounding_boxes, 2, 3, gender_l_to_c)

                for i, bb in enumerate(bounding_boxes):

                    if bb[2] > output.shape[1] or bb[3] > output.shape[0] or bb[0] < 0 or bb[1] < 0:
                        continue
                    cv2.rectangle(
                        np.asarray(output), (bb[0], bb[1]), (bb[2], bb[3]), colors[i % len(colors)],
                        thickness=2
                    )
                cv2.imshow(self.wname, output)
            except cv2.error as e:
                print(e)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    def display_label(self, out, result, colors, bbs, i1, i2, l_to_c, prefix=''):
        font = cv2.QT_FONT_NORMAL
        for i, (r, bb) in enumerate(zip(result, bbs)):
            index = np.argmax(r)
            label = l_to_c[index]
            confidence = max(r)

            pos = (bb[i1], bb[i2])
            if pos[0] > out.shape[1] or pos[1] > out.shape[0] or pos[0] < 0 or pos[1] < 0:
                continue

            cv2.putText(out, prefix + label + " {}%".format(int(confidence * 100)), pos, font, 0.75,
                        colors[i % len(colors)], 1, cv2.LINE_AA)

    @staticmethod
    def getCameraStreaming():
        capture = cv2.VideoCapture(0)
        if not capture:
            print("Failed to initialize camera")
            sys.exit(1)
        return capture


    @staticmethod
    def load_models():
        age_estimator = load_model('model/age.h5')
        gender_recognizer = load_model('model/gender.h5')
        emotion_recognizer = load_model('model/emotion.h5')
        return age_estimator, gender_recognizer, emotion_recognizer

    def setup_window(self):
        cv2.startWindowThread()
        cv2.namedWindow(self.wname)
        #cv2.createTrackbar('yoffset', self.wname, 0, 100, nothing)
        #cv2.createTrackbar('boxsize', self.wname, 0, 400, nothing)

    @staticmethod
    def adience_preprocess(frame, bb):
        crop = frame[bb[1]:bb[3], bb[0]:bb[2]]
        resized = cv2.resize(crop, (227, 227))
        return resized[:, :, ::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
    parser.add_argument('--mode', default='dlib', choices=['cascade', 'dlib'])

    args = parser.parse_args()
    mind_mirror = MindMirror()

    mind_mirror.run()