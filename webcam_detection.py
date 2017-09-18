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



class IntelligentMirror:

    def __init__(self, cam_id, w, h, fps, fullscreen, double, display_faces, wname='Big Data Expo Demo', mode='dlib'):
        self.age, self.gender, self.emotion = self.load_models()

        self.face_detector = dlib.get_frontal_face_detector()

        self.camera = self.getCameraStreaming(cam_id, w, h, fps)

        self.mode = mode
        self.wname = wname
        self.setup_window(fullscreen, double, display_faces)

        self.double_display = double
        self.display_faces = display_faces

    def run(self):
        nfaces = 0

        emotions_result = []
        age_result = []
        gender_result = []

        for count in itertools.count():
            flag, frame = self.camera.read()
            frame = frame[:, ::-1, :] # Do mirroring
            output = np.copy(frame)


            try:
                if count % 2 == 0:
                    bounding_boxes = fdu.get_bounding_boxes(frame, mode=self.mode, fac=np.sqrt(68 / 50), yoffset=0.04)
                bounding_boxes = sorted(bounding_boxes, key=lambda bb: bb[0])
                n_det = len(bounding_boxes)

                if n_det > 0 and (n_det != nfaces or count % 5 == 0):

                    adience_pp_crops = np.concatenate(
                        [self.adience_preprocess(frame, bb) for bb in bounding_boxes]
                    )
                    emotion_pp_crops = np.concatenate(
                        [self.bde_preprocess(frame, bb) for bb in bounding_boxes]
                    )

                    emotions_result = self.emotion.predict(emotion_pp_crops)
                    age_result = self.age.predict(adience_pp_crops)
                    gender_result = self.gender.predict(adience_pp_crops)
                    age_result = np.reshape(age_result, (n_det, 10, len(age_l_to_c))).mean(axis=1)
                    gender_result = np.reshape(gender_result, (n_det, 10, len(gender_l_to_c))).mean(axis=1)
                    emotions_result = np.reshape(emotions_result, (n_det, 10, len(emotion_l_to_c))).mean(axis=1)

                    if self.display_faces:
                        [cv2.imshow('face{}'.format(i), adience_pp_crops[i]) for i in
                         range(self.display_faces)]

                if len(bounding_boxes) != nfaces:
                    nfaces = len(bounding_boxes)

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
                if self.double_display:
                    cv2.imshow(self.wname + ' Small View', cv2.resize(output, (960, 540)))
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
    def getCameraStreaming(cam_id, w, h, fps):
        capture = cv2.VideoCapture(cam_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        capture.set(cv2.CAP_PROP_FPS, fps)
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

    def setup_window(self, fullscreen, double, display_faces):
        cv2.startWindowThread()
        if fullscreen:
            cv2.namedWindow(self.wname, cv2.WINDOW_NORMAL)
        else:
            cv2.namedWindow(self.wname)
        cv2.namedWindow(self.wname)
        cv2.setWindowProperty(self.wname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if double:
            cv2.moveWindow(self.wname, 1920, 0)
            cv2.namedWindow(self.wname + ' Small View')
            cv2.resizeWindow(self.wname + ' Small View', 960, 540)

        if display_faces:
            [cv2.namedWindow('face{}'.format(i)) for i in range(display_faces)]

    def adience_preprocess(self, frame, bb):
        oversampled = self.oversample(frame, bb)
        return np.stack([cv2.resize(im, (227, 227)) for im in oversampled])[:, :, :, ::-1]

    def bde_preprocess(self, frame, bb, face_shape=(48, 48)):
        '''
            This function will crop user's face from the original frame
        '''
        oversampled = self.oversample(frame, bb)
        return np.expand_dims(np.stack([
            cv2.cvtColor(cv2.resize(im, face_shape), cv2.COLOR_BGR2GRAY) for im in oversampled
        ]), axis=3)


    def oversample(self, frame, bb):
        x0, y0, x1, y1 = bb
        h = bb[3] - bb[1]
        w = bb[2] - bb[0]
        dx = w // 20
        dy = h // 20

        def clipped_crop(frame, y0, y1, x0, x1):
            crop_w = x1 - x0
            crop_h = y1 - y0
            h = frame.shape[0]
            w = frame.shape[1]

            y0 = max(y0, 0)
            y1 = y0 + crop_h
            y1 = min(y1, h)
            y0 = y1 - crop_h

            x0 = max(x0, 0)
            x1 = x0 + crop_w
            x1 = min(y1, w)
            x0 = x1 - crop_w

            return frame[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]

        
        crops = np.stack([
            clipped_crop(frame, y0 - dy, y1 - dy, x0 - dx, x1 - dx),
            clipped_crop(frame, y0 + dy, y1 + dy, x0 - dx, x1 - dx),
            clipped_crop(frame, y0 - dy, y1 - dy, x0 + dx, x1 + dx),
            clipped_crop(frame, y0 + dy, y1 + dy, x0 + dx, x1 + dx),
            clipped_crop(frame, y0, y1, x0, x1)
        ])
        return np.concatenate([crops, crops[:, ::-1, :]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
    parser.add_argument('--detector', default='dlib', choices=['dlib', 'cascade'])
    parser.add_argument('--cam_id', default=1, type=int, choices=[0, 1], help='Camera ID, 0 = built-in, 1 = external')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--width', type=int, default=960, help='Capture and display width')
    parser.add_argument('--height', type=int, default=540, help='Capture and display height')
    parser.add_argument('--fullscreen', action='store_true', dest='fullscreen',
                        help='If provided, displays in fullscreen')
    parser.add_argument('--double', action='store_true', dest='double',
                        help='If provided creates a double display, one for code view and the other for fullscreen'
                             'mirror.')
    parser.add_argument('--display_faces', type=int, default=0)
    args = parser.parse_args()
    mind_mirror = IntelligentMirror(args.cam_id, args.width, args.height, args.fps, args.fullscreen, args.double,
                                    args.display_faces, mode=args.detector)

    mind_mirror.run()