import argparse
import sys
sys.path.append("./")
import cv2
import numpy as np
from detection import get_bounding_boxes
from keras.models import load_model
import colorlover as cl
from dnn.labeldefs import *
import itertools
from predictor import Predictor


TOP_LEFT = (0, 1)
TOP_RIGHT = (2, 1)
BOTTOM_RIGHT = (2, 3)


class IntelligentMirror:

    def __init__(self, cam_id, w, h, fps, fullscreen, double, wname='Big Data Expo Demo', mode='dlib'):
        self.age_predictor, self.gender_predictor, self.emotion_predictor = self.load_models()
        self.camera = self.get_camera_streaming(cam_id, w, h, fps)

        self.mode = mode
        self.wname = wname
        self.setup_window(fullscreen, double)

        self.dual_display = double
        self.colors = cl.to_numeric(cl.scales['12']['qual']['Set3'])

    def run(self):
        n_previous_bounding_boxes = 0
        # Initialize the names and confidences with empty lists
        emotion_names, emotion_confidences, age_names, age_confidences, gender_names, gender_confidences = 6 * [[]]

        # Main loop, continues until CTRL-C is pressed
        for count in itertools.count():
            # Read camera
            frame = self.read_camera()
            out = frame.copy()
            try:
                if count % 2 == 0:
                    bounding_boxes = get_bounding_boxes(frame, mode=self.mode, fac=np.sqrt(68 / 50), yoffset=0.04)

                # Bounding boxes not always in same order, bb == (left, right, top, bottom)
                bounding_boxes = sorted(bounding_boxes, key=lambda bb: bb[0])
                n_bounding_boxes = len(bounding_boxes)

                # Perform predictions whenever there are any AND
                # (we have a different number of bounding boxes than before OR we have reached a count of modulo 5)
                if n_bounding_boxes > 0 and (n_bounding_boxes != n_previous_bounding_boxes or count % 5 == 0):
                    emotion_names, emotion_confidences = self.emotion_predictor.predict(
                        bounding_boxes, frame, oversample=True
                    )
                    age_names, age_confidences = self.age_predictor.predict(bounding_boxes, frame, oversample=True)
                    gender_names, gender_confidences = self.gender_predictor.predict(
                        bounding_boxes, frame, oversample=True
                    )

                # Update the number of bounding boxes
                n_previous_bounding_boxes = n_bounding_boxes

                # Display the name of the classes
                self.display_label(out, emotion_names, emotion_confidences, bounding_boxes, TOP_LEFT)
                self.display_label(out, age_names, age_confidences, bounding_boxes, TOP_RIGHT, prefix='Age: ')
                self.display_label(out, gender_names, gender_confidences, bounding_boxes, BOTTOM_RIGHT)

                # Draw the bounding boxes
                self.draw_bounding_boxes(bounding_boxes, out)

                # Show on smaller window
                if self.dual_display:
                    cv2.imshow(self.wname + ' Small View', cv2.resize(out, (960, 540)))

                # Show on main window
                cv2.imshow(self.wname, out)
            except cv2.error as e:
                print(e)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    def read_camera(self):
        _, frame = self.camera.read()
        return self.mirror(frame)

    def draw_bounding_boxes(self, bounding_boxes, output):
        for i, bb in enumerate(bounding_boxes):

            if bb[2] > output.shape[1] or bb[3] > output.shape[0] or bb[0] < 0 or bb[1] < 0:
                continue
            cv2.rectangle(
                np.asarray(output), (bb[0], bb[1]), (bb[2], bb[3]), self.colors[i % len(self.colors)],
                thickness=2
            )

    def display_label(self, out, names, confidences, bbs, pos, prefix=''):
        i1, i2 = pos
        font = cv2.QT_FONT_NORMAL
        for i, (label, confidence, bb) in enumerate(zip(names, confidences, bbs)):
            pos = (bb[i1], bb[i2])
            if pos[0] > out.shape[1] or pos[1] > out.shape[0] or pos[0] < 0 or pos[1] < 0:
                continue
            cv2.putText(out, prefix + label + " {}%".format(int(confidence * 100)), pos, font, 0.75,
                        self.colors[i % len(self.colors)], 1, cv2.LINE_AA)

    @staticmethod
    def get_camera_streaming(cam_id, w, h, fps):
        capture = cv2.VideoCapture(cam_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        capture.set(cv2.CAP_PROP_FPS, fps)
        if not capture:
            print("Failed to initialize camera")
            sys.exit(1)
        return capture

    def load_models(self):
        age_model = load_model('model/age.h5')
        gender_model = load_model('model/gender.h5')
        emotion_model = load_model('model/emotion.h5')

        age_predictor = Predictor(age_l_to_c, age_model, (227, 227), grayscale=False)
        gender_predictor = Predictor(gender_l_to_c, gender_model, (227, 227), grayscale=False)
        emotion_predictor = Predictor(emotion_l_to_c, emotion_model, (48, 48), grayscale=True)

        return age_predictor, gender_predictor, emotion_predictor

    def setup_window(self, fullscreen, double):
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

    @staticmethod
    def mirror(frame):
        return frame[:, ::-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
    parser.add_argument('--detector', default='dlib', choices=['dlib', 'cascade'])
    parser.add_argument('--cam_id', default=1, type=int, choices=[0, 1], help='Camera ID, 0 = built-in, 1 = external')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--width', type=int, default=1920, help='Capture and display width')
    parser.add_argument('--height', type=int, default=1080, help='Capture and display height')
    parser.add_argument('--fullscreen', action='store_true', dest='fullscreen',
                        help='If provided, displays in fullscreen')
    parser.add_argument('--double', action='store_true', dest='double',
                        help='If provided creates a double display, one for code view and the other for fullscreen'
                             'mirror.')
    args = parser.parse_args()
    mind_mirror = IntelligentMirror(args.cam_id, args.width, args.height, args.fps, args.fullscreen, args.double,
                                    mode=args.detector)

    mind_mirror.run()