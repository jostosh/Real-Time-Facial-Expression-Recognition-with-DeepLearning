import argparse
import sys, os
sys.path.append("../")

import cv2
import numpy as np

import webcam.face_detection_utilities as fdu

import model.myVGG as vgg
from model.models import bde_model, bde_adience
import dlib
import colorlover as cl



def nothing(x):
    pass


colors = cl.to_numeric(cl.scales['12']['qual']['Set3'])

detector = dlib.get_frontal_face_detector()

switch = '0 : cascade \n1 : dlib'
windowsName = 'Preview Screen'


def setup_window():
    cv2.namedWindow(windowsName)
    cv2.createTrackbar('yoffset', windowsName, 0, 100, nothing)
    cv2.createTrackbar('boxsize', windowsName, 0, 400, nothing)
    cv2.createTrackbar(switch, windowsName, 0, 1, nothing)


setup_window()

FACE_SHAPE = (48, 48)

emo = ['Angry', 'Disgust', 'Fear', 'Happy',
       'Sad', 'Surprise', 'Neutral']


def get_model():
    if args.model == 'adience':
        return bde_adience((227, 227, 3), weights_path=args.weights_path)
    if args.model == 'vgg':
        return vgg.VGG_16(args.weights_path)
    return model



def detect_emotion(frame, bb, out, color):
    face_img = fdu.preprocess(frame, bb, face_shape=FACE_SHAPE)
    input_img = np.reshape(face_img, (1,) + face_img.shape + (1,))
    result = model.predict(input_img)[0]
    index = np.argmax(result)
    emotion = emo[index]
    confidence = max(result)
    cv2.rectangle(
        np.asarray(out), (bb[0], bb[1]), (bb[2], bb[3]), color,
        thickness=2
    )
    font = cv2.QT_FONT_NORMAL
    cv2.putText(out, emotion + " {}%".format(int(confidence * 100)), (bb[0], bb[1]), font, 1, color, 2, cv2.LINE_AA)


def refresh_frame(frame, bounding_boxes):
    for bb in bounding_boxes:
        fdu.drawFace(frame, bb)
    cv2.imshow(windowsName, frame)


def show_screen_and_detect(capture):
    while True:
        flag, frame = capture.read()
        output = np.copy(frame)

        bb_fac = np.sqrt(cv2.getTrackbarPos('boxsize', windowsName) / 50.0)
        bb_fac = min(max(0.5, bb_fac), 2.0)
        yoffset = cv2.getTrackbarPos('yoffset', windowsName) / 100.0
        mode = cv2.getTrackbarPos(switch, windowsName)
        if mode == 0:
            mode_str = 'cascade'
        elif mode == 1:
            mode_str = 'dlib'
        else:
            mode_str = args.mode
        try:
            bounding_boxes = fdu.get_bounding_boxes(frame, mode=mode_str, fac=bb_fac, yoffset=yoffset)
            for i, bb in enumerate(bounding_boxes):
                detect_emotion(frame, bb, output, colors[i % len(colors)])
            cv2.imshow(windowsName, output)
        except cv2.error as e:
            print(e)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


def getCameraStreaming():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successed to capture video streaming")
        
    return capture

def main():
    '''
    Arguments to be set:
        showCam : determine if show the camera preview screen.
    '''
    print("Enter main() function")
    
    if args.testImage is not None:
        img = cv2.imread(args.testImage)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, FACE_SHAPE)
        print(class_label[result[0]])
        sys.exit(0)

    showCam = 1

    capture = getCameraStreaming()

    if showCam:
        cv2.startWindowThread()
        cv2.namedWindow(windowsName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowsName, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    show_screen_and_detect(capture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
    parser.add_argument('-testImage',
                        help=('Given the path of testing image, the program will predict the result of the image.'
                              "This function is used to test if the model works well."))
    parser.add_argument('--mode', default='dlib', choices=['cascade', 'dlib'])
    parser.add_argument('--weights_path', default='../class_weights.h5')
    parser.add_argument('--model', default='bde', choices=['bde', 'vgg', 'adience'])

    args = parser.parse_args()
    model = get_model()
    main()
