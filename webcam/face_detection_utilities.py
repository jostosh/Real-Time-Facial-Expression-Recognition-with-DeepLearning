import cv2
import numpy as np
from collections import OrderedDict
import dlib

detector = dlib.get_frontal_face_detector()

CASCADE_PATH = "haarcascade_frontalface_default.xml"

RESIZE_SCALE = 3
REC_COLOR = (0, 255, 0)
# import the necessary packages
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV

    x = rect.left()
    y = rect.top()
    w = rect.right() #- x
    h = rect.bottom()  #- y

    fac = 20

    x -= w // fac
    y -= h // fac

    y -= h // (fac // 2)

    h = h + h // (fac // 2)
    w = w + w // (fac // 2)

    # return a tuple of (x, y, w, h)
    return (x * 2, y * 2, w * 2, h * 2)


def get_bounding_boxes(image, mode='cascade'):

    if mode == 'cascade':
        cascade = cv2.CascadeClassifier(CASCADE_PATH)

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        rects = cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(48, 48)
        )
        bbs = []
        for face in rects:
            bbs.append([face[0], face[1], face[0] + face[2], face[1] + face[3]])

        return bbs

    h, w = image.shape[:2]
    subim = cv2.resize(image, (w // 2, h // 2))
    rectangles = detector(subim, 1)
    return [rect_to_bb(r) for r in rectangles]


def drawFace(img, faceCoordinates):
    cv2.rectangle(np.asarray(img), (faceCoordinates[0], faceCoordinates[1]), \
    (faceCoordinates[2], faceCoordinates[3]), REC_COLOR, thickness=2)

def crop_face(img, faceCoordinates):
    '''
    extend_len_x =  (256 - (faceCoordinates[3] - faceCoordinates[1]))/2
    extend_len_y =  (256 - (faceCoordinates[0] - faceCoordinates[2]))/2
    img_size = img.shape
    if (faceCoordinates[1] - extend_len_x) >= 0 :
        faceCoordinates[1] -= extend_len_x
    if (faceCoordinates[3] + extend_len_x) < img_size[0]:
        faceCoordinates[3] += extend_len_x
    if (faceCoordinates[0] - extend_len_y) >= 0 :
        faceCoordinates[0] -= extend_len_y
    if (faceCoordinates[2] + extend_len_y) < img_size[1]:
        faceCoordinates[2] += extend_len_y
    '''
    return img[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

def preprocess(img, faceCoordinates, face_shape=(48, 48)):
    '''
        This function will crop user's face from the original frame
    '''
    face = crop_face(img, faceCoordinates)
    #face = img
    face_scaled = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)
    
    return face_gray
