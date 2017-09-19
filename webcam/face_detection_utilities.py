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


def rect_to_bb(rect, fac, yoffset, resize_fac):
    # take a bounding predicted by dlib and convert it
    # to the format (x0, y0, x1, y1) as we would normally do
    # with OpenCV

    x = rect.left()
    y = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()

    w = x2 - x
    h = y2 - y

    wnew = w * fac
    hnew = h * fac

    y -= hnew * yoffset
    x -= (wnew - w) / 2
    y -= (hnew - h) / 2

    y2 = y + hnew
    x2 = x + wnew

    # return a tuple of (x, y, w, h)
    return (int(x * resize_fac), int(y * resize_fac), int(x2 * resize_fac), int(y2 * resize_fac))


def get_bounding_boxes(image, mode='cascade', fac=20, yoffset=1.0, resize_fac=2):
    h, w = image.shape[:2]
    subim = cv2.resize(image, (w // resize_fac, h // resize_fac))
    rectangles = detector(subim, 1)
    return [rect_to_bb(r, fac, yoffset, resize_fac) for r in rectangles]
