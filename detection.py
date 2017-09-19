import cv2
import dlib

detector = dlib.get_frontal_face_detector()

def rect_to_bb(rect, fac, yoffset, resize_fac):
    """ Convert (x, y, w, h) to (x0, y0, x1, y1) and adjust bounding box slightly by resizing and y offset """
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


def get_bounding_boxes(image, fac=20, yoffset=1.0, resize_fac=4):
    h, w = image.shape[:2]
    subim = cv2.resize(image, (w // resize_fac, h // resize_fac))
    rectangles = detector(subim, 1)
    return [rect_to_bb(r, fac, yoffset, resize_fac) for r in rectangles]

