import cv2
import dlib

detector = dlib.get_frontal_face_detector()

def rect_to_bb(rect, fac, yoffset, resize_fac):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV

    x = rect.left()
    y = rect.top()
    x2 = rect.right() #- x
    y2 = rect.bottom()  #- y

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


def get_bounding_boxes(image, mode='cascade', fac=20, yoffset=1.0, resize_fac=4):

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
    subim = cv2.resize(image, (w // resize_fac, h // resize_fac))
    rectangles = detector(subim, 1)
    return [rect_to_bb(r, fac, yoffset, resize_fac) for r in rectangles]

