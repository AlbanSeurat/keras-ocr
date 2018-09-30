import cv2
import numpy as np
from pathlib import Path


class mser:

    def load(self, filename):

        img = cv2.imread(filename.as_posix())
        mser = cv2.MSER_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting to GrayScale
        gray_img = img.copy()

        regions , _ = mser.detectRegions(gray)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(gray_img, hulls, 1, (0, 0, 255), 2)
        cv2.imwrite('output/%s' % filename.name, gray_img)  # Saving


    def load_word(self, filename):

        large = cv2.imread(filename.as_posix())
        rgb = cv2.pyrDown(large)
        small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        # using RETR_EXTERNAL instead of RETR_CCOMP
        _ , contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mask = np.zeros(bw.shape, dtype=np.uint8)

        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y + h, x:x + w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

            if r > 0.45 and w > 8 and h > 8:
                cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
                cv2.imwrite('output/%d-%s' % (idx, filename.name), rgb[y:y+h, x:x+w])

        cv2.imwrite('output/%s' % filename.name, rgb)  # Saving


if __name__ == "__main__":
    mser = mser()
    mser.load_word(Path("tests/test3.png"))
#    rgb = imclearborder(img, 5)
#    cv2.imwrite('ouptut/%s' % 'test3.png', rgb)  # Saving

