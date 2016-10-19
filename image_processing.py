import numpy as np
from skimage.transform import radon
from matplotlib.mlab import rms_flat
import cv2
from pytesseract import pytesseract

from libs.crop_morphology import process_image


class ImageProcessing(object):

    @classmethod
    def gray(cls, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @classmethod
    def blur(cls, img):
        return cv2.GaussianBlur(img, (5, 5), 5)

    @classmethod
    def denoising(cls, img):
        img = cls.gray(img)
        return cv2.fastNlMeansDenoisingMulti(img, 2, 5, None, 4, 7, 35)

    @classmethod
    def compute_skew(cls, image):
        image = image - np.mean(image)  # Demean; make the brightness extend above and below zero

        # Do the radon transform and display the result
        sinogram = radon(image)

        # Find the RMS value of each row and find "busiest" rotation,
        # where the transform is lined up perfectly with the alternating dark
        # text and white lines
        r = np.array([rms_flat(line) for line in sinogram.transpose()])
        rotation = np.argmax(r)
        return (90 - rotation) / 100

    @classmethod
    def deskewing(cls, image):
        # _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        angle = cls.compute_skew(image)
        print(angle)
        angle = np.math.degrees(angle)
        # image = cv2.bitwise_not(image)
        non_zero_pixels = cv2.findNonZero(image)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)

        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = image.shape
        rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return cv2.getRectSubPix(rotated, (cols, rows), center)

    @classmethod
    def get_text_crop(cls, img):
        return process_image(img)

    @classmethod
    def global_thresholding(cls, img):
        flag, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_OTSU)
        return threshold

    @classmethod
    def adaptive_thresholding(cls, img):
        threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return threshold

    @classmethod
    def print_text(cls, img):
        text = pytesseract.image_to_string(img, lang="eng")
        if text:
            print(text)
