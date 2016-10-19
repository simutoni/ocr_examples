import copy

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
    def find_components(cls, img, max_components=16):
        """Dilate the image until there are just a few connected components.
        Returns contours for these components."""
        # Perform increasingly aggressive dilation until there are just a few
        # connected components.
        count = 21
        dilation = 5
        n = 1
        contours = []
        while count > 16:
            n += 1
            dilated_image = cv2.dilate(img, 3, iterations=n)
            contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = len(contours)
        #print dilation
        #Image.fromarray(edges).show()
        #Image.fromarray(255 * dilated_image).show()
        return contours

    @classmethod
    def get_text_crop(cls, img):
        return process_image(img)

    # erodes image based on given kernel size (erosion = expands black areas)
    @classmethod
    def erode(clscls, img, kern_size = 3 ):
        retval, img = cv2.threshold(img, 254.0, 255.0, cv2.THRESH_BINARY) # threshold to deal with only black and white.
        kern = np.ones((kern_size,kern_size),np.uint8) # make a kernel for erosion based on given kernel size.
        eroded = cv2.erode(img, kern, 1) # erode your image to blobbify black areas
        y,x = eroded.shape # get shape of image to make a white boarder around image of 1px, to avoid problems with find contours.
        return cv2.rectangle(eroded, (0,0), (x,y), (255,255,255), 1)

    # finds contours of eroded image
    @classmethod
    def prep(cls, img, kern_size = 3 ):
        img = cv2.erode(img, kern_size)
        retval, img = cv2.threshold(img, 200.0, 255.0, cv2.THRESH_BINARY_INV) #   invert colors for findContours
        return cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # Find Contours of Image

    # given img & number of desired blobs, returns contours of blobs.
    @classmethod
    def blobbify(cls, img, num_of_labels, kern_size = 3, dilation_rate = 10):
        prep_img, contours, hierarchy = cls.prep( img.copy(), kern_size ) # dilate img and check current contour count.
        while len(contours) > num_of_labels:
            kern_size += dilation_rate # add dilation_rate to kern_size to increase the blob. Remember kern_size must always be odd.
            previous = (prep_img, contours, hierarchy)
            processed_img, contours, hierarchy = cls.prep( img.copy(), kern_size ) # dilate img and check current contour count, again.
        if len(contours) < num_of_labels:
            return (processed_img, contours, hierarchy)
        else:
            return previous

    # finds bounding boxes of all contours
    @classmethod
    def bounding_box(cls, contours):
        bBox = []
        box = None
        for curve in contours:
            box = cv2.boundingRect(curve)
        if box:
            bBox.append(box)
        return bBox

    @classmethod
    def crop(cls, img):
        p = np.array(img)
        p = p[:,:,0:3]
        img2 = img.binary_erosion(img, iterations=40)
        img3 = img.binary_dilation(img2, iterations=40)
        labels, n = img.label(img3)
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        img4 = labels==np.argmax(counts)
        img5 = img.binary_fill_holes(img4)
        result = ~img & img5
        result = img.binary_erosion(result, iterations=3)
        result = img.binary_dilation(result, iterations=3)
        return result
        #pl.imshow(result, cmap="gray")

    @classmethod
    def resize_dpi(cls, img, dpi=300):
        pass

    @classmethod
    def global_thresholding(cls, img):
        flag, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_OTSU)
        return threshold

    @classmethod
    def adaptive_thresholding(cls, img):
        threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return threshold

    @classmethod
    def angle_from_right(cls, deg):
        return min(deg % 90, 90 - (deg % 90))

    @classmethod
    def crop_image(cls, img,tol=0):
        # img is image data
        # tol  is tolerance
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    @classmethod
    def border_removal(cls, img, tol=0):
        # # Use a rotated rectangle (should be a good approximation of a border).
        # # If it's far from a right angle, it's probably two sides of a border and
        # # we should use the bounding box instead.
        # c_im = np.zeros(ary.shape)
        # r = cv2.minAreaRect(contour)
        # degs = r[2]
        # if cls.angle_from_right(degs) <= 10.0:
        #     box = cv2.BoxPoints(r)
        #     box = np.int(box)
        #     cv2.drawContours(c_im, [box], 0, 255, -1)
        #     cv2.drawContours(c_im, [box], 0, 0, 4)
        # else:
        #     x1, y1, x2, y2 = cv2.boundingRect(contour)
        #     cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        #     cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)
        #
        # return np.minimum(c_im, ary)
        pass

    @classmethod
    def print_text(cls, img):
        text = pytesseract.image_to_string(img, lang="eng")
        if text:
            print(text)
