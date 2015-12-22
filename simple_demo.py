from PIL import Image
import cv2
import pytesseract
import sys
import numpy as np
cap = cv2.VideoCapture(0)


def display(blurred, frame, gray, threshold, cropped=False):
    if cropped:
        start_image = 'start_image1'
        image_grey = 'grey1'
        image_blurred = 'blured1'
        image_threshold = 'threshold1'
    else:
        start_image = 'start_image'
        image_grey = 'grey'
        image_blurred = 'blured'
        image_threshold = 'threshold'
    cv2.imshow(start_image, frame)
    cv2.imshow(image_grey, gray)
    cv2.imshow(image_blurred, blurred)
    cv2.imshow(image_threshold, threshold)


def adaptive_threshold(frame, cropped=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 3)
    threshold = cv2.adaptiveThreshold(blurred, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 15)

    display(blurred, frame, gray, threshold, cropped)
    return threshold


def simple_threshold(frame, cropped=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 3)
    flag, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_OTSU)

    display(blurred, frame, gray, threshold, cropped)
    return threshold


def crop_image(frame, found_contour):
    pts = found_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # top-left
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # width
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # height
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    # final points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(frame, M, (max_width, max_height))
    return warp


def morph(frame):
    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    _, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    found_contour = None
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            found_contour = approx
            break
    cv2.drawContours(image, [found_contour], -1, (0, 255, 0), 3)
    display(frame, gray, edged, image)
    if found_contour is not None:
        warp = crop_image(frame, found_contour)
        return warp
    return frame


def print_text(img):
    try:
        text = pytesseract.image_to_string(img, lang='eng')
        if text:
            print(text)
    except Exception:
        pass

def start():
    if len(sys.argv) < 2:
        print('Please provide an argument..')
    else:
        while True:
            ret, frame = cap.read()

            if sys.argv[1] == '1':
                result = simple_threshold(frame)
            elif sys.argv[1] == '2':
                result = adaptive_threshold(frame)
            elif sys.argv[1] == '3':
                result = morph(frame)
                simple_threshold(result, True)
            else:
                break

            img = Image.fromarray(result)
            print_text(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    start()