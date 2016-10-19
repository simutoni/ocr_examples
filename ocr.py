import PIL
import cv2
import numpy
from tkinter import *
from tkinter import filedialog
from PIL import Image
from image_processing import ImageProcessing
from libs.crop_morphology import process_image

root = Tk()


def start():
    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)
    img = ImageProcessing.gray(img)
    img = ImageProcessing.get_text_crop(img)
    img = numpy.asarray(img)

    img = ImageProcessing.blur(img)
    img = ImageProcessing.adaptive_thresholding(img)
    PIL.Image.fromarray(img).save("images\\3_.jpg")
    img_thr = ImageProcessing.global_thresholding(img)
    PIL.Image.fromarray(img_thr).save('images\\2_.jpg')
    img_thr = PIL.Image.fromarray(img_thr)
    ImageProcessing.print_text(img_thr)


Button(text='add image', command=start).pack()
root.mainloop()
