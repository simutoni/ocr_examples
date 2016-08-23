# Ocr Examples

## Setup
###Windows
    1) Install python 3.5 and make sure that you have pip installed also, and add to path C:\Python27\Scripts; (adjust the path if you installed on other place):
    https://www.python.org/downloads/
    2) Install tesseract for windows and add it to path C:\Program Files (x86)\Tesseract-OCR (adjust the path if you installed on other place):
    https://code.google.com/p/tesseract-ocr/downloads/detail?name=tesseract-ocr-setup-3.02.02.exe&can=2&q=
    3) Go to project path in cmd and type pip install -r requirements.txt
    4) Install opencv3 for python 3 on windows: https://scivision.co/install-opencv-3-0-x-for-python-on-windows/
    5) Install Droid cam app - whith this application you can make your phone a webcam or you can use a webcam
    and you will not need this application.
        5.1) Install Droid cam app on your computer: https://www.dev47apps.com/droidcam/windows/
        5.2) Install Droid cam app on your phone: https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&hl=en
        5.3) The phone and computer must be on the same network. The phone will generate an ip and a port, than
        you can add them on the desktop client and press start.

###Ubuntu
    1) make sure you have python 3 installed
        python3
        Python 3.4.3 on linux
        >>> 
        1.1) if not, `sudo apt install python3
    2) clone this repo: 
        git clone https://github.com/simutoni/ocr_examples.git && cd ocr_examples
    3) make a virtualenv:
        virtualenv -p python3 venv && source venv/bin/activate
    4) install requirements:
        pip install -r requirements.txt
    5) install tesseract for ubuntu and opencv for python:
        sudo apt install tesseract-ocr python-opencv
    6) follow this guide to install opencv (from step 10 onwards, replacing python 2 with python 3):
        http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
    7) copy the library for opencv to your venv folder    
        ln -s /usr/local/lib/python3.4/dist-packages/cv2.cpython-34m.so venv/lib/python3.4/site-packages/cv2.so
    8) test that it works
        python
        Python 3.4.3 on linux
        >>> import cv2 
        >>> cv2.__version__
        3.1.0
        

## Run the application
    1) Go to project path in cmd
    2) Run python simple_demo.py (with arguments 1, 2 or 3)
    
##Here are some nice tutorials: 
    1) http://www.pyimagesearch.com/2014/03/10/building-pokedex-python-getting-started-step-1-6/
    2) http://opencv-code.com/tutorials/automatic-perspective-correction-for-quadrilateral-objects/
    3) http://codeplasma.com/2012/12/03/getting-webcam-images-with-python-and-opencv-2-for-real-this-time/
