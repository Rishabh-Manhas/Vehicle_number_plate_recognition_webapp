import numpy as np
import cv2
#import matplotlib.pyplot as plt
import easyocr
#from keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
pt.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
model = cv2.CascadeClassifier('./static/models/indian_license_plate.xml')
read = easyocr.Reader(['en'])


def object_detection(path, filename):
    #read image
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    
    image1=cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)

    

    #make predictions
    coords = model.detectMultiScale(image1, 1.7, 3)

    #draw bounding on top of image
    if (len(coords)==1):
        [[x,y,w,h]] = coords
        plate_img=image[y:y+h,x:x+w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    #convert into bgr

    
    name = str(filename)
    cv2.imwrite('./static/predict/{}'.format(name), image)
    return plate_img

def OCR(path, filename):
    roi = object_detection(path, filename)
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)
    text1  = read.readtext(roi)
    text = text1[0][1]
    print(text)
    return text
