import requests
import numpy
import PIL
import os
import shutil
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras

model = keras.models.load_model('captcha.h5')
index = 0
number = 0
for index in range(1):
    with open('kaptcha.jpg', 'wb') as file:
        res = requests.get('https://eservice.mohw.gov.tw/Login/GetValidateCode?rand=983', verify = True)
        file.write(res.content)
    image = cv2.imread('kaptcha.jpg')
    #pil_image = image.convert('RGB')
    open_cv_image = numpy.array(image)
    nc = cv2.fastNlMeansDenoisingColored(open_cv_image, None, 30, 30, 7, 21)
    imgray = cv2.cvtColor(nc, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key = lambda x: x[1])
    ary = []
    #plt.imshow(thresh)
    for (c, _) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 9 and h >= 12 and h <= 20 and w <= 50:
            ary.append((x, y, w, h))
    for id, (x, y, w, h) in enumerate(ary):
        roi = numpy.ones((28,28))*255

        roi[:h,:w] = imgray[y:y + h, x:x + w]
        roi/=255
        
        roi = numpy.expand_dims(roi, axis=-1)
        roi = numpy.reshape(roi, (1, 28, 28, 1))
        thresh = roi.copy()
        result = model.predict(thresh)
        print(numpy.where(result==max(numpy.reshape(result,(9,))))[1][0])
