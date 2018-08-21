# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image


# Returns numpy image at size imageSize*imageSize
def getProcessedData(img, imageSize):
    img = img.resize((imageSize, imageSize), resample=Image.ANTIALIAS)
    # TODO thong tin ve image name bi mat tu day
    imgData = np.asarray(img, dtype=np.uint8).reshape(imageSize, imageSize, 1)
    # TODO be careful with this step, the original data is not used. Only data point = 255 is used. This is a hyper parameter.
    imgData = imgData / 255.
    return imgData


# Returns numpy image at size imageSize*imageSize
def getImageData(filename, imageSize):
    img = Image.open(filename)
    imgData = getProcessedData(img, imageSize)
    return imgData
