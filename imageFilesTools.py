# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image


# Returns numpy image at size imageSize*imageSize
def get_processed_data(img, slice_size):
    img = img.resize((slice_size, slice_size), resample=Image.ANTIALIAS)
    # TODO thong tin ve image name bi mat tu day
    img_data = np.asarray(img, dtype=np.uint8).reshape(slice_size, slice_size, 1)
    img_data = img_data / 255.
    return img_data


# Returns numpy image at size imageSize*imageSize
def get_image_data(path_to_slice, slice_size):
    img = Image.open(path_to_slice)
    img_data = get_processed_data(img, slice_size)
    return img_data
