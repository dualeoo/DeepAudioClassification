# Import Pillow:
import logging
import os.path

from PIL import Image

# Slices all spectrograms
from config import my_logger_name
from dataset.dataset_helper import check_path_exist

# from config import spectrogramsPath, slicesPath
my_logger = logging.getLogger(my_logger_name)


def createSlicesFromSpectrograms(desiredSize, spectrogramsPath, slicesPath):
    file_names = os.listdir(spectrogramsPath)
    index = 1
    for filename in file_names:
        if filename.endswith(".png"):
            sliceSpectrogram(filename, desiredSize, spectrogramsPath, slicesPath)  # TODOx look inside
            my_logger.debug("Finish slicing for file {}/{}".format(index, len(file_names)))
            index += 1


# Creates slices from spectrogram
# Author_TODO Improvement - Make sure we don't miss the end of the song
def sliceSpectrogram(filename, desiredSliceSize, spectrogramsPath, slicesPath):
    genre = filename.split("_")[0]  # Ex. Dubstep_19.png

    # Load the full spectrogram
    img = Image.open(spectrogramsPath + filename)

    # Compute approximate number of 128x128 samples
    width, height = img.size
    nbSamples = int(width / desiredSliceSize)
    # width - desiredSliceSize  # TODOpro Why do this? I comment out it. Be careful

    # Create path if not existing
    slicePath = slicesPath + "{}/".format(genre)
    check_path_exist(slicePath)

    # For each sample
    for i in range(nbSamples):
        # my_logger.debug("Creating slice: ", (i + 1), "/", nbSamples, "for", filename)
        # Extract and save 128x128 sample
        startPixel = i * desiredSliceSize
        imgTmp = img.crop((startPixel, 1, startPixel + desiredSliceSize, desiredSliceSize + 1))
        imgTmp.save(slicesPath + "{}/{}_{}.png".format(genre, filename[:-4], i))  # TODOx why [:-4]? to remove .png
    # TODOx inform finish slice spectrogram
