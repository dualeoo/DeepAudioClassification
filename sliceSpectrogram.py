# Import Pillow:
import logging
import os.path

from PIL import Image

# Slices all spectrograms
from config import my_logger_name, run_id
from dataset.dataset_helper import check_path_exist

my_logger = logging.getLogger(my_logger_name)


def create_slices_from_spectrogram(desired_size, spectrograms_path, slices_path):
    file_names = os.listdir(spectrograms_path)
    index = 1
    for filename in file_names:
        if filename.endswith(".png"):
            slice_spectrogram(filename, desired_size, spectrograms_path, slices_path)  # TODOx look inside
            my_logger.debug("Finish slicing for file {}/{}".format(index, len(file_names)))
            index += 1


# Creates slices from spectrogram
# Author_TODO Improvement - Make sure we don't miss the end of the song
def slice_spectrogram(filename, desired_slice_size, spectrograms_path, slices_path):
    # fixmeX after I change name of spectrogram
    split_results = filename.split("_")
    genre = split_results[1]  # {run ID}_{genre id}_{song id}_{song file name}.png
    song_id = split_results[2]
    song_name = split_results[3]

    # Load the full spectrogram
    img = Image.open(spectrograms_path + filename)

    # Compute approximate number of 128x128 samples
    width, height = img.size
    nb_samples = int(width / desired_slice_size)
    # width - desiredSliceSize  # TODOpro Why do this? I comment out it. Be careful

    # Create path if not existing
    slice_path = slices_path + "{}/".format(genre)
    check_path_exist(slice_path)

    # For each sample
    for slice_id in range(nb_samples):
        # my_logger.debug("Creating slice: ", (i + 1), "/", nb_samples, "for", filename)
        # Extract and save 128x128 sample
        start_pixel = slice_id * desired_slice_size
        img_tmp = img.crop((start_pixel, 1, start_pixel + desired_slice_size, desired_slice_size + 1))
        # TODOx why [:-4]? to remove .png
        img_tmp.save(slices_path + "{}_{}_{}_{}_{}.png".format(run_id, genre, song_id, song_name, slice_id))
    # TODOx inform finish slice spectrogram
