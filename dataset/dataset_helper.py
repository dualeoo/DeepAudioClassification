import errno
import os

from config import realTestDatasetPrefix, path_to_slices, slices_per_genre_ratio, real_test_dataset_path, \
    file_names_path
from imageFilesTools import get_image_data


def get_default_dataset_name(slice_size, user_args):
    debug = user_args.debug
    if not debug:
        name = "{}".format(100)
    else:
        name = "{}".format("DEBUG")
    name += "_{}".format(slice_size)
    return name


def get_real_test_dataset_name(slice_size):
    real_test_dataset_suffix = "0_{}".format(slice_size)
    real_test_dataset_name = "{}_X_{}".format(realTestDatasetPrefix, real_test_dataset_suffix)
    return real_test_dataset_name


def identify_suitable_number_of_slices(genres):
    number_of_files_in_dir = []
    for genre in genres:
        file_names = os.listdir(path_to_slices + genre)
        number_of_files_in_dir.append(len(file_names))
        return int(min(number_of_files_in_dir) * slices_per_genre_ratio)


def get_path_to_file_of_genre(filename, genre):
    return path_to_slices + genre + "/" + filename


def get_path_to_real_test_dataset(dataset_name):
    # fixmex
    return "{}{}.p".format(real_test_dataset_path, dataset_name)


def get_path_to_file_names(dataset_name):
    return "{}{}.p".format(file_names_path, dataset_name)


def check_path_exist(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def process_data(filename, genre, genres, slice_size):
    imgData = get_image_data(get_path_to_file_of_genre(filename, genre), slice_size)
    # TODOx look inside get_path_to_file_of_genre
    # TODOx look inside get_image_data
    label = [1. if genre == g else 0. for g in genres]
    return imgData, label
