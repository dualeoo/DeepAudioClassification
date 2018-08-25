# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import logging
import os
import pickle
from random import shuffle

import numpy as np

from multiprocessing as mp

from config import dataset_path, nameOfUnknownGenre, realTestDatasetPrefix, batchSize, slicesPath, slicesTestPath, \
    sliceSize, file_names_path, real_test_dataset_path, slices_per_genre_ratio, slices_per_genre_ratio_each_genre, \
    my_logger_name, number_of_slices_debug, number_of_slices_before_informing_users, number_of_workers
from imageFilesTools import get_image_data

my_logger = logging.getLogger(my_logger_name)


# def test_method():
#     my_logger.debug("A testing message inside datasetTools")


def get_default_dataset_name(sliceSize, debug):
    if not debug:
        name = "{}".format(100)
    else:
        name = "{}".format("DEBUG")
    name += "_{}".format(sliceSize)
    return name


def get_real_test_dataset_name(sliceSize):
    real_test_dataset_suffix = "0_{}".format(sliceSize)
    real_test_dataset_name = "{}_X_{}".format(realTestDatasetPrefix, real_test_dataset_suffix)
    return real_test_dataset_name


# Creates or loads dataset if it exists
# Mode = "train" or "test"
def get_dataset(genres, sliceSize, validationRatio, testRatio, mode, debug):
    # TODOx create small data set in debug mode
    dataset_name = "train_X_" + get_default_dataset_name(sliceSize, debug)  # TODOx look inside
    my_logger.debug("[+] Dataset name: {}".format(dataset_name))
    if not os.path.isfile("{}{}.p".format(dataset_path, dataset_name)):  # TODOx look inside get_path_to_dataset
        # TODOx task change all my_logger.debug to my_logger.debug
        my_logger.debug("[+] Creating dataset:The slice size is {}... ⌛️".format(
            sliceSize))
        return create_dataset_from_slices(genres, sliceSize, validationRatio, testRatio, mode, debug)  # fixmex
        # TODOx look inside
    else:
        my_logger.debug("[+] Using existing dataset")
        return load_dataset(sliceSize, mode, debug)  # fixmex


# Loads dataset
# Mode = "train" or "test"
def load_dataset(slice_size, mode, debug):
    dataset_name = get_default_dataset_name(slice_size, debug)

    if mode == "train":
        my_logger.debug("[+] Loading training and validation datasets... ")
        train_x = pickle.load(open("{}train_X_{}.p".format(dataset_path, dataset_name), "rb"))
        train_y = pickle.load(open("{}train_y_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_x = pickle.load(open("{}validation_X_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_y = pickle.load(open("{}validation_y_{}.p".format(dataset_path, dataset_name), "rb"))
        my_logger.debug("    Training and validation datasets loaded! ✅")
        return train_x, train_y, validation_x, validation_y

    elif mode == "test":
        my_logger.debug("[+] Loading testing dataset... ")
        test_x = pickle.load(open("{}test_X_{}.p".format(dataset_path, dataset_name), "rb"))
        test_y = pickle.load(open("{}test_y_{}.p".format(dataset_path, dataset_name), "rb"))
        my_logger.debug("    Testing dataset loaded! ✅")
        return test_x, test_y


# Saves dataset
def save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, sliceSize, debug):
    # Create path for dataset if not existing

    my_logger.debug("[+] Saving dataset... ")
    datasetName = get_default_dataset_name(sliceSize, debug)
    pickle.dump(train_X, open("{}train_X_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(train_y, open("{}train_y_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(validation_X, open("{}validation_X_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(test_X, open("{}test_X_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(test_y, open("{}test_y_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    my_logger.debug("    Dataset saved! ✅💾")


# Creates and save dataset from slices
def identify_suitable_number_of_slices(genres):
    number_of_files_in_dir = []
    for genre in genres:
        file_names = os.listdir(slicesPath + genre)
        number_of_files_in_dir.append(len(file_names))
        return int(min(number_of_files_in_dir) * slices_per_genre_ratio)


def create_dataset_from_slices(genres, slice_size, validation_ratio, test_ratio, mode, debug):
    data = []
    # slices_per_genre = identify_suitable_number_of_slices(genres)
    # my_logger.debug("Number of slices per genre = {}".format(slices_per_genre))

    genre_index = 1
    number_of_genres = len(genres)
    for genre in genres:
        my_logger.debug("-> Adding genre {} ({}/{})".format(genre, genre_index, number_of_genres))
        # Get slices in genre subfolder
        file_names = os.listdir(slicesPath + genre)
        file_names = [filename for filename in file_names if filename.endswith('.png')]
        if not debug:
            slices_per_genre = int(len(file_names) * slices_per_genre_ratio_each_genre[int(genre)])
        else:
            slices_per_genre = number_of_slices_debug
        my_logger.debug("Number of slices used for genre {} = {}".format(genre, slices_per_genre))

        # Randomize file selection for this genre
        shuffle(file_names)
        file_names = file_names[:slices_per_genre]

        # Add data (X,y)
        slice_index = 1
        pool = mp.Pool(processes=number_of_workers)
        for filename in file_names:
            result = pool.apply_async(get_image_data, args=(get_path_to_file_of_genre(filename, genre), slice_size))
            data.push(result)
            if (slice_index % number_of_slices_before_informing_users) == 0:
                my_logger.info("Finish processing slice {}/{}".format(slice_index, slices_per_genre))
            slice_index += 1
        genre_index += 1

    # Shuffle data
    shuffle(data)

    # Split data
    validationNb = int(len(data) * validation_ratio)
    testNb = int(len(data) * test_ratio)
    trainNb = len(data) - (validationNb + testNb)

    train_data = data[:trainNb]
    validation_data = data[trainNb:trainNb + validationNb]
    test_data = data[-testNb:]

    x_train, y_train = zip(*train_data)
    x_val, y_val = zip(*validation_data)
    x_test, y_test = zip(*test_data)

    # Prepare for Tflearn at the same time
    train_X = np.array(x_train).reshape([-1, slice_size, slice_size, 1])  # TODOx what is reshape?
    train_y = np.array(y_train)
    validation_X = np.array(x_val).reshape([-1, slice_size, slice_size, 1])
    validation_y = np.array(y_val)
    test_X = np.array(x_test).reshape([-1, slice_size, slice_size, 1])
    test_y = np.array(y_test)
    my_logger.debug("    Dataset created! ✅")

    # Save
    # TODOx look inside
    # fixmex
    save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, slice_size, debug)

    if mode == "train":
        return train_X, train_y, validation_X, validation_y
    elif mode == "test":
        return test_X, test_y


def process_data(filename, genre, genres, slice_size):
    imgData = get_image_data(get_path_to_file_of_genre(filename, genre), slice_size)
    # TODOx look inside get_path_to_file_of_genre
    # TODOx look inside get_image_data
    label = [1. if genre == g else 0. for g in genres]
    return imgData, label


def get_path_to_file_of_genre(filename, genre):
    return slicesPath + genre + "/" + filename


def load_file_names(real_test_dataset_name):
    path_to_file_names = get_path_to_file_names(real_test_dataset_name)  # TODOx look inside
    file_names = pickle.load(open(path_to_file_names, "rb"))
    return file_names  # TODOx


def load_real_test_dataset(real_test_dataset_name):
    my_logger.debug("[+] Loading REAL testing dataset... ")
    path_to_dataset = get_path_to_real_test_dataset(real_test_dataset_name)
    real_test_x = pickle.load(open(path_to_dataset, "rb"))
    my_logger.debug("    Testing dataset loaded! ✅")
    file_names = load_file_names(real_test_dataset_name)
    return real_test_x, file_names


def get_real_test_dataset(number_of_batches, file_names, i):
    batch_number = i + 1
    my_logger.debug("Current batch = {}/{}".format(batch_number, number_of_batches))
    real_test_dataset_name = get_real_test_dataset_name(sliceSize) + "_{}_{}".format(batch_number,
                                                                                     number_of_batches)  # TODOx look inside
    my_logger.debug("[+] Dataset name: " + real_test_dataset_name)
    path_to_dataset = get_path_to_real_test_dataset(real_test_dataset_name)  # TODOx look inside
    if not os.path.isfile(path_to_dataset):
        my_logger.debug("[+] Creating dataset with of size {}... ⌛️".format(sliceSize))
        starting_file = i * batchSize
        ending_file = starting_file + batchSize
        return create_real_test_dataset_from_slices(sliceSize
                                                    , file_names[starting_file:ending_file],
                                                    real_test_dataset_name)  # TODOx look inside
    else:
        my_logger.debug("[+] Using existing dataset")
        return load_real_test_dataset(real_test_dataset_name)  # TODOx look inside


def save_real_test_dataset(test_real_x, real_test_dataset_name):
    my_logger.debug("[+] Saving dataset... ")
    path_to_dataset = get_path_to_real_test_dataset(real_test_dataset_name)
    pickle.dump(test_real_x, open(path_to_dataset, "wb"), protocol=4)
    my_logger.debug("    Dataset saved! ✅💾")


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


def save_file_names(file_names, real_test_dataset_name):
    path_file_name = get_path_to_file_names(real_test_dataset_name)
    pickle.dump(file_names, open(path_file_name, "wb"), protocol=4)
    # TODOx


def create_real_test_dataset_from_slices(slice_size, files_for_this_batch, real_test_dataset_name):
    # number_of_files_for_this_batch = len(files_for_this_batch)
    data = []
    # file_no = 1

    for filename in files_for_this_batch:
        # my_logger.debug("Adding to dataset file: {}/{} ({})".format(file_no, number_of_files_for_this_batch, filename))
        # file_no += 1
        img_data = get_image_data(slicesTestPath + nameOfUnknownGenre + "/" + filename, slice_size)  # TODOx look inside
        data.append((img_data, filename))

    x, file_names = zip(*data)  # TODOx be careful. The way I extract file_names might be wrong
    test_real_x = np.array(x).reshape([-1, slice_size, slice_size, 1])  # TODOx why -1
    my_logger.debug("    Dataset created! ✅")
    save_real_test_dataset(test_real_x, real_test_dataset_name)  # fixmex
    save_file_names(file_names, real_test_dataset_name)  # fixmex
    return test_real_x, file_names


check_path_exist(real_test_dataset_path)
check_path_exist(dataset_path)
check_path_exist(file_names_path)
