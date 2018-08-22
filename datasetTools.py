# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import pickle
from random import shuffle

import numpy as np

from config import dataset_path, nameOfUnknownGenre, realTestDatasetPrefix, batchSize, slicesPath, slicesTestPath, \
    sliceSize, file_names_path
from imageFilesTools import get_image_data


# Creates name of dataset from parameters


def get_default_dataset_name(nbPerGenre, sliceSize):
    name = "{}".format(nbPerGenre)
    name += "_{}".format(sliceSize)
    return name


def get_real_test_dataset_name(sliceSize):
    real_test_dataset_suffix = "0_{}".format(sliceSize)
    real_test_dataset_name = "{}_X_{}".format(realTestDatasetPrefix, real_test_dataset_suffix)
    return real_test_dataset_name


# Creates or loads dataset if it exists
# Mode = "train" or "test"
def get_dataset(nbPerGenre, genres, sliceSize, validationRatio, testRatio, mode):
    dataset_name = "train_X_" + get_default_dataset_name(nbPerGenre, sliceSize)
    print("[+] Dataset name: {}".format(dataset_name))
    if not os.path.isfile(get_path_to_dataset(dataset_name)):
        print("[+] Creating dataset with {} slices of size {} per genre... ⌛️".format(nbPerGenre, sliceSize))
        return create_dataset_from_slices(nbPerGenre, genres, sliceSize, validationRatio, testRatio, mode)
    else:
        print("[+] Using existing dataset")
        return load_dataset(nbPerGenre, sliceSize, mode)


# Loads dataset
# Mode = "train" or "test"
def load_dataset(nb_per_genre, slice_size, mode):
    dataset_name = get_default_dataset_name(nb_per_genre, slice_size)

    if mode == "train":
        print("[+] Loading training and validation datasets... ")
        train_x = pickle.load(open("{}train_X_{}.p".format(dataset_path, dataset_name), "rb"))
        train_y = pickle.load(open("{}train_y_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_x = pickle.load(open("{}validation_X_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_y = pickle.load(open("{}validation_y_{}.p".format(dataset_path, dataset_name), "rb"))
        print("    Training and validation datasets loaded! ✅")
        return train_x, train_y, validation_x, validation_y

    elif mode == "test":
        print("[+] Loading testing dataset... ")
        test_x = pickle.load(open("{}test_X_{}.p".format(dataset_path, dataset_name), "rb"))
        test_y = pickle.load(open("{}test_y_{}.p".format(dataset_path, dataset_name), "rb"))
        print("    Testing dataset loaded! ✅")
        return test_x, test_y


# Saves dataset
def save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, sliceSize, mode):
    # Create path for dataset if not existing
    check_path_exist(dataset_path)
    print("[+] Saving dataset... ")
    datasetName = get_default_dataset_name(nbPerGenre, sliceSize)
    pickle.dump(train_X, open("{}train_X_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(train_y, open("{}train_y_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(validation_X, open("{}validation_X_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(test_X, open("{}test_X_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    pickle.dump(test_y, open("{}test_y_{}.p".format(dataset_path, datasetName), "wb"), protocol=4)
    print("    Dataset saved! ✅💾")


# Creates and save dataset from slices
def create_dataset_from_slices(slices_per_genre, genres, slice_size, validation_ratio, test_ratio, mode):
    data = []
    for genre in genres:
        print("-> Adding {}...".format(genre))
        # Get slices in genre subfolder
        file_names = os.listdir(slicesPath + genre)
        file_names = [filename for filename in file_names if filename.endswith('.png')]
        # Randomize file selection for this genre
        shuffle(file_names)
        file_names = file_names[:slices_per_genre]

        # Add data (X,y)
        for filename in file_names:
            imgData = get_image_data(get_path_to_file_of_genre(filename, genre), slice_size)
            label = [1. if genre == g else 0. for g in genres]
            data.append((imgData, label))

    # Shuffle data
    shuffle(data)

    # Extract X and y
    X, y = zip(*data)

    # Split data
    validationNb = int(len(X) * validation_ratio)
    testNb = int(len(X) * test_ratio)
    trainNb = len(X) - (validationNb + testNb)

    # Prepare for Tflearn at the same time
    train_X = np.array(X[:trainNb]).reshape([-1, slice_size, slice_size, 1])  # TODOx what is reshape?
    train_y = np.array(y[:trainNb])
    validation_X = np.array(X[trainNb:trainNb + validationNb]).reshape([-1, slice_size, slice_size, 1])
    validation_y = np.array(y[trainNb:trainNb + validationNb])
    test_X = np.array(X[-testNb:]).reshape([-1, slice_size, slice_size, 1])
    test_y = np.array(y[-testNb:])
    print("    Dataset created! ✅")

    # Save
    save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, slices_per_genre, slice_size, mode)
    if mode == "train":
        return train_X, train_y, validation_X, validation_y
    elif mode == "test":
        return test_X, test_y


def get_path_to_file_of_genre(filename, genre):
    return slicesPath + genre + "/" + filename


def load_file_names(real_test_dataset_name):
    path_to_file_names = get_path_to_file_names(real_test_dataset_name)
    file_names = pickle.load(open(path_to_file_names, "rb"))
    return file_names  # TODOx


def load_real_test_dataset(real_test_dataset_name):
    print("[+] Loading REAL testing dataset... ")
    path_to_dataset = get_path_to_dataset(real_test_dataset_name)
    real_test_x = pickle.load(open(path_to_dataset, "rb"))
    print("    Testing dataset loaded! ✅")
    file_names = load_file_names(real_test_dataset_name)
    return real_test_x, file_names


def get_real_test_dataset(number_of_batches, file_names, i):
    batch_number = i + 1
    print("Current batch = {}/{}".format(batch_number, number_of_batches))
    real_test_dataset_name = get_real_test_dataset_name(sliceSize) + "_{}_{}".format(batch_number,
                                                                                     number_of_batches)
    print("[+] Dataset name: " + real_test_dataset_name)
    path_to_dataset = get_path_to_dataset(real_test_dataset_name)
    if not os.path.isfile(path_to_dataset):
        print("[+] Creating dataset with of size {}... ⌛️".format(sliceSize))
        starting_file = i * batchSize
        ending_file = starting_file + batchSize
        return create_real_test_dataset_from_slices(sliceSize
                                                    , file_names[starting_file:ending_file], real_test_dataset_name)
    else:
        print("[+] Using existing dataset")
        return load_real_test_dataset(real_test_dataset_name)


def save_real_test_dataset(test_real_x, real_test_dataset_name):
    check_path_exist(dataset_path)
    print("[+] Saving dataset... ")
    path_to_dataset = get_path_to_dataset(real_test_dataset_name)
    pickle.dump(test_real_x, open(path_to_dataset, "wb"), protocol=4)
    print("    Dataset saved! ✅💾")


def get_path_to_dataset(dataset_name):
    return "{}{}.p".format(dataset_path, dataset_name)


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
    check_path_exist(file_names_path)
    path_file_name = get_path_to_file_names(real_test_dataset_name)
    pickle.dump(file_names, open(path_file_name, "wb"), protocol=4)
    # TODOx


def create_real_test_dataset_from_slices(slice_size, files_for_this_batch, real_test_dataset_name):
    number_of_files_for_this_batch = len(files_for_this_batch)
    data = []
    file_no = 1

    for filename in files_for_this_batch:
        print("Adding to dataset file: {}/{} ({})".format(file_no, number_of_files_for_this_batch, filename))
        file_no += 1
        img_data = get_image_data(slicesTestPath + nameOfUnknownGenre + "/" + filename, slice_size)  # TODOx look inside
        data.append((img_data, filename))

    x, file_names = zip(*data)  # TODOx be careful. The way I extract file_names might be wrong
    test_real_x = np.array(x).reshape([-1, slice_size, slice_size, 1])  # TODOx why -1
    print("    Dataset created! ✅")
    save_real_test_dataset(test_real_x, real_test_dataset_name)  # fixmex
    save_file_names(file_names, real_test_dataset_name)  # fixmex
    return test_real_x, file_names
