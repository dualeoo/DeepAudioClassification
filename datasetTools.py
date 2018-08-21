# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import pickle
from random import shuffle

import numpy as np

from config import datasetPath, nameOfUnknownGenre, realTestDatasetPrefix
from imageFilesTools import getImageData
# Creates name of dataset from parameters
from utility import save_file_names


def getDatasetName(nbPerGenre, sliceSize):
    name = "{}".format(nbPerGenre)
    name += "_{}".format(sliceSize)
    return name


def getRealTestDatasetName(sliceSize):
    def getDatasetNameRT(sliceSize):
        return "0_{}".format(sliceSize)

    real_test_dataset_suffix = getDatasetNameRT(sliceSize)
    real_test_dataset_name = "{}_X_{}".format(realTestDatasetPrefix, real_test_dataset_suffix)
    return real_test_dataset_name


# Creates or loads dataset if it exists
# Mode = "train" or "test"
def getDataset(nbPerGenre, genres, sliceSize, validationRatio, testRatio, mode, slicesPath):
    if "testReal" == mode:
        real_test_dataset_name = getRealTestDatasetName(sliceSize)
        print("[+] Dataset name: " + real_test_dataset_name)
        if not os.path.isfile(datasetPath + real_test_dataset_name + ".p"):
            print("[+] Creating dataset with of size {}... ⌛️".format(sliceSize))
            createDatasetFromSlices(genres=genres, sliceSize=sliceSize, mode=mode, slicesPath=slicesPath,
                                    slicesPerGenre=None, testRatio=None, validationRatio=None)
        else:
            print("[+] Using existing dataset")

        return loadDataset(None, genres, sliceSize, mode)  # TODOx rewrite this one
    else:
        dataset_name = getDatasetName(nbPerGenre, sliceSize)
        print("[+] Dataset name: {}".format(dataset_name))
        if not os.path.isfile(datasetPath + "train_X_" + dataset_name + ".p"):
            print("[+] Creating dataset with {} slices of size {} per genre... ⌛️".format(nbPerGenre, sliceSize))
            createDatasetFromSlices(nbPerGenre, genres, sliceSize, validationRatio, testRatio, mode, slicesPath)
        else:
            print("[+] Using existing dataset")

        return loadDataset(nbPerGenre, genres, sliceSize, mode)


# Loads dataset
# Mode = "train" or "test"
def loadDataset(nbPerGenre, genres, sliceSize, mode):
    # Load existing
    if mode == "testReal":
        dataset_name = getRealTestDatasetName(sliceSize)
    else:
        dataset_name = getDatasetName(nbPerGenre, sliceSize)

    if mode == "train":
        print("[+] Loading training and validation datasets... ")
        train_X = pickle.load(open("{}train_X_{}.p".format(datasetPath, dataset_name), "rb"))
        train_y = pickle.load(open("{}train_y_{}.p".format(datasetPath, dataset_name), "rb"))
        validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath, dataset_name), "rb"))
        validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath, dataset_name), "rb"))
        print("    Training and validation datasets loaded! ✅")
        return train_X, train_y, validation_X, validation_y

    elif mode == "test":
        print("[+] Loading testing dataset... ")
        test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath, dataset_name), "rb"))
        test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath, dataset_name), "rb"))
        print("    Testing dataset loaded! ✅")
        return test_X, test_y

    elif mode == "testReal":
        print("[+] Loading REAL testing dataset... ")
        real_test_x = pickle.load(open("{}{}.p".format(datasetPath, dataset_name), "rb"))
        print("    Testing dataset loaded! ✅")
        return real_test_x


# Saves dataset
def saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, sliceSize, mode,
                test_real_x=None):
    # Create path for dataset if not existing
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    print("[+] Saving dataset... ")

    if mode == "testReal":
        datasetName = getRealTestDatasetName(sliceSize)
        pickle.dump(test_real_x, open("{}{}.p".format(datasetPath, datasetName), "wb"), protocol=4)
    else:
        datasetName = getDatasetName(nbPerGenre, sliceSize)
        pickle.dump(train_X, open("{}train_X_{}.p".format(datasetPath, datasetName), "wb"))
        pickle.dump(train_y, open("{}train_y_{}.p".format(datasetPath, datasetName), "wb"))
        pickle.dump(validation_X, open("{}validation_X_{}.p".format(datasetPath, datasetName), "wb"))
        pickle.dump(validation_y, open("{}validation_y_{}.p".format(datasetPath, datasetName), "wb"))
        pickle.dump(test_X, open("{}test_X_{}.p".format(datasetPath, datasetName), "wb"))
        pickle.dump(test_y, open("{}test_y_{}.p".format(datasetPath, datasetName), "wb"))

    print("    Dataset saved! ✅💾")


# Creates and save dataset from slices
def createDatasetFromSlices(slicesPerGenre, genres, sliceSize, validationRatio, testRatio, mode, slicesPath):
    data = []

    if mode == "testReal":
        genre = nameOfUnknownGenre
        print("-> Adding {}...".format(genre))

        # Get slices in genre subfolder
        file_names = os.listdir(slicesPath + genre)
        file_names = [filename for filename in file_names if filename.endswith('.png')]
        number_of_files = len(file_names)

        file_no = 1;
        for filename in file_names:
            print("Adding to dataset file: {}/{} ({})".format(file_no, number_of_files, filename))
            file_no += 1
            imgData = getImageData(slicesPath + genre + "/" + filename, sliceSize)
            label = [0. for g in genres]
            data.append((imgData, label, filename))

        X, y, file_names = zip(*data)  # TODO be careful. The way I extract file_names might be wrong
        test_real_x = np.array(X).reshape([-1, sliceSize, sliceSize, 1])  # TODOx why -1
        print("    Dataset created! ✅")
        saveDataset(test_real_x=test_real_x, mode=mode, sliceSize=sliceSize,
                    train_X=None, train_y=None, validation_X=None, validation_y=None,
                    test_X=None, test_y=None, nbPerGenre=None)  # TODOx handle this
        save_file_names(file_names)
        return test_real_x
    else:
        for genre in genres:
            print("-> Adding {}...".format(genre))
            # Get slices in genre subfolder
            file_names = os.listdir(slicesPath + genre)
            file_names = [filename for filename in file_names if filename.endswith('.png')]
            # Randomize file selection for this genre
            shuffle(file_names)
            file_names = file_names[:slicesPerGenre]

            # Add data (X,y)
            for filename in file_names:
                imgData = getImageData(slicesPath + genre + "/" + filename, sliceSize)
                label = [1. if genre == g else 0. for g in genres]
                data.append((imgData, label))

        # Shuffle data
        shuffle(data)

        # Extract X and y
        X, y = zip(*data)

        # Split data
        validationNb = int(len(X) * validationRatio)
        testNb = int(len(X) * testRatio)
        trainNb = len(X) - (validationNb + testNb)

        # Prepare for Tflearn at the same time
        train_X = np.array(X[:trainNb]).reshape([-1, sliceSize, sliceSize, 1])  # TODOx what is reshape?
        train_y = np.array(y[:trainNb])
        validation_X = np.array(X[trainNb:trainNb + validationNb]).reshape([-1, sliceSize, sliceSize, 1])
        validation_y = np.array(y[trainNb:trainNb + validationNb])
        test_X = np.array(X[-testNb:]).reshape([-1, sliceSize, sliceSize, 1])
        test_y = np.array(y[-testNb:])
        print("    Dataset created! ✅")

        # Save
        saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, slicesPerGenre, sliceSize, mode)

        return train_X, train_y, validation_X, validation_y, test_X, test_y
