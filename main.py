# -*- coding: utf-8 -*-
import argparse
import os
import random
import string

from config import batchSize, nbEpoch, sliceSize, validationRatio, testRatio, modelPath, modelName, \
    nameOfUnknownGenre, slicesPath, slicesTestPath, rawDataPath, testDataPath, spectrogramsPath, spectrogramsTestPath, \
    pixelPerSecond, desiredSliceSize, length_train_id, number_of_batches_debug, learningRate, slices_per_genre_ratio, \
    number_of_real_test_files_debug
from datasetTools import get_dataset, get_real_test_dataset
from model import createModel
from songToData import createSlicesFromAudio
from utility import save_predict_result, preprocess_predict_result, finalize_result, save_final_result, get_current_time

parser = argparse.ArgumentParser()
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train",
                                                                                "test",
                                                                                "slice",
                                                                                "sliceTest",
                                                                                "testReal"])
args = parser.parse_args()
mode_arg = args.mode
debug = args.debug

run_id = "MusicGenres_" + str(batchSize) + "_" + ''.join(
    random.SystemRandom().choice(string.ascii_uppercase) for _ in range(length_train_id))


if __name__ == "__main__":

    print("--------------------------")
    print("| *** Config *** ")
    print("| Pixel per second: {}".format(pixelPerSecond))
    print("| Cut image into slice of {}px width".format(desiredSliceSize))
    print("| Resize cut slice to {}px x {}px".format(sliceSize, sliceSize))
    print("|")
    print("| Batch size: {}".format(batchSize))
    print("| Number of epoch: {}".format(nbEpoch))
    print("| Learning rate: {}".format(learningRate))
    print("|")
    print("| Validation ratio: {}".format(validationRatio))
    print("| Test ratio: {}".format(testRatio))
    print("|")
    # print("| Slices per genre: {}".format(slicesPerGenre))
    print("| Slices per genre ratio: {}".format(slices_per_genre_ratio))
    print("|")
    print("| Run_ID: {}".format(run_id))
    print("--------------------------")

    if "slice" in mode_arg:
        print("[+] Mode = slice; starting at {}".format(get_current_time()))
        createSlicesFromAudio(rawDataPath, spectrogramsPath, mode_arg,
                              slicesPath)  # TODOx look insude and set debug mode
        print("[+] Ending slice at {}".format(get_current_time()))

    if "sliceTest" in mode_arg:
        print("[+] Mode = sliceTest; starting at {}".format(get_current_time()))
        createSlicesFromAudio(testDataPath, spectrogramsTestPath, mode_arg, slicesTestPath)
        print("[+] Ending sliceTest at {}".format(get_current_time()))

    # List genres
    genres = os.listdir(slicesPath)
    genres = [filename for filename in genres if os.path.isdir(slicesPath + filename)]
    nbClasses = len(genres)

    # Create model
    model = createModel(nbClasses, sliceSize)
    path_to_model = '{}{}'.format(modelPath, modelName)

    if "train" in mode_arg:
        print("[+] Mode = train; Starting at {}".format(get_current_time()))
        # Create or load new dataset
        train_X, train_y, validation_X, validation_y = get_dataset(genres, sliceSize, validationRatio, testRatio,
                                                                   "train")  # TODOx remove slicesPerGenre

        # Train the model
        print("[+] Training the model...")
        model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True,
                  validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
        print("    Model trained! ✅")

        # Save trained model
        print("[+] Saving the weights...")
        model.save(path_to_model)
        print("[+] Weights saved! ✅💾")
        print("[+] Training stop at {}".format(get_current_time()))

    if "test" in mode_arg:
        # Create or load new dataset
        print("Mode = test; Starting at {}".format(get_current_time()))
        test_X, test_y = get_dataset(genres, sliceSize, validationRatio, testRatio,
                                     "test")  # TODOx remove slicesPerGenre

        # Load weights
        print("[+] Loading weights...")
        model.load(path_to_model)
        print("    Weights loaded! ✅")

        testAccuracy = model.evaluate(test_X, test_y)[0]
        print("[+] Test accuracy: {} ".format(testAccuracy))
        print("Test ending at {}".format(get_current_time()))

    if "testReal" in mode_arg:
        print("Mode = testReal; Starting at {}".format(get_current_time()))
        # TODOx handle debug case
        # Load weights
        print("[+] Loading weights...")
        model.load(path_to_model)
        print("    Weights loaded! ✅")

        file_names = os.listdir(slicesTestPath + nameOfUnknownGenre)
        file_names = [filename for filename in file_names if filename.endswith('.png')]
        if not debug:
            total_number_of_files = len(file_names)
        else:
            total_number_of_files = number_of_real_test_files_debug
        print("Total number of slices to process = {}".format(total_number_of_files))
        number_of_batches = int(total_number_of_files / batchSize) + 1
        print("Total number of batches to run = {}".format(number_of_batches))

        final_result = {}

        for i in range(number_of_batches):
            x, file_names_subset = get_real_test_dataset(number_of_batches, file_names, i)  # TODOx look inside
            predictResult = model.predict_label(x)
            predictResult = preprocess_predict_result(predictResult)
            save_predict_result(predictResult, file_names_subset, final_result)  # TODOx look inside
            print("Finish process batch {} of {}".format(i + 1, number_of_batches))

            if debug and i == number_of_batches_debug:
                break

        final_result = finalize_result(final_result)
        save_final_result(final_result, run_id)
        print("[+] Finish prediction at {}".format(get_current_time()))
