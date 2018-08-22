# -*- coding: utf-8 -*-
import argparse
import os
import random
import string
import sys

from config import batchSize, slicesPerGenre, nbEpoch, sliceSize, validationRatio, testRatio, modelPath, modelName, \
    nameOfUnknownGenre
from config import slicesPath, slicesTestPath, rawDataPath, testDataPath, spectrogramsPath, spectrogramsTestPath
from datasetTools import get_dataset, get_real_test_dataset
from model import createModel
from songToData import createSlicesFromAudio
from utility import save_predict_result, preprocess_predict_result, finalize_result, save_final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train",
                                                                                    "test",
                                                                                    "slice",
                                                                                    "sliceTest",
                                                                                    "testReal"])
    args = parser.parse_args()

    print("--------------------------")
    print("| ** Config ** ")
    print("| Validation ratio: {}".format(validationRatio))
    print("| Test ratio: {}".format(testRatio))
    print("| Slices per genre: {}".format(slicesPerGenre))
    print("| Slice size: {}".format(sliceSize))  # TODO be careful, this sliceSize is different from desiredSliceSize
    print("--------------------------")

    if "slice" in args.mode:
        createSlicesFromAudio(rawDataPath, spectrogramsPath, args.mode, slicesPath)
        sys.exit()

    if "sliceTest" in args.mode:
        createSlicesFromAudio(testDataPath, spectrogramsTestPath, args.mode, slicesTestPath)
        sys.exit()

    # List genres
    genres = os.listdir(slicesPath)
    genres = [filename for filename in genres if os.path.isdir(slicesPath + filename)]
    nbClasses = len(genres)

    # Create model
    model = createModel(nbClasses, sliceSize)
    path_to_model = '{}{}'.format(modelPath, modelName)

    if "train" in args.mode:
        print("Mode = train")
        # Create or load new dataset
        train_X, train_y, validation_X, validation_y = get_dataset(slicesPerGenre, genres, sliceSize, validationRatio,
                                                                   testRatio, "train")

        # Define run id for graphs
        run_id = "MusicGenres - " + str(batchSize) + " " + ''.join(
            random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

        # Train the model
        print("[+] Training the model...")
        model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True,
                  validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
        print("    Model trained! ✅")

        # Save trained model
        print("[+] Saving the weights...")
        model.save(path_to_model)
        print("[+] Weights saved! ✅💾")
        sys.exit()

    if "test" in args.mode:
        # Create or load new dataset
        print("Mode = test")
        test_X, test_y = get_dataset(slicesPerGenre, genres, sliceSize, validationRatio, testRatio, "test")

        # Load weights
        print("[+] Loading weights...")
        model.load(path_to_model)
        print("    Weights loaded! ✅")

        testAccuracy = model.evaluate(test_X, test_y)[0]
        print("[+] Test accuracy: {} ".format(testAccuracy))
        sys.exit()

    if "testReal" in args.mode:
        print("Mode = testReal")
        # Load weights
        print("[+] Loading weights...")
        model.load(path_to_model)
        print("    Weights loaded! ✅")

        file_names = os.listdir(slicesTestPath + nameOfUnknownGenre)
        file_names = [filename for filename in file_names if filename.endswith('.png')]
        total_number_of_files = len(file_names)
        print("Total number of slices to process = {}".format(total_number_of_files))
        number_of_batches = int(total_number_of_files / batchSize) + 1
        print("Total number of batches to run = {}".format(number_of_batches))

        final_result = {}

        for i in range(number_of_batches):
            x, file_names_subset = get_real_test_dataset(number_of_batches, file_names, i)
            predictResult = model.predict_label(x)
            predictResult = preprocess_predict_result(predictResult)
            save_predict_result(predictResult, file_names_subset, final_result)  # TODOx reimplement
            print("Finish process batch {} of {}".format(i + 1, number_of_batches))

        final_result = finalize_result(final_result)
        save_final_result(final_result)
        print("[+] Finish prediction!")
        sys.exit()
