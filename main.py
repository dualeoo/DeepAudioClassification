# -*- coding: utf-8 -*-
import argparse
import os
import random
import string
import sys

from config import batchSize, slicesPerGenre, nbEpoch, sliceSize, validationRatio, testRatio, modelPath, modelName, \
    percentage_of_real_test_slices, nameOfUnknownGenre
from config import slicesPath, slicesTestPath, rawDataPath, testDataPath, spectrogramsPath, spectrogramsTestPath
from datasetTools import getDataset
from model import createModel
from songToData import createSlicesFromAudio
from utility import save_predict_result

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
    train_X, train_y, validation_X, validation_y = getDataset(slicesPerGenre, genres, sliceSize, validationRatio,
                                                              testRatio, "train", slicesPath)

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
    test_X, test_y = getDataset(slicesPerGenre, genres, sliceSize, validationRatio, testRatio, "test", slicesPath)

    # Load weights
    print("[+] Loading weights...")
    model.load(path_to_model)
    print("    Weights loaded! ✅")

    testAccuracy = model.evaluate(test_X, test_y)[0]
    print("[+] Test accuracy: {} ".format(testAccuracy))
    sys.exit()

if "testReal" in args.mode:
    print("Mode = testReal")
    # Create or load new dataset
    genre = nameOfUnknownGenre
    file_names = os.listdir(slicesPath + genre)
    number_of_slices_to_score = percentage_of_real_test_slices * len(file_names)
    print(
        "number_of_slices_to_score = {} ({}%)".format(number_of_slices_to_score, percentage_of_real_test_slices * 100))
    X = getDataset(genres=genres, sliceSize=sliceSize, mode="testReal", slicesPath=slicesTestPath,
                   nbPerGenre=number_of_slices_to_score, testRatio=None, validationRatio=None)

    # Load weights
    print("[+] Loading weights...")
    model.load(path_to_model)
    print("    Weights loaded! ✅")

    predictResult = model.predict(X)
    print("The result: {}".format(predictResult))
    save_predict_result(predictResult)
    print("[+] Finish prediction!")
    sys.exit()
