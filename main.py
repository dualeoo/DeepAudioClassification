# -*- coding: utf-8 -*-
import argparse
import os
import random
import string
import sys

from config import batchSize, slicesPerGenre, nbEpoch, sliceSize, validationRatio, testRatio
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

if "train" == args.mode:
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
    model.save('musicDNN.tflearn')
    print("[+] Weights saved! ✅💾")

if "test" == args.mode:
    # Create or load new dataset
    test_X, test_y = getDataset(slicesPerGenre, genres, sliceSize, validationRatio, testRatio, args.mode, slicesPath)

    # Load weights
    print("[+] Loading weights...")
    model.load('musicDNN.tflearn')
    print("    Weights loaded! ✅")

    testAccuracy = model.evaluate(test_X, test_y)[0]
    print("[+] Test accuracy: {} ".format(testAccuracy))

if "testReal" == args.mode:
    # Create or load new dataset
    X = getDataset(genres=genres, sliceSize=sliceSize, mode=args.mode, slicesPath=slicesTestPath,
                   nbPerGenre=None, testRatio=None, validationRatio=None)

    # Load weights
    print("[+] Loading weights...")
    model.load('musicDNN.tflearn')
    print("    Weights loaded! ✅")

    predictResult = model.predict(X)
    print("The result: {}".format(predictResult))
    save_predict_result(predictResult)
    print("[+] Finish prediction!")
