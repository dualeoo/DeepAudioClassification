# -*- coding: utf-8 -*-
import os

from config import batchSize, nbEpoch, sliceSize, validationRatio, testRatio, modelPath, modelName, \
    nameOfUnknownGenre, slicesPath, slicesTestPath, rawDataPath, testDataPath, spectrogramsPath, spectrogramsTestPath, \
    pixelPerSecond, desiredSliceSize, number_of_batches_debug, learningRate, slices_per_genre_ratio, \
    number_of_real_test_files_debug, run_id
from datasetTools import get_dataset, get_real_test_dataset
from model import createModel
from songToData import createSlicesFromAudio
from utility import save_predict_result, preprocess_predict_result, finalize_result, save_final_result, \
    get_current_time, set_up_logging, handle_args

my_logger = set_up_logging()
# print("test")
# my_logger.debug("debug message")
# my_logger.warning("warning message")
# exit()

mode_arg, debug = handle_args()

# run_id = "MusicGenres_" + str(batchSize) + "_" + ''.join(
#     random.SystemRandom().choice(string.ascii_uppercase) for _ in range(length_train_id))


if __name__ == "__main__":

    my_logger.debug("--------------------------")
    my_logger.debug("| *** Config *** ")
    my_logger.debug("| Pixel per second: {}".format(pixelPerSecond))
    my_logger.debug("| Cut image into slice of {}px width".format(desiredSliceSize))
    my_logger.debug("| Resize cut slice to {}px x {}px".format(sliceSize, sliceSize))
    my_logger.debug("|")
    my_logger.debug("| Batch size: {}".format(batchSize))
    my_logger.debug("| Number of epoch: {}".format(nbEpoch))
    my_logger.debug("| Learning rate: {}".format(learningRate))
    my_logger.debug("|")
    my_logger.debug("| Validation ratio: {}".format(validationRatio))
    my_logger.debug("| Test ratio: {}".format(testRatio))
    my_logger.debug("|")
    # my_logger.debug("| Slices per genre: {}".format(slicesPerGenre))
    my_logger.debug("| Slices per genre ratio: {}".format(str(slices_per_genre_ratio)))
    my_logger.debug("|")
    my_logger.debug("| Run_ID: {}".format(run_id))
    my_logger.debug("--------------------------")

    if "slice" in mode_arg:
        my_logger.debug("[+] Mode = slice; starting at {}".format(get_current_time()))
        createSlicesFromAudio(rawDataPath, spectrogramsPath, mode_arg,
                              slicesPath)  # TODOx look insude and set debug mode
        my_logger.debug("[+] Ending slice at {}".format(get_current_time()))

    if "sliceTest" in mode_arg:
        my_logger.debug("[+] Mode = sliceTest; starting at {}".format(get_current_time()))
        createSlicesFromAudio(testDataPath, spectrogramsTestPath, mode_arg, slicesTestPath)
        my_logger.debug("[+] Ending sliceTest at {}".format(get_current_time()))

    # List genres
    genres = os.listdir(slicesPath)
    genres = [filename for filename in genres if os.path.isdir(slicesPath + filename)]
    nbClasses = len(genres)

    # Create model
    model = createModel(nbClasses, sliceSize)
    path_to_model = '{}{}'.format(modelPath, modelName)

    if "train" in mode_arg:
        my_logger.debug("[+] Mode = train; Starting at {}".format(get_current_time()))
        # Create or load new dataset
        train_X, train_y, validation_X, validation_y = get_dataset(genres, sliceSize, validationRatio, testRatio,
                                                                   "train")  # TODOx remove slicesPerGenre

        # Train the model
        my_logger.debug("[+] Training the model...")
        model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True,
                  validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
        my_logger.debug("    Model trained! ✅")

        # Save trained model
        my_logger.debug("[+] Saving the weights...")
        model.save(path_to_model)
        my_logger.debug("[+] Weights saved! ✅💾")
        my_logger.debug("[+] Training stop at {}".format(get_current_time()))

    if "test" in mode_arg:
        # Create or load new dataset
        my_logger.debug("Mode = test; Starting at {}".format(get_current_time()))
        test_X, test_y = get_dataset(genres, sliceSize, validationRatio, testRatio,
                                     "test")  # TODOx remove slicesPerGenre

        # Load weights
        my_logger.debug("[+] Loading weights...")
        model.load(path_to_model)
        my_logger.debug("    Weights loaded! ✅")

        testAccuracy = model.evaluate(test_X, test_y)[0]
        my_logger.debug("[+] Test accuracy: {} ".format(testAccuracy))
        my_logger.debug("Test ending at {}".format(get_current_time()))

    if "testReal" in mode_arg:
        my_logger.debug("Mode = testReal; Starting at {}".format(get_current_time()))
        # TODOx handle debug case
        # Load weights
        my_logger.debug("[+] Loading weights...")
        model.load(path_to_model)
        my_logger.debug("    Weights loaded! ✅")

        file_names = os.listdir(slicesTestPath + nameOfUnknownGenre)
        file_names = [filename for filename in file_names if filename.endswith('.png')]
        if not debug:
            total_number_of_files = len(file_names)
        else:
            total_number_of_files = number_of_real_test_files_debug
        my_logger.debug("Total number of slices to process = {}".format(total_number_of_files))
        number_of_batches = int(total_number_of_files / batchSize) + 1
        my_logger.debug("Total number of batches to run = {}".format(number_of_batches))

        final_result = {}

        for i in range(number_of_batches):
            x, file_names_subset = get_real_test_dataset(number_of_batches, file_names, i)  # TODOx look inside
            predictResult = model.predict_label(x)
            predictResult = preprocess_predict_result(predictResult)
            save_predict_result(predictResult, file_names_subset, final_result)  # TODOx look inside
            my_logger.debug("Finish process batch {} of {}".format(i + 1, number_of_batches))

            if debug and i == number_of_batches_debug:
                break

        final_result = finalize_result(final_result)
        save_final_result(final_result, run_id)
        my_logger.debug("[+] Finish prediction at {}".format(get_current_time()))
