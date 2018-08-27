# -*- coding: utf-8 -*-
import os

from config import batchSize, nbEpoch, sliceSize, validationRatio, testRatio, modelPath, unknown_genre, \
    path_to_slices, \
    path_to_test_slices, path_to_raw_data, path_to_test_data, path_to_spectrogram, path_to_test_spectrogram, \
    number_of_batches_debug, number_of_real_test_files_debug, run_id, show_metric, shuffle_data, snapshot_step, \
    snapshot_epoch, \
    real_test_prefix
from dataset.dataset_get import get_dataset, get_real_test_dataset
from model import createModel
from songToData import create_slices_from_audio
from utility import save_predict_result, preprocess_predict_result, finalize_result, save_final_result, \
    get_current_time, set_up_logging, handle_args, print_intro


def process_slices(data_path, spectrogram_path, slice_path):
    time_starting = log_time_start(user_args.mode)
    # TODOx task change to user_args
    create_slices_from_audio(data_path, spectrogram_path, slice_path, user_args)
    log_time_end(user_args.mode, time_starting)


def log_time_start(mode):
    return log_time_helper(mode)


def log_time_end(mode, time_starting):
    current_time = log_time_helper(mode, False)
    total_amount_of_time = current_time[1] - time_starting[1]
    my_logger.info("[+] Total amount of time it takes to complete the task is {}".format(
        str(total_amount_of_time)))
    return current_time


def log_time_helper(mode, is_starting=True):
    if is_starting:
        w = "starting"
    else:
        w = "ending"
    current_time = get_current_time()
    my_logger.info("[+] Mode = {}; {} at {}".format(mode, w, current_time[0]))
    return current_time


def get_gernes_and_classes():
    global genres, nbClasses
    genres = os.listdir(path_to_slices)
    genres = [filename for filename in genres if os.path.isdir(path_to_slices + filename)]
    nbClasses = len(genres)
    return genres, nbClasses


def start_train():
    my_logger.debug("[+] Mode = train; Starting at {}".format(get_current_time()[0]))
    # Create or load new dataset
    # TODOx task change to user_args
    train_X, train_y, validation_X, validation_y = get_dataset(genres, sliceSize, validationRatio, testRatio,
                                                               user_args)  # TODOx remove slicesPerGenre
    # Train the model
    my_logger.debug("[+] Training the model...")
    model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=shuffle_data,
              validation_set=(validation_X, validation_y), snapshot_step=snapshot_step, show_metric=show_metric,
              run_id=run_id, snapshot_epoch=snapshot_epoch)
    my_logger.debug("    Model trained! ✅")
    # Save trained model
    my_logger.debug("[+] Saving the weights...")
    model.save(path_to_model)
    my_logger.debug("[+] Weights saved! ✅💾")
    my_logger.debug("[+] Training stop at {}".format(get_current_time()[0]))


def start_test():
    # Create or load new dataset
    my_logger.debug("Mode = test; Starting at {}".format(get_current_time()[0]))
    # TODOx task change to user_args
    test_X, test_y = get_dataset(genres, sliceSize, validationRatio, testRatio, user_args)
    # Load weights
    my_logger.debug("[+] Loading weights...")
    model.load(path_to_model)
    my_logger.debug("    Weights loaded! ✅")
    testAccuracy = model.evaluate(test_X, test_y)[0]
    my_logger.debug("[+] Test accuracy: {} ".format(testAccuracy))
    my_logger.debug("Test ending at {}".format(get_current_time()[0]))


def start_test_real():
    time_starting = log_time_start(user_args.mode)
    # TODOx handle debug case
    # Load weights
    my_logger.info("[+] Loading weights...")
    model.load(path_to_model)
    my_logger.info("[+] Weights loaded! ✅")
    file_names = os.listdir(path_to_test_slices + unknown_genre)
    file_names = [filename for filename in file_names if filename.endswith('.png')]
    if not debug:
        total_number_of_files = len(file_names)
    else:
        total_number_of_files = number_of_real_test_files_debug
    my_logger.info("[+] Total number of slices to process = {}".format(total_number_of_files))
    number_of_batches = int(total_number_of_files / batchSize) + 1
    my_logger.info("[+] Total number of batches to run = {}".format(number_of_batches))
    final_result = {}
    for i in range(number_of_batches):
        # TODOx look inside
        x_dataset, file_names_subset = get_real_test_dataset(number_of_batches, file_names, i, user_args)
        predict_result = model.predict(x_dataset)
        predict_result = preprocess_predict_result(predict_result)  # TODOx look inside
        save_predict_result(predict_result, file_names_subset, final_result)  # TODOx look inside
        my_logger.info("[+] Finish process batch {} of {}".format(i + 1, number_of_batches))
        if debug and i == number_of_batches_debug:
            break

    final_result = finalize_result(final_result)  # TODOx look inside
    save_final_result(final_result)  # TODOx look inside
    log_time_end(user_args.mode, time_starting)


if __name__ == "__main__":
    my_logger = set_up_logging()

    # Handle args
    user_args = handle_args()
    mode_arg = user_args.mode
    debug = user_args.debug

    print_intro()
    exit()

    if "slice" == mode_arg:
        process_slices(path_to_raw_data, path_to_spectrogram, path_to_slices)
        exit()

    if "sliceTest" == mode_arg:
        process_slices(path_to_test_data, path_to_test_spectrogram, path_to_test_slices)
        exit()

    # List genres
    genres, nbClasses = get_gernes_and_classes()

    # Create model
    model = createModel(nbClasses, sliceSize)
    path_to_model = '{}{}'.format(modelPath, user_args.model_name)

    if "train" == mode_arg:
        start_train()
        exit()

    if "test" == mode_arg:
        start_test()
        exit()

    if real_test_prefix == mode_arg:
        start_test_real()
        exit()
