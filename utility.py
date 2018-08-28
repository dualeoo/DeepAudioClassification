import argparse
import csv
import datetime
import errno
import logging
import os
from functools import reduce
from sys import stdout

import config

# fixmeX

my_logger = logging.getLogger(config.my_logger_name)


class UserArg:
    def __init__(self, mode, debug, model_name, run_id_for_mode_test):
        self.mode = mode
        self.debug = debug
        self.model_name = model_name
        self.run_id_for_mode_test = run_id_for_mode_test


def process_file_name(file_name):
    split_result = file_name.split("_")
    return split_result[2], split_result[3]  # TODOx


def save_predict_result(predict_results, file_names, final_result):
    # TODO debug value of all variables in this method
    for i in range(len(predict_results)):
        predict_result = predict_results[i]
        file_name = file_names[i]
        file_name, slice_id = process_file_name(file_name)
        if file_name not in final_result:
            final_result[file_name] = {}
        result_of_particular_file = final_result[file_name]
        if predict_result not in result_of_particular_file:
            result_of_particular_file[predict_result] = 1
        else:
            result_of_particular_file[predict_result] += 1
    return final_result


def preprocess_predict_result(predict_results):
    max_length = len(predict_results)
    # TODO ask Q to explain
    max_number = reduce(lambda pre, cur: pre + cur[0], predict_results, 0) / max_length
    for i in range(1, max_length - 1):
        total = reduce(lambda pre, cur: pre + cur[i], predict_results, 0) / max_length
        if total > max_number:
            max_number = total
    # TODO debug Q code
    exit()
    return max_number  # TODOx


def finalize_result(final_result):
    file_names = list(final_result.keys())
    for filename in file_names:
        result = final_result[filename]
        genre = find_max_genre(result)  # TODOx task look inside
        final_result[filename] = genre
    return final_result  # TODOx


def save_final_result(final_result):
    with open(config.predict_result_path + "{}.csv".format(config.run_id), mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Id", "Genre"])
        for file_name, genre in final_result.items():
            csv_writer.writerow([file_name, genre])


def find_max_genre(result):
    genres_n = list(result.keys())
    first_genre = genres_n[0]
    final_genre = first_genre
    max_freq = result[first_genre]

    for genre in genres_n:
        freq = result[genre]
        if freq > max_freq:
            final_genre = genre
            max_freq = freq

    return final_genre  # TODOx


def set_up_logging():
    formatter = logging.Formatter(config.logging_formatter, config.time_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers[0].setLevel(logging.WARNING)

    file_handler_for_root_logger = logging.FileHandler(config.root_logger_file_name, config.log_file_mode)
    file_handler_for_root_logger.setLevel(logging.DEBUG)
    file_handler_for_root_logger.setFormatter(formatter)

    my_logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(stdout)
    console.setLevel(logging.DEBUG)
    # console.setFormatter(formatter)

    file_handler = logging.FileHandler(config.my_logger_file_name, config.log_file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    my_logger.addHandler(console)
    my_logger.addHandler(file_handler)
    root_logger.addHandler(file_handler_for_root_logger)
    return my_logger


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--model-name", dest='model_name')
    # parser.add_argument("--cpu", type=int, dest='number_of_cpu', default=number_of_workers)
    parser.add_argument("mode", help="Trains or tests the CNN", choices=["train", "test", "slice", "sliceTest",
                                                                         config.real_test_prefix])
    parser.add_argument("--run-id-for-test",
                        help="This is the run_in corresponding with the test dataset using with mode test",
                        dest="run_id_for_mode_test")

    args = parser.parse_args()
    mode_arg = args.mode
    debug = args.debug
    model_name = args.model_name
    run_id_for_mode_test = args.run_id_for_mode_test

    if "train" == mode_arg:
        if not model_name:
            model_name = config.model_name_config
    elif "test" == mode_arg or config.real_test_prefix == mode_arg:
        if not model_name:
            raise Exception('Model name must include in test and testReal mode')
    return UserArg(mode_arg, debug, model_name, run_id_for_mode_test)


def print_intro():
    my_logger.info("--------------------------")
    my_logger.info("| *** Config *** ")
    my_logger.info("| Pixel per second: {}".format(config.pixel_per_second))
    my_logger.info("| Cut image into slice of {}px width".format(config.desired_slice_size))
    my_logger.info("| Resize cut slice to {}px x {}px".format(config.slice_size, config.slice_size))
    my_logger.info("|")
    my_logger.info("| Batch size: {}".format(config.batchSize))
    my_logger.info("| Number of epoch: {}".format(config.nbEpoch))
    my_logger.info("| Learning rate: {}".format(config.learningRate))
    my_logger.info("|")
    my_logger.info("| Validation ratio: {}".format(config.validation_ratio))
    my_logger.info("| Test ratio: {}".format(config.test_ratio))
    my_logger.info("|")
    # my_logger.info("| Slices per genre: {}".format(slicesPerGenre))
    my_logger.info("| Slices per genre ratio: {}".format(str(config.slices_per_genre_ratio)))
    my_logger.info("|")
    my_logger.info("| Run_ID: {}".format(config.run_id))
    my_logger.info("--------------------------")
    # TODO print other config


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
    genres = os.listdir(config.path_to_slices_for_training)
    genres = [genre for genre in genres if os.path.isdir(config.path_to_slices_for_training + genre)]
    nb_classes = len(genres)
    return genres, nb_classes


def check_path_exist(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_current_time():
    current_time_l = datetime.datetime.now()
    current_time_string_l = current_time_l.strftime("%Y%m%d_%H%M")
    return current_time_string_l, current_time_l  # fixmex (second element returned)
