import argparse
import csv
import logging
import os
from functools import reduce
from sys import stdout

from config import predictResultPath, logging_formatter, time_formatter, log_file_mode, \
    get_current_time_c, my_logger_file_name, my_logger_name, run_id, root_logger_file_name, model_name_config, \
    real_test_prefix, pixelPerSecond, desiredSliceSize, sliceSize, batchSize, nbEpoch, learningRate, \
    validationRatio, testRatio, slices_per_genre_ratio

my_logger = logging.getLogger(my_logger_name)


def process_file_name(file_name):
    split_result = file_name.split("_")
    return split_result[2], split_result[3]  # TODOx


def save_predict_result(predict_results, file_names, final_result):
    # TODO task debug value of all variables in this method
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
    # TODO task ask Q to explain
    max_number = reduce(lambda pre, cur: pre + cur[0], predict_results, 0) / max_length
    for i in range(1, max_length - 1):
        total = reduce(lambda pre, cur: pre + cur[i], predict_results, 0) / max_length
        if total > max_number:
            max_number = total
    # TODO task debug Q code
    exit()
    return max_number  # TODOx


def finalize_result(final_result):
    file_names = list(final_result.keys())
    for filename in file_names:
        result = final_result[filename]
        genre = find_max_genre(result)  # TODOx task look inside
        final_result[filename] = genre
    return final_result  # TODOx


def get_current_time():
    # fixmex task now return tuple
    return get_current_time_c()


def save_final_result(final_result):
    with open(predictResultPath + "{}.csv".format(run_id), mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Id", "Genre"])
        for file_name, genre in final_result.items():
            csv_writer.writerow([file_name, genre])


def find_max_genre(result):
    genres = list(result.keys())
    first_genre = genres[0]
    final_genre = first_genre
    max_freq = result[first_genre]

    for genre in genres:
        freq = result[genre]
        if freq > max_freq:
            final_genre = genre
            max_freq = freq

    return final_genre  # TODOx


def set_up_logging():
    formatter = logging.Formatter(logging_formatter, time_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers[0].setLevel(logging.WARNING)

    file_handler_for_root_logger = logging.FileHandler(root_logger_file_name, log_file_mode)
    file_handler_for_root_logger.setLevel(logging.DEBUG)
    file_handler_for_root_logger.setFormatter(formatter)

    my_logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(stdout)
    console.setLevel(logging.DEBUG)
    # console.setFormatter(formatter)

    file_handler = logging.FileHandler(my_logger_file_name, log_file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    my_logger.addHandler(console)
    my_logger.addHandler(file_handler)
    root_logger.addHandler(file_handler_for_root_logger)
    return my_logger


class UserArg:
    def __init__(self, mode, debug, model_name):
        self.mode = mode
        self.debug = debug
        self.model_name = model_name
        self.cpu_number = os.cpu_count()


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--model-name", dest='model_name')
    # parser.add_argument("--cpu", type=int, dest='number_of_cpu', default=number_of_workers)
    parser.add_argument("mode", help="Trains or tests the CNN", choices=["train", "test", "slice", "sliceTest",
                                                                         real_test_prefix])
    args = parser.parse_args()
    mode_arg = args.mode
    debug = args.debug
    model_name = args.model_name
    if "train" in mode_arg:
        if not model_name:
            model_name = model_name_config
    elif "test" in mode_arg or real_test_prefix in mode_arg:
        if not model_name:
            raise Exception('Model name must include in test mode')
    return UserArg(mode_arg, debug, model_name)


def print_intro():
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
    # TODO task print other config
