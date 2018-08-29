import argparse
import errno
import logging
import os
from sys import stdout

import config
# fixmeX
from config import get_current_time

my_logger = logging.getLogger(config.my_logger_name)


class UserArg:
    def __init__(self, mode, debug, model_name, run_id, run_id_real_test):
        self.mode = mode
        self.debug = debug
        self.model_name = model_name
        self.run_id = run_id
        self.run_id_real_test = run_id_real_test


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
    parser.add_argument("mode", help="Trains or tests the CNN",
                        choices=["train", "test", "slice", "sliceTest", config.real_test_prefix,
                                 config.name_of_mode_create_spectrogram,
                                 config.name_of_mode_create_spectrogram_for_test_data])
    parser.add_argument("--run-id",
                        help="This is the run_id required to create slices from spectrogram, "
                             "create dataset from slices. For more information, look at note Dung chinh model tim duoc"
                             "de predict song in Onenote",
                        dest="run_id")
    parser.add_argument("--run-id-test-real",
                        help="This is the run_in required to load real-test-dataset "
                             "when run in mode = {} ".format(config.real_test_prefix),
                        dest="run_id_real_test")

    args = parser.parse_args()
    mode_arg = args.mode
    debug = args.debug
    model_name = args.model_name
    run_id_user_supply = args.run_id
    run_id_real_test = args.run_id_real_test

    if not model_name:
        if "train" == mode_arg:
            # note i removed the if not and simpy keep the statment under it?
            # if not model_name:
            model_name = config.model_name_config
        elif mode_arg in ["test", config.real_test_prefix]:
            raise Exception('Model name must include in test and {} mode'.format(config.real_test_prefix))
        else:
            # note assume don't need model name for other modes
            pass

    if not run_id_user_supply:
        # note be care ful, adding config.real_test_prefix to below might be wrong
        if mode_arg in ["test", "slice", "sliceTest", "train", config.real_test_prefix]:
            raise Exception('Run ID must include in test, slice, and sliceTest modes ')
        else:
            run_id_user_supply = config.run_id

    if not run_id_real_test:
        if mode_arg in [config.real_test_prefix]:
            raise Exception('run_id_real_test must include in {} modes!'.format(config.real_test_prefix))
    return UserArg(mode_arg, debug, model_name, run_id_user_supply, run_id_real_test)


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


def get_gernes_and_classes(active_config):
    # fixmeX
    genres = os.listdir(active_config.path_to_slices_for_training)
    genres = [genre for genre in genres if os.path.isdir(active_config.path_to_slices_for_training + genre)]
    nb_classes = len(genres)
    return genres, nb_classes


def check_path_exist(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


