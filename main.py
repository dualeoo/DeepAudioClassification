# -*- coding: utf-8 -*-

import config
import utility
from ActiveConfig import ActiveConfig
from dataset import GetDataset, DatasetHelper
from model import create_model
from modes import CreateSpectrogram, CreateSlice, Training, Test


def start_test():
    # TODOx rewrite start test
    # Create or load new dataset
    # TODOx look inside
    dataset = DatasetHelper("Test_{}".format(user_args.run_id_for_mode_test), config.dataset_path).load()
    # TODOx look inside
    Test(user_args, dataset, model, path_to_model).evaluate()


def start_test_real():
    # TODOx task check again
    dataset_name = "{}_{}".format(config.unknown_genre, config.run_id)
    dataset = GetDataset(config.unknown_genre, config.slice_size, config.dataset_path, dataset_name,
                         active_config.path_to_slices_for_testing, user_args, genres).start()
    Test(user_args, dataset, model, path_to_model).predict()


def check_all_paths_exist():
    utility.check_path_exist(active_config.path_to_spectrogram)
    utility.check_path_exist(active_config.path_to_test_spectrogram)
    utility.check_path_exist(active_config.path_to_slices_for_training)
    utility.check_path_exist(active_config.path_to_slices_for_testing)
    utility.check_path_exist(config.dataset_path)
    utility.check_path_exist(config.predict_result_path)
    # TODOx add more


if __name__ == "__main__":
    my_logger = utility.set_up_logging()
    user_args = utility.handle_args()
    active_config = ActiveConfig(user_args)
    check_all_paths_exist()
    utility.print_intro()

    if user_args.mode == config.name_of_mode_create_spectrogram:
        # TODOx
        time_starting = utility.log_time_start(user_args.mode)

        my_logger.info("Creating spectrograms...")
        CreateSpectrogram(config.path_to_raw_data, active_config.path_to_spectrogram, user_args).start()
        my_logger.info("Spectrograms created!")

        utility.log_time_end(user_args.mode, time_starting)
        exit()

    if user_args.mode == config.name_of_mode_create_spectrogram_for_test_data:
        # TODOx
        time_starting = utility.log_time_start(user_args.mode)

        my_logger.info("Creating spectrograms...")
        # fixmeX path to spectrogram
        CreateSpectrogram(config.path_to_test_data, active_config.path_to_test_spectrogram, user_args).start()
        my_logger.info("Spectrograms created!")

        utility.log_time_end(user_args.mode, time_starting)
        exit()

    if "slice" == user_args.mode:
        # TODOx look inside process_slices one day
        time_starting = utility.log_time_start(user_args.mode)

        my_logger.info("Creating slices...")
        CreateSlice(config.desired_slice_size, active_config.path_to_spectrogram,
                    active_config.path_to_slices_for_training).start()
        my_logger.info("Slices created!")

        utility.log_time_end(user_args.mode, time_starting)
        exit()

    if "sliceTest" == user_args.mode:
        time_starting = utility.log_time_start(user_args.mode)

        my_logger.info("Creating slices for testing...")
        CreateSlice(config.desired_slice_size, active_config.path_to_test_spectrogram,
                    active_config.path_to_slices_for_testing).start()
        my_logger.info("Slices for testing created!")

        utility.log_time_end(user_args.mode, time_starting)
        exit()

    genres, nb_classes = utility.get_gernes_and_classes(active_config)
    path_to_model = '{}{}'.format(config.modelPath, user_args.model_name)
    model = create_model(nb_classes, config.slice_size)

    if "train" == user_args.mode:
        # TODOx check
        Training(user_args, genres, path_to_model, nb_classes, active_config).start_train()
        exit()

    if "test" == user_args.mode:
        # TODO rewrite to predict the WHOLE song
        # TODOx check
        start_test()
        exit()

    if config.real_test_prefix == user_args.mode:
        # TODOx check
        start_test_real()
        exit()
