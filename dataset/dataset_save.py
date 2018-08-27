import logging
import pickle

from config import dataset_path, my_logger_name
from dataset.dataset_helper import get_default_dataset_name, get_path_to_real_test_dataset, get_path_to_file_names

my_logger = logging.getLogger(my_logger_name)


def save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, slice_size, user_args):
    # Create path for dataset if not existing

    my_logger.debug("[+] Saving dataset... ")
    dataset_name = get_default_dataset_name(slice_size, user_args)
    pickle.dump(train_X, open("{}train_X_{}.p".format(dataset_path, dataset_name), "wb"), protocol=4)
    pickle.dump(train_y, open("{}train_y_{}.p".format(dataset_path, dataset_name), "wb"), protocol=4)
    pickle.dump(validation_X, open("{}validation_X_{}.p".format(dataset_path, dataset_name), "wb"), protocol=4)
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(dataset_path, dataset_name), "wb"), protocol=4)
    pickle.dump(test_X, open("{}test_X_{}.p".format(dataset_path, dataset_name), "wb"), protocol=4)
    pickle.dump(test_y, open("{}test_y_{}.p".format(dataset_path, dataset_name), "wb"), protocol=4)
    my_logger.debug("    Dataset saved! ✅💾")


def save_real_test_dataset(test_real_x, real_test_dataset_name):
    my_logger.info("[+] Saving dataset... ")
    path_to_dataset = get_path_to_real_test_dataset(real_test_dataset_name)
    pickle.dump(test_real_x, open(path_to_dataset, "wb"), protocol=4)
    my_logger.info("[+] Dataset saved! ✅💾")


def save_file_names(file_names, real_test_dataset_name):
    path_file_name = get_path_to_file_names(real_test_dataset_name)
    pickle.dump(file_names, open(path_file_name, "wb"), protocol=4)
    # TODOx
