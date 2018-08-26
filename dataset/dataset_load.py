import logging
import pickle

from config import dataset_path, my_logger_name
from dataset.dataset_helper import get_default_dataset_name, get_path_to_file_names, get_path_to_real_test_dataset

my_logger = logging.getLogger(my_logger_name)


def load_dataset(slice_size, user_args):
    mode = user_args.mode
    dataset_name = get_default_dataset_name(slice_size, user_args)

    if mode == "train":
        my_logger.debug("[+] Loading training and validation datasets... ")
        train_x = pickle.load(open("{}train_X_{}.p".format(dataset_path, dataset_name), "rb"))
        train_y = pickle.load(open("{}train_y_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_x = pickle.load(open("{}validation_X_{}.p".format(dataset_path, dataset_name), "rb"))
        validation_y = pickle.load(open("{}validation_y_{}.p".format(dataset_path, dataset_name), "rb"))
        my_logger.debug("    Training and validation datasets loaded! ✅")
        return train_x, train_y, validation_x, validation_y

    elif mode == "test":
        my_logger.debug("[+] Loading testing dataset... ")
        test_x = pickle.load(open("{}test_X_{}.p".format(dataset_path, dataset_name), "rb"))
        test_y = pickle.load(open("{}test_y_{}.p".format(dataset_path, dataset_name), "rb"))
        my_logger.debug("    Testing dataset loaded! ✅")
        return test_x, test_y


def load_file_names(real_test_dataset_name):
    path_to_file_names = get_path_to_file_names(real_test_dataset_name)  # TODOx look inside
    file_names = pickle.load(open(path_to_file_names, "rb"))
    return file_names  # TODOx


def load_real_test_dataset(real_test_dataset_name):
    my_logger.info("[+] Loading REAL testing dataset... ")
    path_to_dataset = get_path_to_real_test_dataset(real_test_dataset_name)
    real_test_x = pickle.load(open(path_to_dataset, "rb"))
    my_logger.debug("[+] Testing dataset loaded! ✅")
    file_names = load_file_names(real_test_dataset_name)  # TODOx look inside
    return real_test_x, file_names
