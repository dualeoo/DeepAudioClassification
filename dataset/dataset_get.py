# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from config import dataset_path, batchSize, sliceSize, file_names_path, real_test_dataset_path, my_logger_name, \
    real_test_dataset_name
from dataset.dataset_create import create_dataset_from_slices, create_real_test_dataset_from_slices
from dataset.dataset_helper import get_default_dataset_name, get_path_to_real_test_dataset, \
    check_path_exist
from dataset.dataset_load import load_dataset, load_real_test_dataset

my_logger = logging.getLogger(my_logger_name)


def get_dataset(genres, slice_size, validation_ratio, test_ratio, user_args):
    # TODOx create small data set in debug mode
    # TODOx process debug
    dataset_name = "train_X_" + get_default_dataset_name(slice_size, user_args)  # TODOx look inside
    my_logger.debug("[+] Dataset name: {}".format(dataset_name))
    if not os.path.isfile("{}{}.p".format(dataset_path, dataset_name)):  # TODOx look inside get_path_to_dataset
        # TODOx task change all my_logger.debug to my_logger.debug
        my_logger.debug("[+] Creating dataset:The slice size is {}... ⌛️".format(
            slice_size))
        # TODOx user_args
        return create_dataset_from_slices(genres, slice_size, validation_ratio, test_ratio, user_args)
        # TODOx look inside
    else:
        my_logger.debug("[+] Using existing dataset")
        # TODOx user_args
        return load_dataset(slice_size, user_args)


def get_real_test_dataset(number_of_batches, file_names, i, user_args):
    batch_number = i + 1
    my_logger.info("[+] Current batch = {}/{}".format(batch_number, number_of_batches))
    # from config import real_test_dataset_name
    dataset_name = real_test_dataset_name + "_{}_{}".format(batch_number,  # TODOx look inside
                                                            number_of_batches)
    my_logger.info("[+] Dataset name: " + dataset_name)
    path_to_dataset = get_path_to_real_test_dataset(dataset_name)  # TODOx look inside
    if not os.path.isfile(path_to_dataset):
        my_logger.debug("[+] Creating dataset with of size {}... ⌛️".format(sliceSize))
        starting_file = i * batchSize
        ending_file = starting_file + batchSize
        return create_real_test_dataset_from_slices(sliceSize, file_names[starting_file:ending_file],
                                                    dataset_name, user_args)  # TODOx look inside
    else:
        my_logger.debug("[+] Using existing dataset")
        return load_real_test_dataset(dataset_name)  # TODOx look inside


check_path_exist(real_test_dataset_path)
check_path_exist(dataset_path)
check_path_exist(file_names_path)
