import logging
import multiprocessing as mp
import os
from random import shuffle

import numpy as np

from config import path_to_slices, slices_per_genre_ratio_each_genre, number_of_slices_debug, \
    number_of_slices_before_informing_users, path_to_test_slices, unknown_genre, my_logger_name
from dataset.dataset_helper import process_data
from dataset.dataset_save import save_dataset, save_real_test_dataset, save_file_names
from imageFilesTools import get_image_data

my_logger = logging.getLogger(my_logger_name)


def create_dataset_from_slices(genres, slice_size, validation_ratio, test_ratio, user_args):
    mode = user_args.mode
    data = []
    # slices_per_genre = identify_suitable_number_of_slices(genres)
    # my_logger.debug("Number of slices per genre = {}".format(slices_per_genre))

    genre_index = 1
    number_of_genres = len(genres)
    for genre in genres:
        my_logger.debug("-> Adding genre {} ({}/{})".format(genre, genre_index, number_of_genres))
        # Get slices in genre subfolder
        file_names = os.listdir(path_to_slices + genre)
        file_names = [filename for filename in file_names if filename.endswith('.png')]
        if not user_args.debug:
            slices_per_genre = int(len(file_names) * slices_per_genre_ratio_each_genre[int(genre)])
        else:
            slices_per_genre = number_of_slices_debug
        my_logger.debug("Number of slices used for genre {} = {}".format(genre, slices_per_genre))

        # Randomize file selection for this genre
        shuffle(file_names)
        file_names = file_names[:slices_per_genre]

        # Add data (X,y)
        slice_index = 1
        pool = mp.Pool(processes=os.cpu_count())
        results = [pool.apply_async(process_data, args=(filename, genre, genres, slice_size))
                   for filename in file_names]
        for result in results:
            img_data, label = result.get()
            data.append((img_data, label))
            if (slice_index % number_of_slices_before_informing_users) == 0:
                my_logger.info("Finish processing slice {}/{}".format(slice_index, slices_per_genre))
            slice_index += 1
        genre_index += 1

    # Shuffle data
    shuffle(data)

    # Split data
    validation_nb = int(len(data) * validation_ratio)
    testNb = int(len(data) * test_ratio)
    trainNb = len(data) - (validation_nb + testNb)

    train_data = data[:trainNb]
    validation_data = data[trainNb:trainNb + validation_nb]
    test_data = data[-testNb:]

    x_train, y_train = zip(*train_data)
    x_val, y_val = zip(*validation_data)
    x_test, y_test = zip(*test_data)

    # Prepare for Tflearn at the same time
    train_X = np.array(x_train).reshape([-1, slice_size, slice_size, 1])  # TODOx what is reshape?
    train_y = np.array(y_train)
    validation_X = np.array(x_val).reshape([-1, slice_size, slice_size, 1])
    validation_y = np.array(y_val)
    test_X = np.array(x_test).reshape([-1, slice_size, slice_size, 1])
    test_y = np.array(y_test)
    my_logger.debug("    Dataset created! ✅")

    # Save
    save_dataset(train_X, train_y, validation_X, validation_y, test_X, test_y, slice_size, user_args)

    if mode == "train":
        return train_X, train_y, validation_X, validation_y
    elif mode == "test":
        return test_X, test_y


class FileNameAndProcess:

    def __init__(self, file_name, process) -> None:
        self.file_name = file_name
        self.process = process


def create_real_test_dataset_from_slices(slice_size, files_for_this_batch, real_test_dataset_name, user_args):
    # TODOx task user_args
    # number_of_files_for_this_batch = len(files_for_this_batch)
    data = []
    # file_no = 1

    pool = mp.Pool(processes=user_args.cpu_number)
    path = path_to_test_slices + unknown_genre + "/"
    results = [FileNameAndProcess(filename, pool.apply_async(get_image_data, args=(path + filename, slice_size)))
               for filename in files_for_this_batch]
    for result in results:
        file_name = result.file_name
        process = result.process
        img_data = process.get()
        data.append((img_data, file_name))

    x, file_names = zip(*data)  # TODOx be careful. The way I extract file_names might be wrong
    test_real_x = np.array(x).reshape([-1, slice_size, slice_size, 1])  # TODOx why -1
    my_logger.debug("[+] Dataset created! ✅")
    save_real_test_dataset(test_real_x, real_test_dataset_name)  # fixmex todox look inside
    save_file_names(file_names, real_test_dataset_name)  # fixmex todox look inside
    return test_real_x, file_names
