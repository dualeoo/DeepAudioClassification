import logging
import multiprocessing as mp
import os
from random import shuffle

import numpy as np

import config
from imageFilesTools import get_image_data

my_logger = logging.getLogger(config.my_logger_name)


class Dataset:
    def __init__(self, x_np, y_np, file_names) -> None:
        self.file_names = file_names
        self.x_np = x_np
        self.y_np = y_np


class DatasetHelper:
    def __init__(self, dataset_name, path_to_dataset) -> None:
        self.dataset_name = dataset_name
        self.path_to_folder_containing_dataset = path_to_dataset
        self.path_to_dataset = self.path_to_folder_containing_dataset + self.dataset_name + "_"
        self.name_of_x = "{}_X.p".format(self.path_to_dataset)
        self.name_of_y = "{}_Y.p".format(self.path_to_dataset)
        self.name_of_file_name = "{}_FN.p".format(self.path_to_dataset)

    def save(self, dataset: Dataset):
        my_logger.info("[+] Saving dataset {} ".format(self.dataset_name))
        np.pickle.dump(dataset.x_np, open(self.name_of_x, "wb"), protocol=4)
        np.pickle.dump(dataset.y_np, open(self.name_of_y, "wb"), protocol=4)
        np.pickle.dump(dataset.file_names, open(self.name_of_file_name, "wb"), protocol=4)
        my_logger.info("[+] Dataset saved! 💾")

    def load(self):
        # TODOx task
        my_logger.info("[+] Loading dataset {}".format(self.dataset_name))
        x_np = np.pickle.load(open(self.name_of_x, "rb"))
        y_np = np.pickle.load(open(self.name_of_y, "rb"))
        file_names = np.pickle.load(open(self.name_of_file_name, "rb"))
        my_logger.info("[+] Dataset loaded! ✅")
        return Dataset(x_np, y_np, file_names)


class GetDataset:
    # fixmeX
    # TODOx task check the test real case
    def __init__(self, genre, slice_size, dataset_path, dataset_name, path_to_slices, user_args, genres):
        self.genre = genre
        self.slice_size = slice_size
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.path_to_slices = path_to_slices
        self.user_args = user_args
        self.genres = genres
        self.path_to_slices_of_genre = self.path_to_slices + "{}/".format(self.genre)
        self.dataset_helper = DatasetHelper(self.dataset_name, self.dataset_path)

    def start(self):
        # TODOx task review how Train use method start
        slice_file_names = os.listdir(self.path_to_slices + self.genre)
        slice_file_names = [filename for filename in slice_file_names if filename.endswith('.png')]

        slices_per_genre = self.get_slices_per_genre(slice_file_names)
        my_logger.info("[+] Total number of slices to process = {}".format(slices_per_genre))

        shuffle(slice_file_names)
        slice_file_names = slice_file_names[:slices_per_genre]

        my_logger.info("[+] Dataset name: " + self.dataset_name)
        # TODOx task check this
        if not os.path.isfile(self.dataset_helper.name_of_x):
            my_logger.info("[+] {} is not created. So create it!".format(self.dataset_name))
            return self.create(slice_file_names, slices_per_genre)
        else:
            my_logger.info("[+] Using existing dataset")
            # TODOx task rework on this
            return self.dataset_helper.load()

    def get_slices_per_genre(self, slice_file_names):
        total_number_of_files = None
        if not self.user_args.debug:
            if self.user_args.mode == config.real_test_prefix:
                total_number_of_files = len(slice_file_names)
            elif self.user_args.mode == "train":
                total_number_of_files = int(len(slice_file_names) *
                                            config.slices_per_genre_ratio_each_genre[int(self.genre)])
        else:
            total_number_of_files = config.number_of_slices_debug
        return total_number_of_files

    def create(self, slice_file_names, slices_per_genre):
        data = []
        pool = mp.Pool(processes=os.cpu_count())
        workers = []
        # TODOx task
        for filename in slice_file_names:
            path_to_slice = self.path_to_slices_of_genre + filename
            job = pool.apply_async(get_image_data,
                                   args=(path_to_slice, self.slice_size))  # TODOx look inside
            workers.append(JobAndFileName(filename, job))

        slice_index = 1
        for result in workers:
            file_name = result.file_name
            process = result.process
            img_data = process.get()

            label = None
            if self.user_args.mode == config.real_test_prefix:
                label = [0. for _ in self.genres]  # fixmeX
            elif self.user_args.mode == "train":
                label = [1. if self.genre == g else 0. for g in self.genres]

            data.append((img_data, label, file_name))

            if (slice_index % config.number_of_slices_before_informing_users) == 0:
                my_logger.info("Finish processing slice {}/{}".format(slice_index, slices_per_genre))
            slice_index += 1

        x, y, file_names = zip(*data)
        x_np = np.array(x).reshape([-1, self.slice_size, self.slice_size, 1])
        y_np = np.array(y)
        my_logger.info("[+] Dataset for {} created! ✅".format(self.genre))
        # TODOx task xem coi ai su dung return result sau
        return Dataset(x_np, y_np, file_names)


class JobAndFileName:

    def __init__(self, file_name, process) -> None:
        self.file_name = file_name
        self.process = process
