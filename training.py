from random import shuffle

import numpy as np

import config
from dataset import GetDataset, DatasetHelper, Dataset
from model import create_model
from utility import log_time_start, log_time_end


class Training:
    def __init__(self, user_args, genres, my_logger, path_to_model, nb_classes) -> None:
        self.user_args = user_args
        self.genres = genres
        self.my_logger = my_logger
        self.path_to_model = path_to_model
        self.nb_classes = nb_classes
        self.model = create_model(nb_classes, config.slice_size)

    def start_train(self):
        time_starting = log_time_start(self.user_args.mode)

        # TODOx look inside
        all_data_points = self.prepare_dataset()
        # TODOx look inside
        train, validation, test = self.divide_into_three_set(all_data_points)
        # TODOx look inside
        self.save_the_three_dataset(train, validation, test)

        self.my_logger.info("[+] Training the model...")
        self.model.fit(train.x_np, train.y_np, n_epoch=config.nbEpoch, batch_size=config.batchSize,
                       shuffle=config.shuffle_data, validation_set=(validation.x_np, validation.y_np),
                       snapshot_step=config.snapshot_step, show_metric=config.show_metric, run_id=config.run_id,
                       snapshot_epoch=config.snapshot_epoch)
        self.my_logger.info("    Model trained! ✅")

        self.my_logger.info("[+] Saving the weights...")
        self.model.save(self.path_to_model)
        self.my_logger.info("[+] Weights saved! ✅💾")

        log_time_end(self.user_args.mode, time_starting)

    def prepare_dataset(self):
        genre_index = 1
        number_of_genres = len(self.genres)
        for genre in self.genres:
            self.my_logger.info("[+] Creating dataset for genre {} ({}/{})".format(genre, genre_index,
                                                                                   number_of_genres))
            # TODOx dataset_name in this case?
            dataset_name = "{}_{}".format(genre, config.run_id)
            # TODO look inside
            dataset = GetDataset(genre, config.slice_size, config.dataset_path, dataset_name
                                 , config.path_to_slices_for_training, self.user_args, self.genres).start()

            # TODO look inside
            dataset.save()
            genre_index += 1

        # TODO look inside
        all_dataset = self.load_dataset_back_to_memory()
        # TODO look inside
        all_dataset = self.zip_again(all_dataset)
        # TODO look inside
        return self.merge_into_one_big_chunk(all_dataset)

    def load_dataset_back_to_memory(self):
        # TODOx task
        genre_index = 1
        number_of_genres = len(self.genres)
        all_dataset = []

        for genre in self.genres:
            self.my_logger.info("[+] Loading dataset for genre {} ({}/{})".format(genre, genre_index,
                                                                                  number_of_genres))
            # TODOx dataset_name in this case?
            dataset_name = "{}_{}".format(genre, config.run_id)
            dataset = DatasetHelper(dataset_name, config.dataset_path).load()
            all_dataset.append(dataset)
            genre_index += 1

        return all_dataset

    @staticmethod
    def zip_again(all_dataset):
        # TODOx task
        all_after_zipped = []
        for dataset in all_dataset:
            zipped_dataset = zip(dataset.x_np, dataset.y_np, dataset.file_names)
            all_after_zipped.append(zipped_dataset)
        return all_after_zipped

    @staticmethod
    def merge_into_one_big_chunk(all_dataset):
        # TODOx task
        after_merge = []
        for dataset in all_dataset:
            after_merge.extend(dataset)
        return after_merge

    def divide_into_three_set(self, data):
        # TODOx task
        # Shuffle data
        shuffle(data)

        # Split data
        validation_nb = int(len(data) * config.validation_ratio)
        testNb = int(len(data) * config.test_ratio)
        trainNb = len(data) - (validation_nb + testNb)

        train_data = data[:trainNb]
        validation_data = data[trainNb:trainNb + validation_nb]
        test_data = data[-testNb:]

        x_train, y_train, fn_train = zip(*train_data)
        x_val, y_val, fn_val = zip(*validation_data)
        x_test, y_test, fn_test = zip(*test_data)

        # Prepare for Tflearn at the same time
        x_train = np.array(x_train).reshape([-1, config.slice_size, config.slice_size, 1])  # TODOx what is reshape?
        y_train = np.array(y_train)
        x_val = np.array(x_val).reshape([-1, config.slice_size, config.slice_size, 1])
        y_val = np.array(y_val)
        x_test = np.array(x_test).reshape([-1, config.slice_size, config.slice_size, 1])
        y_test = np.array(y_test)
        self.my_logger.info("[+] Dataset created! ✅")
        # TODOx fix those using divide_into_three_set
        return Dataset(x_train, y_train, fn_train), Dataset(x_val, y_val, fn_val), Dataset(x_test, y_test, fn_test)

    @staticmethod
    def save_the_three_dataset(train, validation, test):
        # TODO look inside
        DatasetHelper("Train_{}".format(config.run_id), config.dataset_path).save(train)
        DatasetHelper("Validation_{}".format(config.run_id), config.dataset_path).save(validation)
        DatasetHelper("Test_{}".format(config.run_id), config.dataset_path).save(test)
