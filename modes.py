# -*- coding: utf-8 -*-
import csv
import logging
import multiprocessing
import os
from pathlib import Path
from random import shuffle
from subprocess import Popen, PIPE, STDOUT

import eyed3
import numpy as np
from PIL import Image

import config
import utility
from config import run_id
from dataset import GetDataset, DatasetHelper, Dataset
from model import create_model
from utility import check_path_exist, log_time_start, log_time_end

currentPath = os.path.dirname(os.path.realpath(__file__))
# Remove logs
eyed3.log.setLevel("ERROR")
my_logger = logging.getLogger(config.my_logger_name)


class CreateSpectrogram:
    @staticmethod
    def is_mono(filename):
        audio_file = eyed3.load(filename)
        return audio_file.info.mode == 'Mono'

    def get_genre(self, file_path):
        # TODOx re-implement
        # audiofile = eyed3.load(filename)
        # TODO be careful, how comes I comment out the line above?
        filename = (file_path.split("/"))[2]
        genre = self.labelDic[filename]
        if genre:
            return genre
        else:
            raise Exception("There is no file named {} in train.csv".format(filename))

    def __init__(self, path_to_audio, spectrograms_path, user_args) -> None:
        self.user_args = user_args
        self.path_to_audio = path_to_audio
        self.spectrograms_path = spectrograms_path
        self.labelDic = self.initialize_label_dict()

    def create_spectrogram_core(self, filename, new_filename):
        # Create temporary mono track if needed
        # temp = os
        # temp2 = sys.path
        # sys.path.append("C:\Program Files (x86)\sox-14-4-2")
        mono = self.is_mono(self.path_to_audio + filename)

        if mono:
            command = "cp '{}' '/tmp/{}.mp3'".format(self.path_to_audio + filename, new_filename)
        else:
            command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(self.path_to_audio + filename, new_filename)

        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
        output, errors = p.communicate()
        if errors:
            my_logger.error("Errors when using Popen")
            my_logger.error(errors)
            exit()

        # Create spectrogram
        # filename.replace(".mp3", "") # TODOpro why do this? I comment out it. Be careful.
        command = "sox '/tmp/{}.mp3' -n spectrogram " \
                  "-Y 200 -X {} -m -r -o '{}.png'".format(new_filename, config.pixel_per_second,
                                                          self.spectrograms_path + new_filename)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
        output, errors = p.communicate()
        if errors:
            my_logger.error("Errors when using Popen")
            my_logger.error(errors)

        # Remove tmp mono track
        os.remove("/tmp/{}.mp3".format(new_filename))

    def start(self):
        genres_id = dict()
        files = os.listdir(self.path_to_audio)
        files = [file for file in files if file.endswith(".mp3")]
        nb_files = len(files)

        # TODOx make this multi-processing
        # TODOx look at all files currently present in the project
        pool = multiprocessing.Pool(processes=os.cpu_count())
        workers = []
        for index, filename in enumerate(files):
            my_logger.info("Creating spectrogram for file {}/{}...".format(index + 1, nb_files))
            new_filename = self.get_spectrogram_name(filename, genres_id, index)  # TODOx look inside
            # TODO show comes path is Data/train/MusicGenres_20180828_2320_6_1_8339573394770541951
            file = Path('{}{}'.format(self.path_to_audio, new_filename))
            if file.exists():
                my_logger.info("{} already exists so no spectrogram create!".format(new_filename))
            else:
                worker = pool.apply_async(self.create_spectrogram_core,
                                          args=(filename, new_filename))
                workers.append(worker)
            if self.user_args.debug and index >= config.numberOfTrainRawFilesToProcessInDebugMode:
                break

        for worker in workers:
            worker.wait()

    def get_spectrogram_name(self, filename, genres_id, index):
        mode = self.user_args.mode
        genre_id = None
        file_id = None
        if "slice" in mode:
            genre_id = self.get_genre(self.path_to_audio + filename)
            genres_id[genre_id] = genres_id[genre_id] + 1 if genre_id in genres_id else 1
            file_id = genres_id[genre_id]
        elif "sliceTest" in mode:
            file_id = index + 1
            genre_id = config.unknown_genre
        new_filename = "{}_{}_{}_{}".format(config.run_id, genre_id, file_id, filename[:-4])
        return new_filename

    @staticmethod
    def initialize_label_dict():
        labelDic = dict()

        with open(config.train_data_label_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                labelDic[row[0]] = row[1]

        return labelDic


class CreateSlice:
    def __init__(self, desired_slice_size, spectrograms_path, slices_path) -> None:
        # fixmeX
        self.desired_slice_size = desired_slice_size
        self.spectrograms_path = spectrograms_path
        self.slices_path = slices_path

    def start(self):
        file_names = os.listdir(self.spectrograms_path)
        index = 1

        # TODOx make this multiprocessing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        for filename in file_names:
            if filename.endswith(".png"):
                # TODOx be careful with the create spectrogram multi processing just now, i didnt join (wait)
                pool.apply_async(self.slice_spectrogram, (filename,),
                                 callback=lambda: my_logger.info(
                                     "Finish slicing for file {}/{}".format(index, len(file_names))))
                # workers.append((worker, index))
                index += 1

        # for w in workers:
        #     worker = w[0]
        #     index = w[1]
        #     worker.wait()
        #     my_logger.info("Finish slicing for file {}/{}".format(index, len(file_names)))

        pool.close()
        pool.join()

    # Creates slices from spectrogram
    # Author_TODO Improvement - Make sure we don't miss the end of the song
    def slice_spectrogram(self, filename):
        # fixmeX after I change name of spectrogram
        split_results = filename.split("_")
        genre = split_results[1]  # {run ID}_{genre id}_{song id}_{song file name}.png
        song_id = split_results[2]
        song_name = split_results[3]

        # Load the full spectrogram
        img = Image.open(self.spectrograms_path + filename)

        # Compute approximate number of 128x128 samples
        width, height = img.size
        nb_samples = int(width / self.desired_slice_size)
        # width - desiredSliceSize  # TODOpro Why do this? I comment out it. Be careful

        # Create path if not existing
        slice_path = self.slices_path + "{}/".format(genre)
        check_path_exist(slice_path)

        # For each sample
        for slice_id in range(nb_samples):
            # my_logger.info("Creating slice: ", (i + 1), "/", nb_samples, "for", filename)
            # Extract and save 128x128 sample
            start_pixel = slice_id * self.desired_slice_size
            img_tmp = img.crop((start_pixel, 1, start_pixel + self.desired_slice_size, self.desired_slice_size + 1))
            # TODOx why [:-4]? to remove .png
            img_tmp.save(self.slices_path + "{}_{}_{}_{}_{}.png".format(run_id, genre, song_id, song_name, slice_id))


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
            # TODOx look inside
            dataset = GetDataset(genre, config.slice_size, config.dataset_path, dataset_name
                                 , config.path_to_slices_for_training, self.user_args, self.genres).start()

            # fixmeX
            DatasetHelper(dataset_name, config.dataset_path).save(dataset)
            genre_index += 1

        # TODOx look inside
        all_dataset = self.load_dataset_back_to_memory()
        # TODOx look inside
        array_of_array_of_data_points = self.zip_again(all_dataset)
        array_of_data_points = [data_point for array_of_data_points in array_of_array_of_data_points
                                for data_point in array_of_data_points]
        return array_of_data_points

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
        array_of_array_of_data_points = []
        for dataset in all_dataset:
            zipped_dataset = zip(dataset.x_np, dataset.y_np, dataset.file_names)
            array_of_array_of_data_points.append(zipped_dataset)
        return array_of_array_of_data_points

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
        # TODOx look inside
        DatasetHelper("Train_{}".format(config.run_id), config.dataset_path).save(train)
        DatasetHelper("Validation_{}".format(config.run_id), config.dataset_path).save(validation)
        DatasetHelper("Test_{}".format(config.run_id), config.dataset_path).save(test)


class Test:
    def __init__(self, user_args, dataset, my_logger, model, path_to_model) -> None:
        self.user_args = user_args
        self.dataset = dataset
        self.my_logger = my_logger
        self.model = model
        self.path_to_model = path_to_model
        self.load_model()

    def predict(self):
        time_starting = utility.log_time_start(self.user_args.mode)

        x_np = self.dataset.x_np
        file_names = self.dataset.file_names

        final_result = {}
        predict_result = self.model.predict(x_np)

        self.my_logger.warning("Stop predicting because starting from this point forward coudd be wrong. "
                               "Please start debugging!")
        exit()

        # TODO be careful. Starting from here could be wrong
        predict_result = utility.preprocess_predict_result(predict_result)  # TODOx look inside
        # TODO change the name of method save_predict_result
        utility.save_predict_result(predict_result, file_names, final_result)  # TODOx look inside
        final_result = utility.finalize_result(final_result)  # TODOx look inside
        utility.save_final_result(final_result)  # TODOx look inside

        utility.log_time_end(self.user_args.mode, time_starting)

    def load_model(self):
        self.my_logger.info("[+] Loading weights...")
        self.model.load(self.path_to_model)
        self.my_logger.info("[+] Weights loaded! ✅")

    def evaluate(self):
        time_starting = utility.log_time_start(self.user_args.mode)

        test_X = self.dataset.x_np
        test_y = self.dataset.y_np

        testAccuracy = self.model.evaluate(test_X, test_y)[0]
        self.my_logger.info("[+] Test accuracy: {} ".format(testAccuracy))

        utility.log_time_end(self.user_args.mode, time_starting)
