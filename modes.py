# -*- coding: utf-8 -*-
import csv
import logging
import math
import multiprocessing
import os
from pathlib import Path
from random import shuffle
from subprocess import Popen, PIPE, STDOUT
from typing import Dict, List

import eyed3
import numpy as np
from PIL import Image

import MainHelper
import config
from MainHelper import check_path_exist, log_time_start, log_time_end
from dataset import GetDataset, DatasetHelper, Dataset
from model import create_model

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
        # TODOx be careful, how comes I comment out the line above?
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
            # my_logger.info("Creating spectrogram for file {}/{}...".format(index + 1, nb_files))
            new_filename = self.get_spectrogram_name(filename, genres_id)  # TODOx look inside
            # fixmeX how comes path is Data/train/MusicGenres_20180828_2320_6_1_8339573394770541951
            # fixmeX it would always create new file
            file = Path('{}{}'.format(self.spectrograms_path, new_filename))
            if file.exists():
                my_logger.info("{} already exists so no spectrogram create!".format(new_filename))
            else:
                worker = pool.apply_async(self.create_spectrogram_core,
                                          args=(filename, new_filename))
                workers.append(worker)
            if self.user_args.debug and index >= config.numberOfTrainRawFilesToProcessInDebugMode:
                break

        index = 0
        for worker in workers:
            my_logger.info("Creating spectrogram for file {}/{}...".format(index + 1, nb_files))
            worker.wait()
            index += 1

    def get_spectrogram_name(self, filename, genres_id):
        mode = self.user_args.mode
        genre_id = None
        if config.name_of_mode_create_spectrogram == mode:
            genre_id = self.get_genre(self.path_to_audio + filename)
            genres_id[genre_id] = genres_id[genre_id] + 1 if genre_id in genres_id else 1
        elif config.name_of_mode_create_spectrogram_for_test_data == mode:
            genre_id = config.unknown_genre
        new_filename = "{}_{}".format(genre_id, filename[:-4])
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
        workers = []
        for filename in file_names:
            if filename.endswith(".png"):
                # TODOx be careful with the create spectrogram multi processing just now, i didnt join (wait)
                worker = pool.apply_async(self.slice_spectrogram, (filename,))
                workers.append((worker, index))
                index += 1

        for w in workers:
            worker = w[0]
            index = w[1]
            worker.wait()
            my_logger.info("Finish slicing for file {}/{}".format(index, len(file_names)))

        pool.close()
        pool.join()

    # Creates slices from spectrogram
    # Author_TODO Improvement - Make sure we don't miss the end of the song
    def slice_spectrogram(self, filename):
        # fixmeX after I change name of spectrogram
        # fixmeX
        split_results = filename.split("_")
        genre = split_results[0]  # {genre id}_{song file name}.png
        # song_id = split_results[2]
        song_name = split_results[1]

        # Load the full spectrogram
        img = Image.open(self.spectrograms_path + filename)

        # Compute approximate number of 128x128 samples
        width, height = img.size
        nb_samples = int(width / self.desired_slice_size)
        # width - desiredSliceSize  # TODOpro Why do this? I comment out it. Be careful

        # Create path if not existing
        path_slice_of_specific_genre = self.slices_path + "{}/".format(genre)
        check_path_exist(path_slice_of_specific_genre)

        # For each sample
        for slice_id in range(nb_samples):
            # my_logger.info("Creating slice: ", (i + 1), "/", nb_samples, "for", filename)
            # Extract and save 128x128 sample
            start_pixel = slice_id * self.desired_slice_size
            img_tmp = img.crop((start_pixel, 1, start_pixel + self.desired_slice_size, self.desired_slice_size + 1))
            # TODOx why [:-4]? to remove .png
            img_tmp.save(path_slice_of_specific_genre + "{}_{}_{}.png".format(genre, song_name, slice_id))


class Training:
    def __init__(self, user_args, genres, path_to_model, nb_classes, active_config) -> None:
        self.user_args = user_args
        self.genres = genres
        self.my_logger = my_logger
        self.path_to_model = path_to_model
        self.nb_classes = nb_classes
        self.model = create_model(nb_classes, config.slice_size)
        self.active_config = active_config

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
            dataset_name = "{}_{}".format(genre, self.user_args.run_id)
            # TODOx look inside
            # fixmeX
            # TODOx run inspection
            GetDataset(genre, config.slice_size, config.dataset_path, dataset_name
                       , self.active_config.path_to_slices_for_training, self.user_args, self.genres).start()
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
            dataset_name = "{}_{}".format(genre, self.user_args.run_id)
            dataset = DatasetHelper(dataset_name, config.dataset_path).load()
            all_dataset.append(dataset)
            genre_index += 1

        return all_dataset

    def zip_again(self, all_dataset):
        self.my_logger.info("[+] Start zipping all files")
        # TODOx task
        array_of_array_of_data_points = []
        for dataset in all_dataset:
            zipped_dataset = zip(dataset.x_np, dataset.y_np, dataset.file_names)
            array_of_array_of_data_points.append(zipped_dataset)
        self.my_logger.info("[+] Finish zipping all files")
        return array_of_array_of_data_points

    def divide_into_three_set(self, data):
        # TODOx task
        # Shuffle data
        self.my_logger.info("[+] Start dividing into Train, Validation, and Test")
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
        self.my_logger.info("[+] Finish dividing into three sets! ✅")
        # TODOx fix those using divide_into_three_set
        return Dataset(x_train, y_train, fn_train), Dataset(x_val, y_val, fn_val), Dataset(x_test, y_test, fn_test)

    def save_the_three_dataset(self, train, validation, test):
        # TODOx look inside
        DatasetHelper("Train_{}".format(self.user_args.run_id), config.dataset_path).save(train)
        DatasetHelper("Validation_{}".format(self.user_args.run_id), config.dataset_path).save(validation)
        DatasetHelper("Test_{}".format(self.user_args.run_id), config.dataset_path).save(test)


class Test:
    def __init__(self, user_args: MainHelper.UserArg, dataset, model, path_to_model: str) -> None:
        self.user_args = user_args
        self.dataset = dataset
        self.my_logger = my_logger
        self.model = model
        self.path_to_model = path_to_model
        self.load_model()

    def predict(self):
        time_starting = MainHelper.log_time_start("predict")

        x_np = self.dataset.x_np
        # file_names = self.dataset.file_names

        starting_index = 0
        ending_index = config.nb_data_points_per_batch
        x_np_size = x_np.shape[0]
        nm_of_batches = math.ceil(x_np_size / config.nb_data_points_per_batch)
        final_result = {}

        for i in range(nm_of_batches):
            self.my_logger.info("[+] Start predicting batch {}/{}".format(i + 1, nm_of_batches))
            # noteX can be wrong
            # TODOx increase batch size next time predict, probably 1024
            active_x_np = x_np[starting_index:ending_index, ...]
            predict_result = self.model.predict(active_x_np)
            self.group_slices_of_same_song(predict_result, final_result, i)
            self.my_logger.info("[+] Finish predicting batch {}/{}".format(i + 1, nm_of_batches))
            starting_index += config.nb_data_points_per_batch
            ending_index += config.nb_data_points_per_batch

        self.my_logger.info("[+] tflearn finish predicting!")
        finalized_result = self.finalize_result(final_result)  # TODOx look inside
        self.save_finalized_result(finalized_result)  # TODOx look inside

        MainHelper.log_time_end(self.user_args.mode, time_starting)

    @staticmethod
    def find_max_genre(results: List[List[float]]) -> int:
        probability_for_each_genre = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, }
        for result in results:
            for index, probability in enumerate(result):
                probability_for_each_genre[index + 1] += probability

        final_genre = list(probability_for_each_genre.keys())[0]
        max_probability = probability_for_each_genre[final_genre]

        for genre, probability_for_a_genre in probability_for_each_genre.items():
            if probability_for_a_genre > max_probability:
                final_genre = genre
                max_probability = probability_for_a_genre

        return final_genre  # TODOx

    def finalize_result(self, final_result: Dict[str, List[List[float]]]) -> Dict[str, int]:
        self.my_logger.info("[+] Start finalize result!")
        finalized_results = {}
        file_names = list(final_result.keys())
        for filename in file_names:
            results = final_result[filename]
            genre = self.find_max_genre(results)  # TODOx task look inside
            finalized_results[filename] = genre
        self.my_logger.info("[+] Done finalize result!")
        return finalized_results  # TODOx

    @staticmethod
    def process_file_name(file_name):
        split_result = file_name.split("_")  # 1_4728348676381658827.png_23.png
        return split_result[0], split_result[1], split_result[2][:-4]  # TODOx

    def group_slices_of_same_song(self, predict_results: List[List[float]],
                                  final_result: Dict[str, List[List[float]]],
                                  batch_number):
        # TODOx debug value of all variables in this method
        self.my_logger.info("[+] Start group slices of same song for batch {}!".format(batch_number + 1))
        # final_result = {}
        for i in range(len(predict_results)):
            predict_result = predict_results[i]
            corresponding_file_id = batch_number * config.nb_data_points_per_batch + i
            file_name = self.dataset.file_names[corresponding_file_id]
            # fixmeX process_file_name
            genre, file_name, slice_id = self.process_file_name(file_name)

            if file_name not in final_result:
                final_result[file_name] = []
            result_for_a_song = final_result[file_name]
            result_for_a_song.append(predict_result)
        self.my_logger.info("[+] Done group slices of same song!")

    def save_finalized_result(self, final_result: Dict[str, int]):
        # fixmeX make the name of the test set used saved
        run_id_for_predicted_result_file = self.get_run_id_for_predicted_result_file()
        path_to_save_result = config.predict_result_path + "{}_{}.csv".format(run_id_for_predicted_result_file,
                                                                              self.user_args.mode)
        self.my_logger.info("[+] Saving result to {}!".format(path_to_save_result))
        with open(path_to_save_result, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Id", "Genre"])
            for file_name, genre in final_result.items():
                csv_writer.writerow([file_name, genre])
        self.my_logger.info("[+] Done result to {}!".format(path_to_save_result))

    def get_run_id_for_predicted_result_file(self):
        if self.user_args.mode == "test":
            run_id_for_predicted_result_file = self.user_args.run_id
        else:
            run_id_for_predicted_result_file = self.user_args.run_id_real_test
        return run_id_for_predicted_result_file

    def load_model(self):
        self.my_logger.info("[+] Loading weights...")
        self.model.load(self.path_to_model)
        self.my_logger.info("[+] Weights loaded! ✅")

    def evaluate(self):
        time_starting = MainHelper.log_time_start(self.user_args.mode)

        test_X = self.dataset.x_np
        test_y = self.dataset.y_np

        testAccuracy = self.model.evaluate(test_X, test_y)[0]
        self.my_logger.info("[+] Test accuracy: {} ".format(testAccuracy))

        MainHelper.log_time_end(self.user_args.mode, time_starting)

    def evaluate_whole_song(self):
        self.my_logger.info("[+] Start evaluating song prediction result")
        predicted_result = self.load_predicted_result()
        truth = self.load_truth()
        nb_results = len(predicted_result)
        nb_correct = 0
        for song_name, song_genre in predicted_result.items():
            true_genre = truth[song_name]
            # TODOx be careful number and string of genre
            if true_genre == song_genre:
                nb_correct += 1
        accuracy = nb_correct / nb_results
        self.my_logger.info("The model accurately predict {}% of songs!".format(accuracy * 100))

    def load_predicted_result(self) -> Dict[str, int]:
        # fixmeX when I use this method to load predicted_result of testReal
        run_id_for_predicted_result = self.get_run_id_for_predicted_result_file()
        path_to_save_result = config.predict_result_path + "{}_{}.csv".format(run_id_for_predicted_result,
                                                                              self.user_args.mode)
        return self.load_csv_file(path_to_save_result,
                                  "[+] Loading predicted result to memory ({})!",
                                  "[+] Finish loading predicted result to memory!", True)

    def load_truth(self):
        return self.load_csv_file(config.train_data_label_path,
                                  "[+] Loading truth to memory ({})!",
                                  "[+] Finish loading truth to memory!", False)

    def load_csv_file(self, path_to_csv_file, begin_message, end_message, contain_header):
        self.my_logger.info(begin_message.format(path_to_csv_file))
        result = {}
        with open(path_to_csv_file, mode='r') as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if contain_header:
                next(csv_reader)
            for row in csv_reader:
                # fixmeX for trainLabel, doesnt have header row
                song_name = row[0][:-4]
                genre = row[1]
                result[song_name] = genre
        self.my_logger.info(end_message)
        return result

    def rearrange_result_file(self):
        # TODOx look inside
        predicted_results = self.load_predicted_result()
        sample_submission = self.load_sample_submission()

        path_to_save_result = config.predict_result_path \
                              + "{}_{}_rearranged.csv".format(self.get_run_id_for_predicted_result_file(),
                                                              self.user_args.mode)

        self.my_logger.info("[+] Saving result to {}!".format(path_to_save_result))
        with open(path_to_save_result, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Id", "Genre"])
            for file_name in sample_submission:
                csv_writer.writerow([file_name, predicted_results[file_name]])
        self.my_logger.info("[+] Done saving result to {}!".format(path_to_save_result))

    def load_sample_submission(self) -> List[str]:
        # TODOx
        path_to_csv_file = config.path_to_sample_submission
        self.my_logger.info("[+] Start loading submission result to memory ({})".format(path_to_csv_file))
        result = []
        with open(path_to_csv_file, mode='r') as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if True:
                next(csv_reader)
            for row in csv_reader:
                file_name = row[0]
                result.append(file_name)
        self.my_logger.info("[+] Finish loading submission result")
        return result
