# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

import eyed3

from audioFilesTools import is_mono, get_genre
from config import desiredSliceSize, pixelPerSecond, unknown_genre, numberOfTrainRawFilesToProcessInDebugMode, \
    path_to_spectrogram, my_logger_name, run_id
from dataset.dataset_helper import check_path_exist
from sliceSpectrogram import create_slices_from_spectrogram

# Tweakable parameters


# Define
currentPath = os.path.dirname(os.path.realpath(__file__))

# Remove logs
eyed3.log.setLevel("ERROR")
my_logger = logging.getLogger(my_logger_name)


# Create spectrogram from mp3 files
def create_spectrogram_core(filename, new_filename, path_to_audio, spectrograms_path):
    # Create temporary mono track if needed
    # temp = os
    # temp2 = sys.path
    # sys.path.append("C:\Program Files (x86)\sox-14-4-2")
    mono = is_mono(path_to_audio + filename)
    if mono:
        command = "cp '{}' '/tmp/{}.mp3'".format(path_to_audio + filename, new_filename)
    else:
        # command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(rawDataPath + filename, newFilename)
        command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(path_to_audio + filename, new_filename)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    output, errors = p.communicate()
    if errors:
        my_logger.debug(errors)

    # Create spectrogram
    # filename.replace(".mp3", "") # TODOpro why do this? I comment out it. Be careful.
    command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(new_filename, pixelPerSecond,
                                                                                       spectrograms_path + new_filename)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    output, errors = p.communicate()
    if errors:
        my_logger.debug(errors)

    # Remove tmp mono track
    os.remove("/tmp/{}.mp3".format(new_filename))


# Creates .png whole spectrograms from mp3 files
def create_spectrogram(path_to_audio, spectrograms_path, user_args):
    genres_id = dict()
    files = os.listdir(path_to_audio)
    files = [file for file in files if file.endswith(".mp3")]
    nb_files = len(files)

    # Rename files according to genre
    # TODOx process debug
    for index, filename in enumerate(files):
        # TODOx process user_args
        my_logger.debug("Creating spectrogram for file {}/{}...".format(index + 1, nb_files))
        # TODOx user_args
        new_filename = get_spectrogram_name(filename, genres_id, index, path_to_audio, user_args)  # TODOx look inside
        # TODOx if spectrogram already exist, do not create
        file = Path('{}{}'.format(path_to_audio, new_filename))
        if file.exists():
            my_logger.debug("{} already exists so no spectrogram create!".format(new_filename))
        else:
            create_spectrogram_core(filename, new_filename, path_to_audio, spectrograms_path)  # TODOx look inside
        if user_args.debug and index >= numberOfTrainRawFilesToProcessInDebugMode:
            break


def get_spectrogram_name(filename, genres_id, index, path_to_audio, user_args):
    mode = user_args.mode
    genre_id = None
    file_id = None
    if "slice" in mode:
        genre_id = get_genre(path_to_audio + filename)
        genres_id[genre_id] = genres_id[genre_id] + 1 if genre_id in genres_id else 1
        file_id = genres_id[genre_id]
    elif "sliceTest" in mode:
        file_id = index + 1
        genre_id = unknown_genre
    new_filename = "{}_{}_{}_{}".format(run_id, genre_id, file_id, filename[:-4])
    return new_filename


# Whole pipeline .mp3 -> .png slices
def create_slices_from_audio(path_to_audio, spectrograms_path, slices_path, user_args):
    my_logger.info("Creating spectrograms...")
    # todox process user_args inside
    create_spectrogram(path_to_audio, spectrograms_path, user_args)  # TODOx look inside
    my_logger.info("Spectrograms created!")

    my_logger.info("Creating slices...")
    create_slices_from_spectrogram(desiredSliceSize, spectrograms_path, slices_path)  # TODOx look inside
    my_logger.info("Slices created!")


# Create path if not existing
check_path_exist(path_to_spectrogram)
