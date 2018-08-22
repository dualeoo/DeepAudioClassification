# -*- coding: utf-8 -*-
import os
from subprocess import Popen, PIPE, STDOUT

import eyed3

from audioFilesTools import isMono, getGenre
from config import desiredSliceSize, pixelPerSecond, nameOfUnknownGenre, numberOfTrainRawFilesToProcessInDebugMode, \
    spectrogramsPath
from datasetTools import check_path_exist
from main import debug
from sliceSpectrogram import createSlicesFromSpectrograms

# Tweakable parameters


# Define
currentPath = os.path.dirname(os.path.realpath(__file__))

# Remove logs
eyed3.log.setLevel("ERROR")


# Create spectrogram from mp3 files
def createSpectrogram(filename, newFilename, pathToAudio, spectrogramsPath):
    # Create temporary mono track if needed
    # temp = os
    # temp2 = sys.path
    # sys.path.append("C:\Program Files (x86)\sox-14-4-2")
    mono = isMono(pathToAudio + filename)
    if mono:
        command = "cp '{}' '/tmp/{}.mp3'".format(pathToAudio + filename, newFilename)
    else:
        # command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(rawDataPath + filename, newFilename)
        command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(pathToAudio + filename, newFilename)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    output, errors = p.communicate()
    if errors:
        print(errors)

    # Create spectrogram
    # filename.replace(".mp3", "") # TODOpro why do this? I comment out it. Be careful.
    command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(newFilename, pixelPerSecond,
                                                                                       spectrogramsPath + newFilename)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    output, errors = p.communicate()
    if errors:
        print(errors)

    # Remove tmp mono track
    os.remove("/tmp/{}.mp3".format(newFilename))


# Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio(pathToAudio, spectrogramsPath, mode):
    genresID = dict()
    files = os.listdir(pathToAudio)
    files = [file for file in files if file.endswith(".mp3")]
    nbFiles = len(files)

    # Rename files according to genre
    if not debug:
        for index, filename in enumerate(files):
            get_file_name_and_create_spectrogram(filename, genresID, index, mode, nbFiles, pathToAudio,
                                                 spectrogramsPath)  # TODOx look inside
    else:
        for index, filename in enumerate(files):
            get_file_name_and_create_spectrogram(filename, genresID, index, mode, nbFiles, pathToAudio,
                                                 spectrogramsPath)
            if index >= numberOfTrainRawFilesToProcessInDebugMode: break


def get_file_name_and_create_spectrogram(filename, genresID, index, mode, nbFiles, pathToAudio, spectrogramsPath):
    print("Creating spectrogram for file {}/{}...".format(index + 1, nbFiles))
    newFilename = getNewFileName(filename, genresID, index, mode, pathToAudio)  # TODOx look inside
    createSpectrogram(filename, newFilename, pathToAudio, spectrogramsPath)  # TODOx look inside


def getNewFileName(filename, genresID, index, mode, pathToAudio):
    newFilename = ""
    if "slice" in mode:
        fileGenre = getGenre(pathToAudio + filename)
        genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
        fileID = genresID[fileGenre]
        newFilename = fileGenre + "_" + str(fileID)
    elif "sliceTest" in mode:
        fileID = index + 1
        newFilename = nameOfUnknownGenre + "_" + str(fileID) + "_" + filename
    return newFilename


# Whole pipeline .mp3 -> .png slices
def createSlicesFromAudio(pathToAudio, spectrogramsPath, mode, slicesPath):
    print("Creating spectrograms...")
    createSpectrogramsFromAudio(pathToAudio, spectrogramsPath, mode)  # TODOx look inside
    print("Spectrograms created!")

    print("Creating slices...")
    createSlicesFromSpectrograms(desiredSliceSize, spectrogramsPath, slicesPath)  # TODOx look inside
    print("Slices created!")


# Create path if not existing
check_path_exist(spectrogramsPath)
