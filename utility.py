import argparse
import csv
import datetime
import logging
from sys import stdout

from config import predictResultPath, logging_formatter, time_formatter, log_file_name, log_file_mode


def process_file_name(file_name):
    split_result = file_name.split("_")
    return split_result[2], split_result[3]  # TODOx


def save_predict_result(predict_results, file_names, final_result):
    for i in range(len(predict_results)):
        predict_result = predict_results[i]
        file_name = file_names[i]
        # TODO debug value of predict_result and file_name
        file_name, slice_id = process_file_name(file_name)
        # TODO debug value of new file_name
        if file_name not in final_result:
            final_result[file_name] = {}
        result_of_particular_file = final_result[file_name]
        # TODO debug value of result_of_particular_file
        if predict_result not in result_of_particular_file:
            result_of_particular_file[predict_result] = 1
        else:
            result_of_particular_file[predict_result] += 1
    return final_result


def preprocess_predict_result(predict_results):
    new_result = []
    for result in predict_results:
        # TODO debug value of result
        new_result.append(result[0])
    return new_result  # TODOx


def finalize_result(final_result):
    file_names = list(final_result.keys())
    for filename in file_names:
        result = final_result[filename]
        genre = find_max_genre(result)
        final_result[filename] = genre
    return final_result  # TODOx


def get_current_time():
    x = datetime.datetime.now()
    x = x.strftime("%Y%m%d_%H%M")
    return x


def save_final_result(final_result, run_id):
    file_id = get_current_time()
    with open(predictResultPath + "{}_{}.csv".format(run_id, file_id), mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Id", "Genre"])
        for file_name, genre in final_result.items():
            csv_writer.writerow([file_name, genre])


def find_max_genre(result):
    genres = list(result.keys())
    first_genre = genres[0]
    final_genre = first_genre
    max_freq = result[first_genre]

    for genre in genres:
        freq = result[genre]
        if freq > max_freq:
            final_genre = genre
            max_freq = freq

    return final_genre  # TODOx


def set_up_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)

    formatter = logging.Formatter(logging_formatter, time_formatter)

    default_handler = root_logger.handlers[0]
    default_handler.setLevel(logging.WARNING)

    console = logging.StreamHandler(stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_name, log_file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)


def handle_args():
    global mode_arg, debug
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train",
                                                                                    "test",
                                                                                    "slice",
                                                                                    "sliceTest",
                                                                                    "testReal"])
    args = parser.parse_args()
    mode_arg = args.mode
    debug = args.debug
    return mode_arg, debug
