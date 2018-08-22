import csv
import datetime

from config import predictResultPath


def process_file_name(file_name):
    split_result = file_name.split("_")
    return split_result[2], split_result[3]  # TODOx


def save_predict_result(predict_results, file_names, final_result):
    for i in range(len(predict_results)):
        predict_result = predict_results[i]
        file_name = file_names[i]
        file_name, slice_id = process_file_name(file_name)
        if file_name not in final_result:
            final_result[file_name] = {}
        result = final_result[file_name]
        if predict_result not in result:
            result[predict_result] = 1
        else:
            result[predict_result] += 1
    return final_result


def preprocess_predict_result(predict_results):
    new_result = []
    for result in predict_results:
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
