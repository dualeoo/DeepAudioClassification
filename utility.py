import csv

from config import realTestDataFileNamesPath, predictResultPath


def save_predict_result(predictResults):
    # TODOx task
    with open(predictResultPath, 'w') as f:
        writer = csv.writer(f)
        for result in predictResults:
            writer.writerows(result)


def save_file_names(file_names):
    # TODOx
    with open(realTestDataFileNamesPath, 'w') as f:
        writer = csv.writer(f)
        for file_name in file_names:
            writer.writerows([file_name])
