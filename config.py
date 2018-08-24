import datetime


def get_current_time_c():
    x = datetime.datetime.now()
    x = x.strftime("%Y%m%d_%H%M")
    return x


nameOfUnknownGenre = "unknownGenre"
realTestDatasetPrefix = "testReal"
modelName = "musicDNN.tflearn"
log_file_name = "myLog.log"
run_id = "MusicGenres_" + get_current_time_c()
my_logger_name = run_id
log_folder_name = "log/"
my_logger_file_name = log_folder_name + my_logger_name + ".log"

# Define paths for files
spectrogramsPath = "Data/Spectrograms/"
slicesPath = "Data/Slices/"
slicesTestPath = "Data/SlicesTest/"
dataset_path = "Data/Dataset/"
real_test_dataset_path = dataset_path + realTestDatasetPrefix
rawDataPath = "Data/Raw/"
testDataPath = "Data/Test/"
spectrogramsTestPath = "Data/SpectrogramsTest/"
trainDataLabelPath = "Data/train.csv"
predictResultPath = "Data/PredictResult/"
file_names_path = "Data/FileNames/"
modelPath = "model/"

# Spectrogram resolution
pixelPerSecond = 50
desiredSliceSize = pixelPerSecond * 3
sliceSize = 128  # Slice parameters - this will be the size the original image is resized to

# percentage_of_real_test_slices = 0.01
# Dataset parameters
validationRatio = 0.3
testRatio = 0.1

# Model parameters
batchSize = 128
learningRate = 0.001
nbEpoch = 20

# slicesPerGenre = 1000  # This is the number of slices per gerne
slices_per_genre_ratio = 1.0  # TODOx be careful. THis might cause memory error latter
# TODOx task remove slies_per_genre_ratio
slices_per_genre_ratio_each_genre = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, }
# slices_per_genre_ratio_int = int(100 * slices_per_genre_ratio)
numberOfTrainRawFilesToProcessInDebugMode = 10
number_of_real_test_files_debug = 256
number_of_slices_debug = 10
number_of_slices_before_informing_users = 1000

length_train_id = 10
number_of_batches_debug = 10

logging_formatter = '%(asctime)s:%(levelname)s:%(message)s:%(threadName)s:%(funcName)s'
time_formatter = '%Y%m%d %I:%M:%S %p'
log_file_mode = 'a'
