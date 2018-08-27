import datetime


def get_current_time_c():
    current_time_l = datetime.datetime.now()
    current_time_string_l = current_time_l.strftime("%Y%m%d_%H%M")
    return current_time_string_l, current_time_l  # fixmex (second element returned)


# Name
nameOfUnknownGenre = "unknownGenre"
realTestDatasetPrefix = "testReal"
log_file_name = "myLog.log"
current_time_string, current_time = get_current_time_c()
run_id = "MusicGenres_" + current_time_string
my_logger_name = run_id
log_folder_name = "log/"
my_logger_file_name = log_folder_name + my_logger_name + ".log"
model_name_config = run_id
root_logger_file_name = log_folder_name + run_id + "_root.log"

# Define paths for files
path_to_spectrogram = "Data/Spectrograms/"
path_to_slices = "Data/Slices/"
path_to_test_slices = "Data/SlicesTest/"
dataset_path = "Data/Dataset/"
real_test_dataset_path = dataset_path + realTestDatasetPrefix + "/"
path_to_raw_data = "Data/Raw/"
path_to_test_data = "Data/Test/"
path_to_test_spectrogram = "Data/SpectrogramsTest/"
trainDataLabelPath = "Data/train.csv"
predictResultPath = "Data/PredictResult/"
file_names_path = "Data/FileNames/"

# Spectrogram resolution and Slices
pixelPerSecond = 50
desiredSliceSize = pixelPerSecond * 3
sliceSize = 128  # Slice parameters - this will be the size the original image is resized to
slices_per_genre_ratio = 1.0  # TODOx be careful. THis might cause memory error latter
slices_per_genre_ratio_each_genre = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, }

# Dataset parameters
validationRatio = 0.3
testRatio = 0.1

# Model parameters
batchSize = 128
learningRate = 0.001
nbEpoch = 8

# Debug
numberOfTrainRawFilesToProcessInDebugMode = 10
number_of_batches_debug = 10
number_of_real_test_files_debug = 256
number_of_slices_debug = 10

# Config tflearn
modelPath = "model/"
check_point_dir = "./checkpoint/"
checkpoint_path = "{}{}.ckpt".format(check_point_dir, run_id)
best_checkpoint_path = "{}{}_best.ckpt".format(check_point_dir, run_id)
tensorboard_dir = "tflearn_logs/"
tensorboard_verbose = 0
max_checkpoints = None  # TODO task test
# keep_checkpoint_every_n_hours = 10 / 60
best_val_accuracy = 0.70
show_metric = True
shuffle_data = True
snapshot_step = 100
snapshot_epoch = True

# Log
logging_formatter = '%(asctime)s:%(levelname)s:%(message)s:%(threadName)s:%(funcName)s'
time_formatter = '%Y%m%d %I:%M:%S %p'
log_file_mode = 'a'

# Rest
number_of_slices_before_informing_users = 1000
length_train_id = 10

# Multiprocessing
number_of_workers = 4  # TODOx task allow users to input number of worker as argument
