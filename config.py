# TODO recheck this section one day
import datetime


def get_current_time():
    current_time_l = datetime.datetime.now()
    current_time_string_l = current_time_l.strftime("%Y%m%d_%H%M")
    return current_time_string_l, current_time_l  # fixmex (second element returned)


unknown_genre = "unknownGenre"
real_test_prefix = "testReal"
name_of_mode_create_spectrogram = "spectrogram"
name_of_mode_create_spectrogram_for_test_data = "spectrogramTest"
log_file_name = "myLog.log"
current_time_string, current_time = get_current_time()
run_id = "MusicGenres_" + current_time_string
my_logger_name = run_id
log_folder_name = "log/"
my_logger_file_name = log_folder_name + my_logger_name + ".log"
model_name_config = run_id
root_logger_file_name = log_folder_name + run_id + "_root.log"
real_test_dataset_name = "{}_X_{}".format(real_test_prefix, run_id)

# Define paths for files
path_to_raw_data = "Data/train/"
path_to_test_data = "Data/test/"

path_to_spectrogram = "Data/Spectrograms/{}/"
# fixmeX
path_to_test_spectrogram = "Data/SpectrogramsTest/{}/"

# fixmeX
path_to_slices_for_training = "Data/Slices/{}/"
path_to_slices_for_testing = "Data/SlicesTest/{}/"

dataset_path = "Data/Dataset/"

train_data_label_path = "Data/train.csv"
predict_result_path = "Data/PredictResult/"

# Spectrogram resolution and Slices
pixel_per_second = 50
desired_slice_size = pixel_per_second * 3
slice_size = 128  # Slice parameters - this will be the size the original image is resized to
slices_per_genre_ratio = 1.0  # TODOx be careful. THis might cause memory error latter
slices_per_genre_ratio_each_genre = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0.5, 9: 1, 10: 1, }

# Dataset parameters
validation_ratio = 0.3
test_ratio = 0.1

# Model parameters
batchSize = 128
learningRate = 0.001
nbEpoch = 8

# Debug
numberOfTrainRawFilesToProcessInDebugMode = 10
number_of_slices_debug = 256

# Config tflearn
# TODO rethink about tflearn param one day
modelPath = "model/"
check_point_dir = "./checkpoint/"
checkpoint_path = "{}{}.ckpt".format(check_point_dir, run_id)
best_checkpoint_path = "{}{}_best.ckpt".format(check_point_dir, run_id)
tensorboard_dir = "tflearn_logs/"
tensorboard_verbose = 0
max_checkpoints = None  # TODO test case max_checkpoints = 1
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

# Predict
nb_data_points_per_patch = 256
