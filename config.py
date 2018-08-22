#Define paths for files
spectrogramsPath = "Data/Spectrograms/"
slicesPath = "Data/Slices/"
slicesTestPath = "Data/SlicesTest/"
dataset_path = "Data/Dataset/"
rawDataPath = "Data/Raw/"
testDataPath = "Data/Test/"
spectrogramsTestPath = "Data/SpectrogramsTest/"
trainDataLabelPath = "Data/train.csv"
predictResultPath = "Data/predictResult.csv"
file_names_path = "Data/FileNames/"
modelPath = "model/"


#Spectrogram resolution
pixelPerSecond = 50
desiredSliceSize = pixelPerSecond * 3
sliceSize = 128  # Slice parameters - this will be the size the original image is resized to

# percentage_of_real_test_slices = 0.01
#Dataset parameters
validationRatio = 0.3
testRatio = 0.1

#Model parameters
batchSize = 128
learningRate = 0.001
nbEpoch = 20

slicesPerGenre = 1000  # This is the number of slices per gerne
numberOfTrainRawFilesToProcessInDebugMode = 50
length_train_id = 10
number_of_batches_debug = 10

nameOfUnknownGenre = "unknownGenre"
realTestDatasetPrefix = "testReal"
modelName = "musicDNN.tflearn"
