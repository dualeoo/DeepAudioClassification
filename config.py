#Define paths for files
spectrogramsPath = "Data/Spectrograms/"
slicesPath = "Data/Slices/"
slicesTestPath = "Data/SlicesTest/"
datasetPath = "Data/Dataset/"
rawDataPath = "Data/Raw/"
testDataPath = "Data/Test/"
spectrogramsTestPath = "Data/SpectrogramsTest/"
trainDataLabelPath = "Data/train.csv"
realTestDataFileNamesPath = "Data/realTestNames.csv"
predictResultPath = "Data/realTestNames.csv"

#Spectrogram resolution
pixelPerSecond = 50

# Slice parameters - this will be the size the original image is resized to
sliceSize = 128

#Dataset parameters
slicesPerGenre = 1000  # This is the number of slices per gerne
validationRatio = 0.3
testRatio = 0.1

#Model parameters
batchSize = 128
learningRate = 0.001
nbEpoch = 20

desiredSliceSize = pixelPerSecond * 3

nameOfUnknownGenre = "unknownGenre"
realTestDatasetPrefix = "testReal"
