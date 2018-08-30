# Deep Audio Classification
## 1. Big picture
### Training
```
def train(List[mp3 songs]) -> model:  
    spectrograms = create_spectrogram(List[mp3 songs])  
    
    for spectrogram in spectrograms:  
        list_of_slices = slice(spectrogram)  
    
    for slice in list_of_slices:  
        data_point  = convert_to_numpy_array(slice)
    
    train_x, train_y, val_x, val_y, test_x, test_y = split_data(all_data_points)
    
    model.fit(train_x, train_y, val_x, val_y, test_x, test_y)
```

To run the training: `python main.py --run-id={to fill in}`. `--run-id` is actually the name of the folder 
containing all the slices. For example, if the that folder is `Data/Slices/MusicGenres_20180829_1017`, 
the value to pass to `--run-id` is `MusicGenres_20180829_1017`. 
### Testing

To test the model on the `test_x` and `test_y` datasets created above, call `python main.py test test --run-id={to fill 
 in} --model={to fill in}`. `--run-id` is the name of the folder containing all the slices for **testing**. Model
 is the name of the model created from the training step.
 
For example `python main.py test --run-id=MusicGenres_20180829_1017 
 --model=MusicGenres_20180825_0715` 
 
To predict the real test set, call: `python main.py testReal --run-id={to fill in} --model={to fill in} 
--run-id-test-real={to fill in}`. `--run-id` is the name of the folder containing all the slices for **training**. 
`--run-id-test-real` is the name of the the folder containing all the slices for **testing**.

For example: `testReal --run-id=MusicGenres_20180829_1017 
--model=MusicGenres_20180825_0715 --run-id-test-real=MusicGenres_20180829_2101`

## 2. Required install:

All packages required are listed in `requirements.txt` file

However, the program depends on an external program: **sox** and **libsox-fmt-mp3**. To install them on Linux, run 
`sudo apt instlal sox` and `sudo apt-get install libsox-fmt-mp3`. You can run the Dockerfile attached to automate this 
process.

Please put all the train data under `Data/train/` and all the real test data at `Data/test`.

## 3. Notice
Most editable parameters are in the config.py file, the model can be changed in the model.py file.
