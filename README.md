Team NNoM


### How to train model
> Please skip if you are not willing to train a new model. 

1. Copy `tinyml_contest_data_training` to this folder.

2. Run `dataset.py`

3. Run `main.py`

Tensorflow model will be generated under `model` folder.   

NNoM model `weights_contest.h` will be generated under this folder.

### Run on MCU

Once we have the `weights_contest.h` generated, go to `tinyml-contest-mcu`, open the keil project file, compile and download to the NUCLEO board. 

Then run validation.py from the contest repository to validate the result. 

### Run NNoM on PC

Install Visual Studio, open the `contest.sln` to run it from VS. 

OR

Install `scons` and `GCC or msvc`, then run `scons` in this folder. 
