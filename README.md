**Team NNoM submission for TinyML Contest** ([2022 ACM/IEEE TinyML Design Contest at ICCAD](https://tinymlcontest.github.io/TinyML-Design-Contest/index.html))

## Model

This model consist of a few 1D-Convolution layer (Inception structure) and 1 GRU recurrent layer. 

The total trainable parameters is just under `2k` a tiny model that can achieve 95+ score. The model is quantised into 8-bit ops and accelerated with CMSIS-NN backend. Can inference one test data in `34ms @ 80MHz Cortex M4`, with the complexity only `220k MAC`. 

Code size

`Program Size: Code=29944 RO-data=6376 RW-data=24 ZI-data=52784  `

### How to train model (optional)

> Please skip if you are not willing to train a new model. 

1. Copy `tinyml_contest_data_training` to this folder.

2. Run `dataset.py`

3. Run `main.py` to train, quantise and covert the mode to NNoM model file. 

Tensorflow model will be generated under `model` folder.   

NNoM model `weights_contest.h` will be generated under this folder.

### Run on MCU

Once we have the `weights_contest.h` generated, go to `tinyml-contest-mcu`, open the Keil project file, compile and download to the NUCLEO board. 

Then run `validation.py` from the contest repository to validate the result. 

### Run NNoM on PC

Install Visual Studio, open the `contest.sln` to run it from VS. 

OR

Install `scons` and `GCC or msvc`, then run `scons` in this folder. 

> During model conversion, the dataset is also exported, the pc program will read them and do validation on nnom model. 

## Further Info

Due to the validation of the contest uses serial port to transmit data, the log of NNoM cannot be printed on the console. But we can run the same model from PC and check the memory and complexity of the model. 

For example:

```
validation size: 7036882
Model version: 0.4.3
NNoM version 0.4.3
To disable logs, please void the marco 'NNOM_LOG(...)' in 'nnom_port.h'.
Data format: Channel last (HWC)
Start compiling model...
Layer(#)         Activation    output shape    ops(MAC)   mem(in, out, buf)      mem blk lifetime
-------------------------------------------------------------------------------------------------
#1   Input      -          - (   1,1250,   1,)          (  1250,  1250,     0)    4 - - -  - - - - 
#2   Conv2D     - ReLU     - (   1, 625,   8,)      25k (  1250,  5000,     0)    3 1 - -  - - - - 
#3   MaxPool    -          - (   1, 156,   8,)          (  5000,  1248,     0)    2 1 1 1  - - - - 
#4   Conv2D     - ReLU     - (   1, 625,   4,)      22k (  1250,  2500,     0)    2 1 - 1  - - - - 
#5   MaxPool    -          - (   1, 156,   4,)          (  2500,   624,     0)    1 1 1 1  1 - - - 
#6   Conv2D     - ReLU     - (   1, 625,   4,)      32k (  1250,  2500,     0)    1 1 - 1  1 - - - 
#7   MaxPool    -          - (   1, 156,   4,)          (  2500,   624,     0)    1 1 1 1  1 - - - 
#8   Concat     -          - (   1, 156,  16,)          (  2496,  2496,     0)    1 - 1 1  1 - - - 
#9   Conv2D     - ReLU     - (   1, 156,   4,)      29k (  2496,   624,     0)    1 1 - -  - - - - 
#10  MaxPool    -          - (   1,  78,   4,)          (   624,   312,     0)    1 1 1 -  - - - - 
#11  RNN/GRU    -          - (  78,  12,     )     111k (   312,   936,   224)    1 1 1 -  - - - - 
#12  Flatten    -          - ( 936,          )          (   936,   936,     0)    - 1 - -  - - - - 
#13  Dense      - Sigmoid  - (   1,          )      936 (   936,     1,  1872)    1 1 1 -  - - - - 
#14  Output     -          - (   1,          )          (     1,     1,     0)    - - 1 -  - - - - 
-------------------------------------------------------------------------------------------------
Memory cost by each block:
 blk_0:2496  blk_1:5000  blk_2:624  blk_3:1248  blk_4:624  blk_5:0  blk_6:0  blk_7:0  
 Memory cost by network buffers: 9992 bytes
 Total memory occupied: 15024 bytes
Processing 2%
Processing 4%
Processing 6%
Processing 9%
Processing 11%
Processing 13%
Processing 15%
Processing 18%
Processing 20%
Processing 22%
Processing 25%
Processing 27%
Processing 29%
Processing 31%
Processing 34%
Processing 36%
Processing 38%
Processing 40%
Processing 43%
Processing 45%
Processing 47%
Processing 50%
Processing 52%
Processing 54%
Processing 56%
Processing 59%
Processing 61%
Processing 63%
Processing 65%
Processing 68%
Processing 70%
Processing 72%
Processing 75%
Processing 77%
Processing 79%
Processing 81%
Processing 84%
Processing 86%
Processing 88%
Processing 91%
Processing 93%
Processing 95%
Processing 97%
Processing 100%

Prediction summary:
Test frames: 0
Test running time: 0 sec
Model running time: 0 ms

Print running stat..
Layer(#)        -   Time(us)     ops(MACs)   ops/us 
--------------------------------------------------------
#1  Input      -         0                  
#2  Conv2D     -         0          25k     
#3  MaxPool    -         0                  
#4  Conv2D     -         0          22k     
#5  MaxPool    -         0                  
#6  Conv2D     -         0          32k     
#7  MaxPool    -         0                  
#8  Concat     -         0                  
#9  Conv2D     -         0          29k     
#10 MaxPool    -         0                  
#11 RNN/GRU    -         0         111k     
#12 Flatten    -         0                  
#13 Dense      -         0          936     
#14 Output     -         0                  

Summary:
Total ops (MAC): 222272(0.22M)
Prediction time :0us
Total memory:15024
Top 1 Accuracy on Tensorflow 96.57%
Top 1 Accuracy on NNoM  96.41%

Process finished with exit code 0

```

More information will be updated on https://github.com/majianjia/tinyml-contest and https://github.com/majianjia/nnom
