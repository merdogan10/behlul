# BEHLUL

My awesome deep learning library from scratch with C++. BEHLUL is acronym for `Behlul is an Efficient High Level Useful Library`. Name is inspired from the famous fictitious Turkish novel character Behlul Ziyagil.

# Layers

All layers are implemented in different classes. Sizes of outputs and inputs must match. Here are the classes:

- Convolution Layer
- ReLU Layer
- Max Pool Layer
- Dense Layer
- Softmax Layer
- Cross Entropy Layer

Xor network is deprecated and dense layer is implemented all over again.

# Documentation

Documentation is generated by doxygen and inside `docs` folder. For pdf version, open `refman.pdf` file. For html version, open `html/index.html`.

# State Farm Distracted Driver Detection (Final Project)

## How to run?

To run State Farm Distracted Driver Detection from python notebook, run those commands in the same order.

```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install jupyter Pillow numpy pytest matplotlib
$ jupyter notebook
```
Open the link given by jupyter on the browser (something like `http://127.0.0.1:8888/tree`). Open `state_farm_cnn.ipynb`. Then run the cells of the notebook in an order according to your purpose (run all, run pretrained etc.)

To run unit tests, run those commands in the same order.
```
$ cd build
$ cmake ..
$ make
$ ./unit_test_main
```

## How to use Behlul in python?

When Behlul is compiled in previous step, it generates a file named `my_project.cpython-36m-x86_64-linux-gnu.so` in build folder. I created my notebook in root folder, so i import behlul as `import build.my_project as behlul` and here is an example conv layer created from Behlul: `conv = behlul.Conv_Layer (28, 28, 1, 5, 1, 6)`.

## Data

I get the Driver Images data from [kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) as zip file. I extracted `state-farm-distracted-driver-detection/imgs/train` to `data/train` folder. I only used train data as my whole  dataset for simplicity. I splitted the given train data to ~80% of it as train data, ~10% of it as validation data and remaining ~10% of it as test data. Pretrained weights are also kept under `data` folder as `data/state_conv1.out`, `data/state_conv2.out`, `data/state_dense.out`.

## Data Preprocess

All images are 480x640. I cropped 80px from left and rights to make images square for my model. As main attraction points are almost always in the middle of the images, I didn't lose much useful information. Initially, I was going to resize images to 224x224 to run AlexNet. However, when I tried my old model from MNIST with resized 28x28 new images, I got some successful results (50% acc. for 1 epoch with 10 classes). Then, I didn't go further and stayed with my model.

## Model

I used the same model as I used in MNIST.

* Conv_Layer 1: input: 28x28x1 filter: 5x5x1 num_filters: 6 stride: 1 output: 24x24x6
* ReLU 1: output: 24x24x6
* Max_Pool 1: input: 24x24x6 filter: 2x2x6 stride: 2 output: 12x12x6
* Conv_Layer 2: input: 12x12x6 filter: 5x5x6 num_filters: 16 stride: 1 output: 8x8x16
* ReLU 2: output: 8x8x16
* Max_Pool 2: input: 8x8x16 filter: 2x2x16 stride: 2 output: 4x4x16
* Dense_Layer: input: 4x4x16 output: 1x10
* Softmax
* Cross_Entropy

## Result

I ran the data 5 epoch and it took 30 mins to get output.
```
  Training accuracy: 0.78
  Validation accuracy: 0.75
  Test accuracy: 0.72
```

# MNIST classifier (Milestone)

## How to run?

To run MNIST classifier from c++ main, run those commands in the same order.
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./run_main
```
If you don't want to train data all over again and want to use pretrained weights, press `y` and then `enter` when the program asks after the start. Otherwise, press another character `not y` and then `enter` to standard long hours training.

Normal training takes ~4 mins.
By using pretrained weights, running train set and validation set takes ~40 secs.
By using pretrained weights, running only validation set takes ~7 secs.

So I decided to use train set and validation set with pretrained weights for demo purpose.
(I printed the index on every 1000 example to keep track.)

## Data

I get the MNIST data from [kaggle](https://www.kaggle.com/c/digit-recognizer/data) as csv files. I read the data from `data/train.csv` and `data/test.csv`. I splitted the given train data to 90% of it as train data and remaining 10% of it as validation data. I get the output of the test data and send it to kaggle competition. Pretrained weights are also kept under `data` folder as `data/conv1.out`, `data/conv2.out`, `data/dense.out`.

## Model
* Conv_Layer 1: input: 28x28x1 filter: 5x5x1 num_filters: 6 stride: 1 output: 24x24x6
* ReLU 1: output: 24x24x6
* Max_Pool 1: input: 24x24x6 filter: 2x2x6 stride: 2 output: 12x12x6
* Conv_Layer 2: input: 12x12x6 filter: 5x5x6 num_filters: 16 stride: 1 output: 8x8x16
* ReLU 2: output: 8x8x16
* Max_Pool 2: input: 8x8x16 filter: 2x2x16 stride: 2 output: 4x4x16
* Dense_Layer: input: 4x4x16 output: 1x10
* Softmax
* Cross_Entropy

## Result

I ran the data 1 epoch and it took 4 mins to get output.
```
  Training accuracy: 0.970026
  Validation accuracy: 0.96881
  Test accuracy: 0.96700
```
![result](/kaggle.png)
