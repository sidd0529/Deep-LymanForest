# Deep-LymanForest
Measurement of signature of weak gravitational lensing of Lyman alpha forests using neural networks.

## What does this project do?
This repository contains python functions for two tasks:

* **Creation of mock catalogs:** Construct a catalog of mock catalogs. In each of these mock datasets, we have X and Y coordinates for source and lensed positions. For all these mock datasets, we make sure that the two dimensional lensed coordinates (X, Y ) are the same across the dataset. Corresponding to each entity in this dataset, the python functions also give us the lens potential field.

* **Neural network and prediction of lens potential:** We have python function which uses a multi layer perception with 4 hidden layers. In the feedforward hidden layer we have 400 nodes in each of the hidden layers. We use a learning rate of λ = 0.0005 , weight decay = 0.0005. We use Adam optimizer as our preferred optimizer and use mean square error (MSE) loss between input and reconstructed (output) images of lens potential field as our criterion. We use minibatches of 64 mocks in the optimization algorithm.

## How to get started with this project?
```
$ git clone https://github.com/sidd0529/Deep-LymanForest.git
```

## How to run this project?
In the first step, you will need to create mock catalogs using the function `generate_simulations.py`. Following is an example of how one can use `generate_simulations.py` to create 50,000 mock catalogs:

```
python generate_simulations.py -start 0 -end 50000
```

In an ideal case, one would want to use GPUs for such large scale generation of mocks in a time efficient manner. An example job submission script for the submission of `generate_simulations.py` (to clusters in Pittsburgh Supercomputing Center) can be found in `submit.job`.

Following the generation of mock catalogs, one will want to use `neuralnet_pytorch_boost10.py` which creates a neural network model based on multi-layer perceptrons. An example submission script for the submission of this function to GPUs (in Pittsburgh Supercomputing Center) can be found in neuralnet_pytorch_boost10_simg.sh .


## Where can you get help with this project?
I will be very happy to help in case you have any questions regarding this project. You can find me at siddharthsatpathy.ss@gmail.com .
