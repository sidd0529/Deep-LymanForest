# Deep-LymanForest
Measurement of signature of weak gravitational lensing of Lyman alpha forests using neural networks.

## What does this project do?
This repository contains python functions for two tasks:

* **Creation of mock catalogs:** Construct a catalog of mock catalogs. In each of these mock datasets, we have X and Y coordinates for source and lensed positions. For all these mock datasets, we make sure that the two dimensional lensed coordinates (X, Y ) are the same across the dataset. Corresponding to each entity in this dataset, the python functions also give us the lens potential field.

* **Neural network and prediction of lens potential:** We have python function which uses a multi layer perception with 4 hidden layers. In the feedforward hidden layer we have 400 nodes in each of the hidden layers. We use a learning rate of \lambda = 0.0005 , weight decay = 0.0005. We use Adam optimizer as our preferred optimizer and use mean square error (MSE) loss between input and reconstructed (output) images of lens potential field as our criterion. We use minibatches of 64 mocks in the optimization algorithm.

