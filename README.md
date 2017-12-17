# Convolutional-Neural-Network
This is a 3 layers Convolutional Neural Network. It has two convolution layers and one fully connected layer. The detailled structure of the network as be shown as the following.

## Detailed Structure
Data -> Convolution -> ReLU -> Max-pooling -> Convolution -> ReLU -> Max-pooling -> Dropout -> Dense -> ReLU -> Dense -> Softmax.

## Hyper Parameters
In convolution layers, I use same padding, and stride = 1.
In max-pooling layers, I use filter size = 2*2, stride = 2.
