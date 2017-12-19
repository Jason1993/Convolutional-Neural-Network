# Convolutional-Neural-Network
This is a 3 layers Convolutional Neural Network. It has two convolution layers and one fully connected layer. Since it is a simple demo, layers was not written as classes, instead I wrote layers as functions for simplicity. The detailled structure of the network as be shown as the following.

## Detailed Structure
Data -> Convolution -> ReLU -> Max-pooling -> Convolution -> ReLU -> Max-pooling -> Dropout -> Dense -> ReLU -> Dense -> Softmax -> Output.

## Some Hyper Parameters
In convolution layers, Padding is 'same' padding, and stride = 1.


In max-pooling layers, filter size is (2*2), and stride = 2.


This program uses mini-batch stochastic gradient descent, and batch_size = 100, so there are 600 mini-batches in total. Running an epoch over the all 600 mini-batches will take about 15 minutes on my laptop.

## Results
Running an epoch over all 60000 training items is quite time consuming. I just run 10 epoches on the training set, and then make predictions on test set, and the accuracy is 97.98%. If increase the epoch number, the accuracy will also increase.

## Improvement
In this program, after every epoch, the learning rate will derease, as

```Python
learning_rate *= 0.9
```

This will help improve the loss function as setting down the learning rate with epoches will make the loss function more stable.
