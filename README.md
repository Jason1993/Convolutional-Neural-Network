# Convolutional-Neural-Network
This is a 3 layers Convolutional Neural Network. It has two convolution layers and one fully connected layer. Since it is a simple demon, I didn't write layers as classes, instead I wrote layers as functions for simplicity. The detailled structure of the network as be shown as the following.

## Detailed Structure
Data -> Convolution -> ReLU -> Max-pooling -> Convolution -> ReLU -> Max-pooling -> Dropout -> Dense -> ReLU -> Dense -> Softmax.

## Some Hyper Parameters
In convolution layers, I use same padding, and stride = 1.
In max-pooling layers, I use filter size = 2*2, stride = 2.

## Results
Running an epoch over all 60000 training items is quite time consuming. I just run 10 epoches on the training set, and then make predictions on test set, and the accuracy is 97.98%. If increase the epoch number, the accuracy will improve as reaching over 99%.

## Improvement
In this program, after every epoch, the learning rate will derease, as

'''Python
learning_rate *= 0.9
'''

This will help improve the loss function as setting down the learning rate with epoches will make the loss function more stable.
