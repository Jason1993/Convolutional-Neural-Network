import struct
import numpy as np

def readData():
    '''
    The files should be uncompressed.
    The file path should be modified if not put in the same directory as cnn.py
    return:
    tr_label: numpy array contains training labels, and shape is (60000,).
    tr_img: numpy array contains training images, and shape is (60000, 28, 28).
    te_label: numpy array contains test labels, and shape is (10000,)
    te_img: numpy array contains test images, and shape is (10000, 28, 28)
    '''
    train_label = 'train-labels-idx1-ubyte'
    train_img = 'train-images-idx3-ubyte'
    test_img = 't10k-images-idx3-ubyte'
    test_label = 't10k-labels-idx1-ubyte'

    with open(train_label, 'rb') as trainLabelFile:
        struct.unpack(">II", trainLabelFile.read(8))
        tr_label = np.fromfile(trainLabelFile, dtype=np.int8)

    with open(train_img, 'rb') as trainImgFile:
        magic, num, rows, cols = struct.unpack(">IIII", trainImgFile.read(16))
        tr_img = np.fromfile(trainImgFile, dtype=np.uint8).reshape(len(tr_label), rows, cols)
        tr_img = tr_img[ :,np.newaxis, :, :]

    with open(test_label, 'rb') as testLabelFile:
        struct.unpack(">II", testLabelFile.read(8))
        te_label = np.fromfile(testLabelFile, dtype=np.int8)

    with open(test_img, 'rb') as testImgFile:
        magic, num, rows, cols = struct.unpack(">IIII", testImgFile.read(16))
        te_img = np.fromfile(testImgFile, dtype=np.uint8).reshape(len(te_label), rows, cols)
        te_img = te_img[ :, np.newaxis, :, :]

    return tr_label, tr_img, te_label, te_img

def conv_forward(A_prev, W, b):
    '''
    Computing convolution forward step. In this program, convolution layers are always using Stride = 1, same padding.
    Usually if we use traditional for-loop to compute convolution explicitly, it will be very time-consuming as it will require 4 nested for-loops.
    Insteading, we convert the input matrix and filter matrix into 2D-matrices, and use matrix dot to speed up convolution
    Each depth cut of input volume is converted into column.
    :param A_prev: Input from previous layer.
    :param W: Filters matrix, which dimension are (Number of filters, Depth of preivous layer, filter width, filter width)
    :param b: Bias matrix, which dimensions are (Number of filters, 1)
    :return Conv_output, cache
    '''
    n_C, n_C_prev, f, f = W.shape
    m, prev_d, prev_h, prev_w = A_prev.shape
    pad = int((f-1)/2) #Same padding number
    stride = 1
    n_H = int((prev_h - f + 2 * pad) / stride + 1) #Output volume height
    n_W = int((prev_w - f + 2 * pad) / stride + 1) #Output volume width
    A_prev_pad = np.pad(A_prev, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    #Compute column indices
    Col_indices_depth = np.repeat(np.arange(n_C_prev), f * f).reshape(-1, 1)
    index_height_part = stride * np.repeat(np.arange(n_H), n_W)
    Col_indices_height = np.tile(np.arange(f), f * n_C_prev).reshape(-1,1) + index_height_part.reshape(1,-1)
    index_width_part = stride * np.tile(np.arange(n_W), n_H)
    Col_indices_width = np.tile(np.arange(f), f * n_C_prev).reshape(-1,1) + index_width_part.reshape(1,-1)

    #Use column indices to convert the input volume into a 2D matrix for computing convolution
    A_prev_cols = A_prev_pad[:, Col_indices_depth, Col_indices_height, Col_indices_width]
    A_prev_cols = A_prev_cols.transpose(1, 2, 0).reshape(f * f * prev_d, -1)
    W_col = W.reshape(n_C,-1)
    out = np.matmul(W_col, A_prev_cols) + b
    out = out.reshape(n_C, n_H, n_W, m)
    Conv_output = out.transpose(3, 0, 1, 2)
    cache = (A_prev, A_prev_cols, W, b, pad, stride, Col_indices_depth, Col_indices_height, Col_indices_width)
    return Conv_output, cache

def conv_backward(dZ, cache):
    '''
    Compute Backward Propagation of gradients after convolution layer.
    :param dZ: Gradients from previous layer in back propagation
    :param cache: cache stored from forward pass
    :return: Conv_backward, dW, db
    '''
    A_prev, A_prev_cols, W, b, pad, stride, Col_indices_depth, Col_indices_height, Col_indices_width = cache
    m, prev_d, prev_h, prev_w = A_prev.shape
    n_C, n_C_prev, f, f = W.shape
    dZ_reshaped = dZ.transpose(1, 2, 3, 0).reshape(n_C, -1)
    dW = np.dot(dZ_reshaped, A_prev_cols.T).reshape(W.shape)
    db = np.sum(dZ, axis=(0, 2, 3)).reshape(n_C, -1)
    W_reshape = W.reshape(n_C, -1)
    out = np.matmul(W_reshape.T, dZ_reshaped)
    H_padded, W_padded = prev_h + 2 * pad, prev_w + 2 * pad
    A_prev_pad = np.zeros((m, prev_d, H_padded, W_padded), dtype=A_prev_cols.dtype)
    cols_reshaped = out.reshape(prev_d * f * f, -1, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(A_prev_pad, (slice(None), Col_indices_depth, Col_indices_height, Col_indices_width), cols_reshaped)
    Conv_backward = A_prev_pad[:, :, pad:-pad, pad:-pad]
    return Conv_backward, dW, db

def max_pool_forward(A_prev, hparameters):
    '''
    Compute max-pooling forward pass. Also convert input volumes to 2D matrices to speed up the computing progress.
    :param A_prev: Input from previous layer
    :param hparameters: hyper-parameters
    :return: Max_pool_output, cache
    '''
    f = hparameters["f"]
    stride = hparameters["stride"]
    m, prev_d, prev_h, prev_w = A_prev.shape
    n_H = int(1 + (prev_h - f) / stride)
    n_W = int(1 + (prev_w - f) / stride)

    col_indices_height = np.repeat(np.arange(f), f).reshape(-1, 1) + stride * np.repeat(np.arange(n_H), n_W).reshape(1, -1)
    col_indices_width = np.tile(np.arange(f), f).reshape(-1, 1) + stride * np.tile(np.arange(n_W), n_H).reshape(1, -1)
    A_prev_cols = A_prev[:, :, col_indices_height, col_indices_width]
    max_cols = np.amax(A_prev_cols, axis=2)
    argmax_cols = np.argmax(A_prev_cols, axis=2)
    Max_pool_output = max_cols.reshape(m, prev_d, n_H, n_W)
    cache = (A_prev, argmax_cols, hparameters, col_indices_height, col_indices_width)
    return Max_pool_output, cache

def max_pool_backward(dA, cache):
    '''
    Compute max-pooling backward pass.
    :param dA: Gradients from previous layer in back propagation.
    :param cache: cache stored from forward pass.
    :return: Max_pool_backward
    '''
    A_prev, argmax_cols, hparameters, col_indices_height, col_indices_width = cache
    temp = np.zeros((A_prev.shape))
    m, prev_d, prev_h, prev_w = A_prev.shape
    map_size = col_indices_height.shape[1]
    height  = np.tile(col_indices_height, (1, m * prev_d))
    width = np.tile(col_indices_width, (1, m * prev_d))
    max_height = height[argmax_cols.reshape(-1), np.arange(height.shape[1])]
    max_width = width[argmax_cols.reshape(-1), np.arange(width.shape[1])]
    max_m = np.repeat(np.arange(m), map_size * prev_d)
    max_depth = np.tile(np.repeat(np.arange(prev_d), map_size), m)
    np.add.at(temp, (max_m, max_depth, max_height, max_width), dA.reshape(-1))
    Max_pool_backward = temp
    return Max_pool_backward

def random_mini_batches(X, Y, mini_batch_size = 5):
    '''
    Shuffle the whole batch and devide into several mini-batches
    :param X:
    :param Y:
    :param mini_batch_size:
    :return:
    '''
    m = X.shape[0]  # number of training examples
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation,:]
    num_complete_minibatches = int(np.floor(m / mini_batch_size))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: (k + 1) * mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X, Y, learning_rate = 0.001, epoch_num = 20, mini_batch_size = 100):
    #define max_pooling hyper-parameters
    max_pool_hp = {}
    max_pool_hp['f'] = 2
    max_pool_hp['stride'] = 2

    # convert Y to One_hot_Y
    Y = Y.reshape(-1)
    One_hot_Y = np.eye(10)[Y]

    # initialize parameters
    W1 = np.random.normal(0, 0.1, (25, 1, 5, 5))
    b1 = np.random.normal(0, 0.1, (25, 1))
    W2 = np.random.normal(0, 0.1, (25, 25, 3, 3))
    b2 = np.random.normal(0, 0.1, (25, 1))
    W3 = np.random.normal(0, 0.01, (1225, 512))
    b3 = np.random.normal(0, 0.01, (1, 512))
    W4 = np.random.normal(0, 0.01, (512, 10))
    b4 = np.random.normal(0, 0.01, (1, 10))

    for epoch in range(epoch_num):
        minibatches = random_mini_batches(X, One_hot_Y, mini_batch_size)
        count = 1
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            # forward propagation
            Z1,cacheConv1 = conv_forward(minibatch_X, W1, b1)   #First convolution layer
            A1 = Z1 * (Z1 > 0)                                  #ReLu activation
            P1, cachePool1 = max_pool_forward(A1, max_pool_hp)  #First max-pooling layer
            Z2, cacheConv2 = conv_forward(P1, W2, b2)           #Second convolution layer
            A2 = Z2 * (Z2 > 0)                                  #ReLu activation
            P2, cachePool2 = max_pool_forward(A2, max_pool_hp)  #Second max-pooling layer
            P2_flat = P2.reshape(P2.shape[0], -1)               #Flatten layer
            mask = (np.random.uniform(0.0, 1.0, P2_flat.shape) >= 0.5).astype(float) * (1.0 / (1.0 - 0.5))
            P2_dropout = np.multiply(P2_flat, mask)             #Drop-out layer
            Z3 = np.dot(P2_dropout, W3) + b3                    #Dense layer
            A3 = Z3 * (Z3 > 0)                                  #ReLu activation
            Z4 = np.dot(A3, W4) + b4                            #Dense layer
            calib = Z4 - np.amax(Z4, axis=1, keepdims=True)     #Softmax layer
            sum_exp = np.sum(np.exp(calib), axis=1, keepdims=True)
            A4 = np.exp(calib) / sum_exp

            #Compute cross-entropy loss
            loss = - np.sum(np.multiply(minibatch_Y, calib - np.log(sum_exp))) / minibatch_Y.shape[0]

            # back propagation
            dZ4 = (A4 - minibatch_Y) / minibatch_Y.shape[0]     #Gradients output from softmax
            dW4 = np.dot(A3.T, dZ4)
            db4 = np.sum(dZ4, axis=0) / len(dZ4)
            dA3 = np.dot(dZ4, W4.T)                             #Gradients output from dense layer
            dZ3 = np.multiply(dA3, (Z3 > 0))                    #Gradients output from ReLu
            dW3 = np.dot(P2_flat.T, dZ3)
            db3 = np.sum(dZ3, axis=0) / len(dZ3)
            dP2_dropout = np.array(np.dot(dZ3, W3.T))           #Gradients output from dense layer
            dP2_flat = np.multiply(dP2_dropout, mask)           #Gradients output from dropout layer
            dP2 = dP2_flat.reshape(P2.shape)                    #Gradients output from flatten layer
            dA2 = max_pool_backward(dP2, cachePool2)            #Gradients output from second max-pooling layer
            dZ2 = np.multiply(dA2, (Z2 > 0))                    #Gradients output from ReLu
            dP1, dW2, db2 = conv_backward(dZ2, cacheConv2)      #Gradients output from second convolution layer
            dA1 = max_pool_backward(dP1, cachePool1)            #Gradients output from first max-pooling layer
            dZ1 = np.multiply(dA1, (Z1 > 0))                    #Gradients output from ReLu
            dA0, dW1, db1 = conv_backward(dZ1, cacheConv1)      #Gradients output from first convolution layer

            # Update parameters using gradient descent
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3
            W4 -= learning_rate * dW4
            b4 -= learning_rate * db4
            print("Finished running "+str(count)+" minibatches, and the loss is "+ str(loss))
            count += 1
        print("After " + str(epoch) + " epoches, the loss is " + str(loss))
        parameters = (W1, b1, W2, b2, W3, b3, W4, b4)
        minibatch_Y_normal = [np.where(r == 1)[0][0] for r in minibatch_Y]
        predict(parameters, minibatch_X, minibatch_Y_normal)
        learning_rate *= 0.9
    return parameters

def predict(parameters, X, Y):
    hp_pool = {}
    hp_pool['stride'] = 2
    hp_pool['f'] = 2
    (W1,b1,W2,b2,W3,b3,W4,b4) = parameters
    Z1, cacheConv1 = conv_forward(X,W1,b1)
    A1 = Z1 * (Z1 > 0)
    P1, cachePool1 = max_pool_forward(A1, hp_pool)
    Z2, cacheConv2 = conv_forward(P1, W2, b2)
    A2 = Z2 * (Z2 > 0)
    P2, cachePool2 = max_pool_forward(A2, hp_pool)
    P2_flat = P2.reshape(P2.shape[0], -1)
    Z3 = np.dot(P2_flat, W3) + b3
    A3 = Z3 * (Z3 > 0)
    Z4 = np.dot(A3, W4) + b4
    calib = Z4 - np.amax(Z4, axis=1, keepdims=True)
    sum_exp = np.sum(np.exp(calib), axis=1, keepdims=True)
    A4 = np.exp(calib) / sum_exp
    Y_predict = np.argmax(A4, axis=1)
    counter = np.sum(Y_predict == Y)
    ac = counter/len(Y)
    print("Accuracy is "+str(ac))

def main():
    np.random.seed(2)
    Y_train, X_train, Y_test, X_test = readData()
    (W1, b1, W2, b2, W3, b3, W4, b4) = model(X_train,Y_train, learning_rate= 0.001, epoch_num= 20)
    params = (W1, b1, W2, b2, W3, b3, W4, b4)
    print("Traning set:")
    predict(params, X_train, Y_train)
    print("Test set:")
    predict(params, X_test, Y_test)

if __name__ == '__main__':
    main()