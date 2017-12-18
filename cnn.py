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

def get_3_dimension_indices(volume_shape, f, padding, stride):
    '''
    Get volume indices of depth, height, width dimensions.
    '''
    M, D, H, W = volume_shape
    out_height = int((H + 2 * padding - f) / stride) + 1
    out_width = int((W + 2 * padding - f) / stride) + 1
    depth_indices = np.repeat(np.arange(D), f * f).reshape(-1, 1)
    height_part1 = np.tile(np.repeat(np.arange(f), f), D).reshape(-1,1)
    height_part2 = stride * np.repeat(np.arange(out_height), out_width).reshape(1,-1)
    height_indices = height_part1+height_part2
    width_part1 = np.tile(np.arange(f), f * D).reshape(-1,1)
    width_part2 = stride * np.tile(np.arange(out_width), out_height).reshape(1,-1)
    width_indices = width_part1+width_part2
    return depth_indices, height_indices, width_indices

def transform_to_column(A_prev, f, padding, stride):
    '''
    Transform volume to a 2D matrix to be used for convolution
    '''
    M, D, H, W = A_prev.shape
    indices_d, indices_h, indices_w = get_3_dimension_indices(A_prev.shape,f,padding,stride)
    A_prev_pad = np.pad(A_prev, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    A_column = A_prev_pad[:, indices_d, indices_h, indices_w]
    A_column_reshaped = A_column.transpose(1, 2, 0).reshape(f*f*D, -1)
    return A_column_reshaped

def transform_to_volume(A_prev_col, volume_shape, f, padding, stride):
    '''
    Transform from 2D matrix to volume for convolution.
    '''
    M, D, H, W = volume_shape
    indices_d, indices_h, indices_w = get_3_dimension_indices(volume_shape,f,padding,stride)
    Zero = np.zeros((M, D, H+2*padding, W+2*padding), dtype=A_prev_col.dtype)
    A_prev_col_reshape = A_prev_col.reshape(D*f*f, -1, M).transpose(2,0,1)
    np.add.at(Zero, (slice(None), indices_d, indices_h, indices_w), A_prev_col_reshape)
    return Zero[:,:, padding:-padding, padding:-padding]

def get_2_dimension_indices(volume_shape, f, padding, stride):
    '''
    Get indices of height and width dimensions.
    '''
    M, D, H, W = volume_shape
    out_height = int((H + 2 * padding - f) / stride) + 1
    out_width = int((W + 2 * padding - f) / stride) + 1
    height_part1 = np.repeat(np.arange(f), f).reshape(-1, 1)
    height_part2 = stride * np.repeat(np.arange(out_height), out_width).reshape(1, -1)
    height_indices = height_part1 + height_part2
    width_part1 = np.tile(np.arange(f), f).reshape(-1, 1)
    width_part2 = stride * np.tile(np.arange(out_width), out_height).reshape(1, -1)
    width_indices = width_part1 + width_part2
    return height_indices,width_indices

def maxpool_transform_to_column(A_prev, f, padding, stride):
    '''
        Transform volume to a 2D matrix to be used for maxpooling.
    '''
    A_prev_pad = np.pad(A_prev, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    height_indices, width_indices = get_2_dimension_indices(A_prev.shape, f, padding, stride)
    A_prev_column = A_prev_pad[:,:, height_indices, width_indices]
    max_column = np.amax(A_prev_column, axis= 2)
    argmax_column = np.argmax(A_prev_column, axis= 2)
    return max_column, argmax_column

def maxpool_transform_to_volumn(dA, argmax_cols, A_prev_shape, f, padding, stride):
    '''
        Transform from 2D matrix to volume for maxpooling
    '''
    M, D, W, H = A_prev_shape
    Zeros = np.zeros((M, D, H+ 2*padding, W + 2*padding), dtype=dA.dtype)
    heigth_indices, width_indices = get_2_dimension_indices(A_prev_shape, f, padding, stride)
    map_size = heigth_indices.shape[1]
    height = np.tile(heigth_indices,(1, M*D))
    width = np.tile(width_indices,(1, M*D))
    max_H_indices = height[argmax_cols.reshape(-1), np.arange(height.shape[1])]
    max_W_indices = width[argmax_cols.reshape(-1), np.arange(width.shape[1])]
    max_M_indices = np.repeat(np.arange(M), map_size * D)
    max_D_indices = np.tile(np.repeat(np.arange(D), map_size), M)
    np.add.at(Zeros, (max_M_indices, max_D_indices, max_H_indices, max_W_indices), dA.reshape(-1))
    if padding == 0:
        return Zeros
    return Zeros[:, :, padding:-padding, padding:-padding]

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
    pad = int((f-1)/2) #Same padding number
    stride = 1
    n_C, n_C_prev, f, f = W.shape
    n_x, d_x, h_x, w_x = A_prev.shape
    n_H = int((h_x - f + 2 * pad) / stride) + 1
    n_W = int((w_x - f + 2 * pad) / stride) + 1

    A_col = transform_to_column(A_prev, f, pad, stride)
    W_col = W.reshape(n_C, -1)

    out = np.matmul(W_col, A_col) + b
    out = out.reshape(n_C, n_H, n_W, n_x)
    out_forward = out.transpose(3, 0, 1, 2)
    cache = (A_prev, A_col, W, b, pad, stride)
    return out_forward, cache

def conv_backward(dZ, cache):
    '''
    Compute Backward Propagation of gradients after convolution layer.
    :param dZ: Gradients from previous layer in back propagation
    :param cache: cache stored from forward pass
    :return: Conv_backward, dW, db
    '''
    A_prev, A_col, W, b, pad, stride = cache
    n_C, n_C_prev, f, f = W.shape
    dZ_reshaped = dZ.transpose(1, 2, 3, 0).reshape(n_C, -1)
    dW = np.dot(dZ_reshaped, A_col.T).reshape(W.shape)
    W_reshape = W.reshape(n_C, -1)
    out = np.matmul(W_reshape.T, dZ_reshaped)
    out_backward = transform_to_volume(out, A_prev.shape, f, pad, stride)
    db = np.sum(dZ, axis=(0, 2, 3)).reshape(n_C, -1)
    return out_backward, dW, db

def max_pool_forward(A_prev, hparameters):
    '''
    Compute max-pooling forward pass. Also convert input volumes to 2D matrices to speed up the computing progress.
    :param A_prev: Input from previous layer
    :param hparameters: hyper-parameters
    :return: Max_pool_output, cache
    '''
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int((n_H_prev - f) / stride)+1
    n_W = int((n_W_prev - f) / stride)+1
    n_C = n_C_prev

    max_cols, argmax_cols = maxpool_transform_to_column(A_prev, f, 0, stride)
    A = max_cols.reshape(m, n_C, n_H, n_W)
    cache = (A_prev, argmax_cols, hparameters)
    return A, cache

def max_pool_backward(dA, cache):
    '''
    Compute max-pooling backward pass.
    :param dA: Gradients from previous layer in back propagation.
    :param cache: cache stored from forward pass.
    :return: Max_pool_backward
    '''
    A_prev, argmax_cols, hparameters = cache
    stride = hparameters['stride']
    f = hparameters['f']

    dA_prev = maxpool_transform_to_volumn(dA, argmax_cols, A_prev.shape, f, 0, stride)
    return dA_prev

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
    (W1, b1, W2, b2, W3, b3, W4, b4) = model(X_train,Y_train, epoch_num= 10)
    params = (W1, b1, W2, b2, W3, b3, W4, b4)

    print("Test set:")
    predict(params, X_test, Y_test)

if __name__ == '__main__':
    main()