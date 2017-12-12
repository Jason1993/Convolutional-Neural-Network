import struct
import numpy as np
np.random.seed(1)
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
        tr_img = tr_img[ :, :, :, np.newaxis]

    with open(test_label, 'rb') as testLabelFile:
        struct.unpack(">II", testLabelFile.read(8))
        te_label = np.fromfile(testLabelFile, dtype=np.int8)

    with open(test_img, 'rb') as testImgFile:
        magic, num, rows, cols = struct.unpack(">IIII", testImgFile.read(16))
        te_img = np.fromfile(testImgFile, dtype=np.uint8).reshape(len(te_label), rows, cols)
        te_img = te_img[ :, :, :, np.newaxis]

    return tr_label, tr_img, te_label, te_img

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
    cache = (A_prev, W, b, hparameters)
    return Z,cache

def max_pool_forward(A_prev, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_prev_slice = A_prev[i, vert_start: vert_end, horiz_start: horiz_end, c]
                    A[i, h, w, c] = np.max(a_prev_slice)
    cache = (A_prev, hparameters)
    return A,cache

def conv_backward(dZ, cache):
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[:,:, :]
    return dA_prev, dW, db

def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask

def distribute_value(dz, shape):
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix (≈1 line)
    average = dz / n_H / n_W

    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones((n_H, n_W)) * average

    return a

def max_pool_backward(dA, cache):
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):  # loop over the training examples

        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i, :, :, :]

        for h in range(n_H):  # loop on the vertical axis
            for w in range(n_W):  # loop on the horizontal axis
                for c in range(n_C):  # loop over the channels (depth)

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                    a_prev_slice = a_prev[vert_start: vert_end, horiz_start: horiz_end, c]
                    # Create the mask from a_prev_slice (≈1 line)
                    mask = create_mask_from_window(a_prev_slice)
                    # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                    dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]

    return dA_prev

def random_mini_batches(X, Y, mini_batch_size = 100):
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

def compute_cost(A4, Y):
    '''
    :param A4: softmax output, and shape is [m,10]
    :param Y: the ground truth label, and shape is [m,10]
    :return: cost: the whole cross entropy cost of the set
    '''
    loss = np.sum(np.multiply( np.log(A4), Y),axis= 1)*(-1)
    cost = np.sum(loss)/A4.shape[0]
    return cost


def model(X, Y, learning_rate = 0.009, epoch_num = 100, mini_batch_size = 100):
    #initialize hyper parameters
    hp_conv = {}
    hp_conv['stride'] = 1
    hp_conv['pad'] = 0
    hp_pool = {}
    hp_pool['stride'] = 2
    hp_pool['f'] = 2

    #convert Y to One_hot_Y
    Y = Y.reshape(-1)
    One_hot_Y = np.eye(10)[Y]

    #initialize parameters
    W1 = np.random.rand(5, 5, 1, 32) * 0.01
    b1 = np.zeros((1, 1, 1, 32))
    W2 = np.random.rand(5, 5, 32, 64) * 0.01
    b2 = np.zeros((1, 1, 1, 64))
    W3 = np.random.rand(1024,512)
    b3 = np.zeros(512)
    W4 = np.random.rand(512,10)
    b4 = np.zeros(10)
    for epoch in range(epoch_num):
        minibatches = random_mini_batches(X,One_hot_Y)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            #forward propagation
            #first convolution layer
            Z1,cacheConv1 = conv_forward(minibatch_X, W1, b1, hp_conv)
            #relu activation
            A1 = Z1* (Z1 > 0)
            #first max-pooling layer
            P1, cachePool1 = max_pool_forward(A1, hp_pool)
            #second convolution layer
            Z2,cacheConv2 = conv_forward(P1, W2, b2, hp_conv)
            #relu activation
            A2 = Z2 * (Z2 > 0)
            #second max-pool layer
            P2, cachePool2 = max_pool_forward(A2, hp_pool)
            #dense layer
            P2_flat = P2.reshape((-1,4*4*64))
            Z3 = np.dot(P2_flat,W3)+b3
            #relu
            A3 = Z3 * (Z3 > 0)
            #softmax
            Z4 = np.dot(A3,W4)+b4
            exps = np.exp(Z4 - np.max(Z4))
            A4 = exps/np.sum(exps)

            cost = compute_cost(Z4,minibatch_Y)

            #back propagation
            #compute gradients back chain rule
            dZ4 = A4 - minibatch_Y
            dW4 = np.dot(A3.T,dZ4)
            db4 = np.sum(dZ4, axis=0)/len(dZ4)
            dA3 = np.dot(dZ4,W4.T)
            dZ3 = dA3 * (Z3 > 0)
            dW3 = np.dot(P2_flat.T,dZ3)
            db3 = np.sum(dZ3, axis=0)/len(dZ3)
            dP2_flat = np.dot(dZ3,W3.T)
            dP2 = dP2_flat.reshape(P2.shape)
            dA2 = max_pool_backward(dP2,cachePool2)
            dZ2 = dA2 * (Z2 > 0)
            dP1, dW2, db2 = conv_backward(dZ2,cacheConv2)
            dA1 = max_pool_backward(dP1,cachePool1)
            dZ1 = dA1 * (Z1 > 0)
            dA0, dW1, db1 = conv_backward(dZ1,cacheConv1)

            #update parameters
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3
            W4 -= learning_rate * dW4
            b4 -= learning_rate * db4
        print("After "+str(epoch)+" epoches, the cost is "+ str(cost))
    parameters = (W1,b1,W2,b2,W3,b3,W4,b4)
    return parameters

def predict(parameters, X, Y):
    hp_conv = {}
    hp_conv['stride'] = 1
    hp_conv['pad'] = 0
    hp_pool = {}
    hp_pool['stride'] = 2
    hp_pool['f'] = 2
    (W1,b1,W2,b2,W3,b3,W4,b4) = parameters
    Z1, cacheConv1 = conv_forward(X,W1,b1,hp_conv)
    A1 = Z1 * (Z1 > 0)
    P1, cachePool1 = max_pool_forward(A1, hp_pool)
    Z2, cacheConv2 = conv_forward(P1, W2, b2, hp_conv)
    A2 = Z2 * (Z2 > 0)
    P2, cachePool2 = max_pool_forward(A2, hp_pool)
    P2_flat = P2.reshape((-1, 7 * 7 * 64))
    Z3 = np.dot(P2_flat, W3) + b3
    A3 = Z3 * (Z3 > 0)
    Z4 = np.dot(A3, W4) + b4
    exps = np.exp(Z4 - np.max(Z4))
    exps = np.exp(Z4 - np.max(Z4))
    A4 = exps / np.sum(exps)
    Y_predict = np.argmax(A4, axis=1)
    counter = np.sum(Y_predict == Y)
    ac = counter/len(Y)
    print("Accuracy is "+str(ac))

def main():
    Y_train, X_train, Y_test, X_test = readData()
    (W1, b1, W2, b2, W3, b3, W4, b4) = model(X_train,Y_train)
    params = (W1, b1, W2, b2, W3, b3, W4, b4)
    print("Traning set:")
    predict(params, X_train, Y_train)
    print("Test set:")
    predict(params, X_test, Y_test)

if __name__ == '__main__':
    main()