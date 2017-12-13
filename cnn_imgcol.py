import struct
import numpy as np
np.random.seed(1)

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def get_maxpool_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return i, j


def maxpool_im2col_indices(x, field_height, field_width, padding=0, stride=1):

    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    i, j = get_maxpool_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, :, i, j]

    max_cols = np.amax(cols, axis = 2)
    argmax_cols = np.argmax(cols, axis = 2)
    return max_cols, argmax_cols


def maxpool_col2im_indices(grad, argmax_cols, x_shape, field_height=3, field_width=3, padding=0, stride=1):

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=grad.dtype)

    i, j = get_maxpool_im2col_indices(x_shape, field_height, field_width, padding, stride)
    map_size = i.shape[1]
    i = np.tile(i, (1, N * C))
    j = np.tile(j, (1, N * C))
    max_i = i[argmax_cols.reshape(-1), np.arange(i.shape[1])]
    max_j = j[argmax_cols.reshape(-1), np.arange(j.shape[1])]
    max_n = np.repeat(np.arange(N), map_size * C)
    max_c = np.tile(np.repeat(np.arange(C), map_size), N)
    np.add.at(x_padded, (max_n, max_c, max_i, max_j), grad.reshape(-1))
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
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

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_C, n_C_prev, f, f = W.shape
    n_x, d_x, h_x, w_x = A_prev.shape
    n_H = int((h_x - f + 2 * pad) / stride + 1)
    n_W = int((w_x - f + 2 * pad) / stride + 1)

    A_col = im2col_indices(A_prev, f, f, pad, stride)
    W_col = W.reshape(n_C, -1)

    out = np.matmul(W_col, A_col) + b
    out = out.reshape(n_C, n_H, n_W, n_x)
    out_forward = out.transpose(3, 0, 1, 2)
    cache = (A_prev, A_col, W, b, hparameters)
    return out_forward, cache

def max_pool_forward(A_prev, hparameters):
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    max_cols, argmax_cols = maxpool_im2col_indices(A_prev, f, f, 0, stride)
    A = max_cols.reshape(m, n_C, n_H, n_W)
    cache = (A_prev, argmax_cols, hparameters)
    return A,cache

def conv_backward(dZ, cache):
    (A_prev, A_col, W, b, hparameters) = cache

    stride = hparameters['stride']
    pad = hparameters['pad']
    n_C, n_C_prev, f, f = W.shape

    dZ_reshape = dZ.transpose(1, 2, 3, 0).reshape(n_C, -1)
    W_reshape = W.reshape(n_C, -1)
    out = np.matmul(W_reshape.T, dZ_reshape)
    out_backward = col2im_indices(out, A_prev.shape, f, f, pad, stride)
    dW = np.matmul(dZ_reshape , A_col.T).reshape(W.shape)
    db = np.sum(dZ, axis=(0, 2, 3)).reshape(n_C, -1)
    return out_backward, dW, db

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
    (A_prev, argmax_cols, hparameters) = cache

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']

    dA_prev = maxpool_col2im_indices(dA, argmax_cols, A_prev.shape, f, f, 0, stride)
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
    #loss = np.sum(np.multiply( np.log(A4), Y),axis= 1)*(-1)
    #cost = np.sum(loss)/A4.shape[0]
    calib = A4 - np.amax(A4, axis=1, keepdims=True)
    sum_exp = np.sum(np.exp(calib), axis=1, keepdims=True)
    cost = - np.sum(np.multiply(Y,calib - np.log(sum_exp)))/ A4.shape[0]
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
    W1 = np.random.normal(0, 0.1,(32, 1, 5, 5))
    b1 = np.random.normal(0, 0.1, (32, 1))
    W2 = np.random.normal(0, 0.1,(64, 32, 5, 5))
    b2 = np.random.normal(0, 0.1, (64, 1))
    W3 = np.random.rand(1024,512) * 0.001
    b3 = np.zeros((1,512))
    W4 = np.random.rand(512,10) * 0.001
    b4 = np.zeros((1,10))
    for epoch in range(epoch_num):
        minibatches = random_mini_batches(X,One_hot_Y)
        count= 0
        for minibatch in minibatches:
            count += 1
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
            P2_flat = P2.reshape(P2.shape[0], -1)
            #P2_ori = P2_flat.reshape(P2.shape)
            Z3 = np.dot(P2_flat,W3)+b3
            #relu
            A3 = Z3 * (Z3 > 0)
            #softmax
            Z4 = np.dot(A3,W4)+b4
            temp = np.matrix(np.max(Z4, axis=1)).T
            res = temp.repeat(10, axis =1)
            exps = np.exp(Z4 - res)
            A4 = exps/np.sum(exps, axis =1)

            cost = compute_cost(Z4,minibatch_Y)

            #back propagation
            #compute gradients back chain rule
            dZ4 = A4 - minibatch_Y
            dW4 = np.dot(A3.T,dZ4)
            db4 = np.sum(dZ4, axis=0)/len(dZ4)
            dA3 = np.dot(dZ4,W4.T)
            dZ3 = np.multiply(dA3,(Z3 > 0))
            dW3 = np.dot(P2_flat.T,dZ3)
            db3 = np.sum(dZ3, axis=0)/len(dZ3)
            dP2_flat = np.dot(dZ3,W3.T)
            dP2_flat = np.array(dP2_flat)
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
            print("Finished running "+str(count)+" minibatches, and the cost is "+ str(cost))
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
    P2_flat = P2.reshape((-1, 4 * 4 * 64))
    Z3 = np.dot(P2_flat, W3) + b3
    A3 = Z3 * (Z3 > 0)
    Z4 = np.dot(A3, W4) + b4
    temp = np.matrix(np.max(Z4, axis=1)).T
    res = temp.repeat(10, axis=1)
    exps = np.exp(Z4 - res)
    A4 = exps / np.sum(exps, axis =1)
    Y_predict = np.argmax(A4, axis=1)
    counter = np.sum(Y_predict == Y)
    ac = counter/len(Y)
    print("Accuracy is "+str(ac))

def main():
    Y_train, X_train, Y_test, X_test = readData()
    (W1, b1, W2, b2, W3, b3, W4, b4) = model(X_train,Y_train, epoch_num= 20)
    params = (W1, b1, W2, b2, W3, b3, W4, b4)
    print("Traning set:")
    predict(params, X_train, Y_train)
    print("Test set:")
    predict(params, X_test, Y_test)

if __name__ == '__main__':
    main()