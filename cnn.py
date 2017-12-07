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

    with open(test_label, 'rb') as testLabelFile:
        struct.unpack(">II", testLabelFile.read(8))
        te_label = np.fromfile(testLabelFile, dtype=np.int8)

    with open(test_img, 'rb') as testImgFile:
        magic, num, rows, cols = struct.unpack(">IIII", testImgFile.read(16))
        te_img = np.fromfile(testImgFile, dtype=np.uint8).reshape(len(te_label), rows, cols)

    return tr_label, tr_img, te_label, te_img
