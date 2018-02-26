import tensorflow as tf
import gzip
import os
from  six.moves import urllib
import numpy as np

SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
DATA_DIRECTORY ="data"

image_size = 28
num_channels = 1
pixel_depth = 255
num_labels = 10
validation_size = 5000

def maybe_download(filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)

    if not tf.gfile.Exists(filepath):
        filepath,_ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size  = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename,num_images,norm_scale = True):
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size*image_size*num_images*num_channels)
        data = np.frombuffer(buf,dtype = np.uint8).astype(np.float32)
        if norm_scale:
            data = data/pixel_depth
        data = data.reshape(num_images,image_size,image_size,num_channels)
        data = np.reshape(data,[num_images,-1])
        return data

def  extract_labels(filename,num_images):

    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,num_labels))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, num_labels])
    return one_hot_encoding



def prepare_MNIST_Data():
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    train_data = extract_data(train_data_filename,60000)
    train_labels = extract_labels(train_labels_filename,60000)
    test_data = extract_data(test_data_filename,10000)
    test_labels = extract_labels(test_labels_filename,10000)

    validation_data = train_data[:validation_size,:]
    validation_labels = train_labels[:validation_size,:]
    train_data = train_data[validation_size:,:]
    train_labels = train_labels[validation_size:,:]

    return train_data,train_labels,validation_data,validation_labels,test_data,test_labels

