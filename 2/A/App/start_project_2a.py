#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle



NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = [3,50,60]
learning_rate = 0.001
epochs = 10
batch_size = 128


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def scaleImage(data):
    scaledData = data/255.0
    return scaledData

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_




def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS[0]])
    
    # Conv 1
    # input: 32px 32px *3 channels
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS[0], NUM_CHANNELS[1]], stddev=1.0/np.sqrt(NUM_CHANNELS[0]*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros(NUM_CHANNELS[1]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

    dim_sum_1 = pool_1.get_shape()[1].value * pool_1.get_shape()[2].value * pool_1.get_shape()[3].value 
    # pool_1_flat = tf.reshape(pool_1, [-1, dim_sum_1])

    # Conv 2
    W2 = tf.Variable(tf.truncated_normal([5,5,NUM_CHANNELS[1], NUM_CHANNELS[2]], stddev=1.0/np.sqrt(NUM_CHANNELS[1]*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros(NUM_CHANNELS[2]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    dim_sum_2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    pool_2_flat = tf.reshape(pool_2, [-1, dim_sum_2])
    
    # Fully connected layer
    # TODO: try with sigmoidal + relu activation functions
    W_connected = tf.Variable(tf.truncated_normal([dim_sum_2,300],stddev=1.0/np.sqrt(dim_sum_2),name='weights_connected'))
    b_connected = tf.Variable(tf.zeros(300), name = 'biases_connected')
    connected_output = tf.matmul(pool_2_flat,W_connected) + b_connected

    #Softmax
    W_softmax = tf.Variable(tf.truncated_normal([300, 10], stddev=1.0/np.sqrt(300)), name='weights_softmax')
    b_softmax = tf.Variable(tf.zeros([10]), name='biases_3')
    logits = tf.matmul(connected_output, W_softmax) + b_softmax

    return logits


def main():

    trainX, trainY = load_data('../Data/data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('../Data/test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS[0]])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    
    logits = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            _, loss_ = sess.run([train_step, loss], {x: trainX, y_: trainY})

            print('epoch', e, 'entropy', loss_)


    ind = np.random.randint(low=0, high=10000)
    X = trainX[ind,:]
    
    plt.figure()
    plt.gray()
    X_show = X.reshape(IMG_SIZE, IMG_SIZE, NUM_CHANNELS).transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(X_show)
    plt.savefig('./p1b_2.png')


if __name__ == '__main__':
  main()
