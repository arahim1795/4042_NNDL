import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pylab as plt
from tqdm import tqdm

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE = [[20, 20],[20,1]] # original window size
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
batch_size = 128
epochs = 100
learning_rate= 0.01
EMBEDDED_SIZE = 20

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def word_cnn_model(train_data,test_data,keep_probability):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    # Embedding Layer
    # embedding_layer = layers.Embedding(x,20,input_length=20)
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=no_words, embed_dim=EMBEDDED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)
    input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDED_SIZE,1])
    # CNN layer 1
    conv_1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE[0],
        padding='VALID',
        activation=tf.nn.relu)
    # Pooling layer 1
    pool_1 = tf.layers.max_pooling2d(
        conv_1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    drop_1 = tf.nn.dropout(pool_1, keep_probability)  # DROP-OUT here
    # CNN layer 2
    conv_2 = tf.layers.conv2d(
        drop_1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE[1],
        padding='VALID',
        activation=tf.nn.relu)
    # Pooling layer 1
    pool_2 = tf.layers.max_pooling2d(
        conv_2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    drop_2 = tf.nn.dropout(pool_2, keep_probability)  # DROP-OUT here
    dim_sum_2 = drop_2.get_shape()[1].value * drop_2.get_shape()[2].value * drop_2.get_shape()[3].value
    drop_2_flat = tf.reshape(drop_2, [-1, dim_sum_2])
    #pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

    # Softmax Layer
    W_softmax = tf.Variable(tf.truncated_normal([dim_sum_2, 15], stddev=1.0/np.sqrt(dim_sum_2)), name='weights_softmax')
    b_softmax = tf.Variable(tf.zeros([15]), name='biases_3')
    logits = tf.matmul(drop_2_flat, W_softmax) + b_softmax
    
    test_accuracy,entropy_cost = [],[]
    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(entropy)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_,MAX_LABEL),1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)


    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in tqdm(range(epochs)):
            idx = np.arange(len(train_data[0]))
            np.random.shuffle(idx)
            trainX, trainY = train_data[0][idx], train_data[1][idx] #shuffle
            # Mini-batch training
            for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
                sess.run(train_op, {x: trainX[start:end], y_: trainY[start:end]})
            # evaluation    
            acc_,loss_ = sess.run([accuracy, entropy], {x: test_data[0], y_: test_data[1]})
            test_accuracy.append(acc_)
            entropy_cost.append(entropy.eval(feed_dict={x: train_data[0], y_: train_data[1]}))
        sess.close()
    tf.reset_default_graph()
    data = []
    data.append(test_accuracy)
    data.append(entropy_cost)
    return data


def data_read_words():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('../Data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("../Data/test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)
    trainData, testData = [],[]
    trainData.append(x_train)
    trainData.append(y_train)
    testData.append(x_test)
    testData.append(y_test)
    return trainData, testData, no_words


  
def main():
  global no_words
  train_data, test_data, no_words= data_read_words()

  word_cnn_data = word_cnn_model(train_data,test_data,1)

  fig1 = plt.figure(figsize=(16,8))
  plt.plot(range(epochs),word_cnn_data[0],label="Test Accuracy for Word CNN")
  plt.xlabel("Epochs")
  plt.ylabel("Train Accuracy")
  plt.legend()
  fig1.savefig("../Out/B2_Accuracy.png")

  fig2 = plt.figure(figsize=(16,8))
  plt.plot(range(epochs),word_cnn_data[1],label="Entropy Cost for Word CNN")
  plt.xlabel("Epochs")  
  plt.ylabel("Entropy Cost")
  plt.legend()
  fig2.savefig("../Out/B2_Cost.png")

  with open("../Out/2.csv", "w") as f:
    f.write("epoch,test accuracy,entropy_cost\n")
    for e in range(epochs):
      f.write("%s,%s,%s\n" % (str(e), str(word_cnn_data[0][e]), str(word_cnn_data[1][e])))

if __name__ == '__main__':
  main()
