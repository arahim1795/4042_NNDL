import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pylab as plt
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#general
epochs = 10
learning_rate= 0.01
EMBEDDED_SIZE = 20
MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_CHAR = 256
N_FILTERS = 10
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

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
    trainData, testData = [],[]
    trainData.append(x_train)
    trainData.append(y_train)
    testData.append(x_test)
    testData.append(y_test)
    return trainData, testData, no_words

def data_read_chars():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('../Data/train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open('../Data/test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))
    
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    
    
    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values

    trainData, testData = [],[]
    trainData.append(x_train)
    trainData.append(y_train)
    testData.append(x_test)
    testData.append(y_test)
    
    return trainData,testData

def char_rnn_model_vanilla_2layer(train_data,test_data,keep_probability):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    #input layer
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, MAX_CHAR])
    inputs = tf.unstack(input_layer,axis=1)

    #hidden layer
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE),tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)]) 
    _, encoding = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
    encoding = encoding[-1]
    dropped = tf.nn.dropout(encoding, keep_probability)  # DROP-OUT here
    #output layer
    logits = tf.layers.dense(dropped, MAX_LABEL, activation=None)

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

def word_rnn_model_vanilla_2layer(train_data,test_data,keep_probability):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    #input layer
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=no_words, embed_dim=EMBEDDED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)
    input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDED_SIZE])

    #hidden layer
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE),tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)]) 
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
    encoding = encoding[-1]
    dropped = tf.nn.dropout(encoding, keep_probability)  # DROP-OUT here
    #output layer
    logits = tf.layers.dense(dropped, MAX_LABEL, activation=None)
    
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

def char_rnn_model_LSTM_2layer(train_data,test_data,keep_probability):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    #input layer
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, MAX_CHAR])
    inputs = tf.unstack(input_layer,axis=1)

    #hidden layer
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE),tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)]) 
    _, encoding = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
    encoding = encoding[-1]
    dropped = tf.nn.dropout(encoding, keep_probability)  # DROP-OUT here
    #output layer
    logits = tf.layers.dense(dropped, MAX_LABEL, activation=None)

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

def word_rnn_model_LSTM_2layer(train_data,test_data,keep_probability):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    #input layer
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=no_words, embed_dim=EMBEDDED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)
    input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDED_SIZE])

    #hidden layer
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE),tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)]) 
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
    encoding = encoding[-1]
    dropped = tf.nn.dropout(encoding, keep_probability)  # DROP-OUT here
    #output layer
    logits = tf.layers.dense(dropped, MAX_LABEL, activation=None)
    
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

def char_rnn_model(train_data,test_data,keep_probability):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    #input layer
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, MAX_CHAR])
    inputs = tf.unstack(input_layer,axis=1)

    #hidden layer
    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE) 
    _, encoding = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
    dropped = tf.nn.dropout(encoding, keep_probability)  # DROP-OUT here
    #output layer
    logits = tf.layers.dense(dropped, MAX_LABEL, activation=None)

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

def word_rnn_model(train_data,test_data,keep_probability):
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    #input layer
    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=no_words, embed_dim=EMBEDDED_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)
    input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDED_SIZE])

    #hidden layer
    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE) 
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
    dropped = tf.nn.dropout(encoding, keep_probability)  # DROP-OUT here
    #output layer
    logits = tf.layers.dense(dropped, MAX_LABEL, activation=None)
    
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

def main():
    global no_words
    train_word, test_word, no_words= data_read_words()
    train_char, test_char = data_read_chars()

    char_rnn_data = char_rnn_model(train_char,test_char,1)
    char_rnn_data_vanilla_2layer = char_rnn_model_vanilla_2layer(train_char,test_char,1)
    char_rnn_data_LSTM_2layer = char_rnn_model_LSTM_2layer(train_char,test_char,1)
    word_rnn_data = word_rnn_model(train_word,test_word,1)
    word_rnn_data_vanilla_2layer = word_rnn_model_vanilla_2layer(train_word,test_word,1)
    word_rnn_data_LSTM_2layer = word_rnn_model_LSTM_2layer(train_word,test_word,1)

    accuracy_list,entropy_list = [],[]
    accuracy_list.append(char_rnn_data[0])
    accuracy_list.append(char_rnn_data_vanilla_2layer[0])
    accuracy_list.append(char_rnn_data_LSTM_2layer[0])
    accuracy_list.append(word_rnn_data[0])
    accuracy_list.append(word_rnn_data_vanilla_2layer[0])
    accuracy_list.append(word_rnn(data_LSTM_2layer[0]))
    entropy_list.append(char_rnn_data[0])
    entropy_list.append(char_rnn_data_2layer[0])
    entropy_list.append(word_rnn_data[0])
    entropy_list.append(word_rnn_data_2layer[0])


    name_list = ["Char RNN Original", "Char RNN 2 Layer", "Word RNN Original","Word RNN 2 Layer"]

    fig1 = plt.figure(figsize=(16,8))
    for i in range(4):
        plt.plot(range(epochs),entropy_list[i],label="Entropy Cost for " + str(name_list[i]))
    plt.xlabel("Epochs")  
    plt.ylabel("Entropy Cost")
    plt.legend()
    fig1.savefig("../Out/B6b_Cost.png")

    fig2 = plt.figure(figsize=(16,8))
    for i in range(4):
        plt.plot(range(epochs),accuracy_list[i],label="Test Accuracy for " + str(name_list[i]))
    plt.xlabel("Epochs")
    plt.ylabel("Train Accuracy")
    plt.legend()
    fig2.savefig("../Out/B6b_Accuracy.png")

    with open("../Out/6b.csv", "w") as f:
        f.write("type,epoch,test accuracy,entropy_cost\n")
        for i in range(4):
            for e in range(epochs):
                f.write("%s,%s,%s,%s\n" % (name_list[i],str(e), str(accuracy_list[i][e]), str(entropy_list[i][e])))

if __name__ == '__main__':
    main()