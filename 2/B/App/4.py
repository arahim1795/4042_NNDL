import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pylab as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDED_SIZE = 20
batch_size = 128
epochs = 100
learning_rate = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def word_rnn_model(x):
  #input layer
  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=no_words, embed_dim=EMBEDDED_SIZE)
  word_list = tf.unstack(word_vectors, axis=1)
  input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDED_SIZE])

  #hidden layer
  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE) 
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  #output layer
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  print(logits)
  return logits, word_list

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

  return x_train, y_train, x_test, y_test, no_words

def main():
  global no_words
  trainX, trainY, testX, testY, no_words= data_read_words()

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, inputs = word_rnn_model(x)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(learning_rate).minimize(entropy)

  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_,MAX_LABEL),1)), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # training
  test_accuracy,entropy_cost = [],[]
  with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            N = len(trainX)
            idx = np.arange(N)
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx] #shuffle
            # Mini-batch training
            for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
                sess.run(train_op, {x: trainX[start:end], y_: trainY[start:end]})
            # evaluation    
            acc_,loss_ = sess.run([accuracy, entropy], {x: testX, y_: testY})
            test_accuracy.append(acc_)
            entropy_cost.append(loss_)
            print('epoch', e, 'entropy', loss_,'accuracy', acc_)

        fig1 = plt.figure(figsize=(16,8))
        plt.plot(range(epochs),entropy_cost,label="Entropy Cost")
        plt.xlabel("Epochs")
        plt.ylabel("Entropy Cost")
        plt.legend()
        fig1.savefig("../Out/B4_Cost.png")

        fig2 = plt.figure(figsize=(16,8))
        plt.plot(range(epochs),test_accuracy,label="Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Entropy Cost")
        plt.legend()
        fig2.savefig("../Out/B4_Accuracy.png")

  with open("../Out/4.csv", "w") as f:
        f.write("epoch,test accuracy,entropy_cost\n")
        for e in range(epochs):
          f.write("%s,%s,%s\n" % (str(e), str(test_accuracy[e]), str(entropy_cost[e])))

if __name__ == '__main__':
  main()
