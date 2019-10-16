import math
import multiprocessing as mp
import numpy as np
import pylab as plt
from sklearn.utils import gen_batches
import tensorflow as tf
import time


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# hyper-parameters
np.random.seed(291)
FEATURE_INPUT = 5

batch_size = 8
decay = math.pow(10, -3)
epochs = 100000
keep_probability = 0.8
learning_rate = math.pow(10, -3)
num_neurons = 50


# * Function
def scale(data):
    data_scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data_scaled


def process_data(file):
    # parameters
    data = []

    # import data
    raw_data = np.genfromtxt(file, delimiter=",")

    # process and split input and observations
    X, Y = raw_data[:, 1:8], raw_data[:, -1]
    X = scale(X)
    Y = Y.reshape(Y.shape[0], 1)

    data.append(X)
    data.append(Y)

    return data


def process_drop_feature(data, feature_num):
    # parameter
    dataset = []

    #  drop feature
    dropped_data = np.delete(data[0], feature_num, axis=1)

    # append to list
    dataset.append(dropped_data)
    dataset.append(data[1])

    return dataset


def randomise_data(data):
    idx = np.arange(len(data[0]))
    np.random.shuffle(idx)

    data[0] = data[0][idx]
    data[1] = data[1][idx]

    return data


def process_data_batch(data):
    # parameters
    entries = len(data[0])
    batched_data = []

    # slicer
    slices = gen_batches(entries, 8)

    # batching
    for s in slices:
        data_store = []
        data_store.append(data[0][s])
        data_store.append(data[1][s])
        batched_data.append(data_store)

    return batched_data

def nn_model_3(train_data, test_data):
    x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # hidden layer 1: relu
    layer_1_weights = tf.Variable(
        tf.truncated_normal(
            [FEATURE_INPUT, num_neurons], stddev=1.0 / math.sqrt(float(FEATURE_INPUT))
        ),
        name="one_weights",
    )
    layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
    layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

    layer_1_output = tf.nn.relu(layer_1_var)

    # final layer: linear
    layer_final_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, 1], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="final_weights",
    )
    layer_final_biases = tf.Variable(tf.zeros([1]), name="final_biases")
    logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases

    # regularise (l2)
    regularization = (
        tf.nn.l2_loss(layer_1_weights)
        + tf.nn.l2_loss(layer_final_weights)
    )
    loss = tf.reduce_mean(tf.square(y_ - logits))
    l2_loss = tf.reduce_mean(loss + decay * regularization)

    # minimise loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(l2_loss)

    train_loss, test_loss, per_epoch_time = [], [], []

    # run model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            # batching
            batched_data = process_data_batch(train_data)

            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            train_loss.append(
                l2_loss.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            )

            test_loss.append(
                l2_loss.eval(feed_dict={x: test_data[0], y_: test_data[1]})
            )

            # randomise
            train_data = randomise_data(train_data)
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_loss)
    data.append(test_loss)
    data.append(per_epoch_time)

    return data

def nn_model_4(train_data, test_data):
    x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # hidden layer 1: relu
    layer_1_weights = tf.Variable(
        tf.truncated_normal(
            [FEATURE_INPUT, num_neurons], stddev=1.0 / math.sqrt(float(FEATURE_INPUT))
        ),
        name="one_weights",
    )
    layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
    layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

    layer_1_output = tf.nn.relu(layer_1_var)

    # hidden layer 2: relu
    layer_2_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, num_neurons], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="two_weights",
    )
    layer_2_biases = tf.Variable(tf.zeros([num_neurons]), name="two_biases")
    layer_2_var = tf.matmul(layer_1_output, layer_2_weights) + layer_2_biases

    layer_2_output = tf.nn.relu(layer_2_var)

    # final layer: linear
    layer_final_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, 1], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="final_weights",
    )
    layer_final_biases = tf.Variable(tf.zeros([1]), name="final_biases")
    logits = tf.matmul(layer_2_output, layer_final_weights) + layer_final_biases

    # regularise (l2)
    regularization = (
        tf.nn.l2_loss(layer_1_weights)
        + tf.nn.l2_loss(layer_2_weights)
        + tf.nn.l2_loss(layer_final_weights)
    )
    loss = tf.reduce_mean(tf.square(y_ - logits))
    l2_loss = tf.reduce_mean(loss + decay * regularization)

    # minimise loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(l2_loss)

    train_loss, test_loss, per_epoch_time = [], [], []

    # run model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            # batching
            batched_data = process_data_batch(train_data)

            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            train_loss.append(
                l2_loss.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            )

            test_loss.append(
                l2_loss.eval(feed_dict={x: test_data[0], y_: test_data[1]})
            )

            # randomise
            train_data = randomise_data(train_data)
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_loss)
    data.append(test_loss)
    data.append(per_epoch_time)

    return data


def nn_model_4d(train_data, test_data):
    x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # hidden layer 1: relu
    layer_1_weights = tf.Variable(
        tf.truncated_normal(
            [FEATURE_INPUT, num_neurons], stddev=1.0 / math.sqrt(float(FEATURE_INPUT))
        ),
        name="one_weights",
    )
    layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
    layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

    layer_1_output = tf.nn.relu(layer_1_var)

    layer_1_dropout = tf.nn.dropout(layer_1_output, keep_probability)

    # hidden layer 2: relu
    layer_2_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, num_neurons], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="two_weights",
    )
    layer_2_biases = tf.Variable(tf.zeros([num_neurons]), name="two_biases")
    layer_2_var = tf.matmul(layer_1_dropout, layer_2_weights) + layer_2_biases

    layer_2_output = tf.nn.relu(layer_2_var)

    layer_2_dropout = tf.nn.dropout(layer_2_output, keep_probability)

    # final layer: linear
    layer_final_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, 1], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="final_weights",
    )
    layer_final_biases = tf.Variable(tf.zeros([1]), name="final_biases")
    logits = tf.matmul(layer_2_dropout, layer_final_weights) + layer_final_biases

    # regularise (l2)
    regularization = (
        tf.nn.l2_loss(layer_1_weights)
        + tf.nn.l2_loss(layer_2_weights)
        + tf.nn.l2_loss(layer_final_weights)
    )
    loss = tf.reduce_mean(tf.square(y_ - logits))
    l2_loss = tf.reduce_mean(loss + decay * regularization)

    # minimise loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(l2_loss)

    train_loss, test_loss, per_epoch_time = [], [], []

    # run model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            # batching
            batched_data = process_data_batch(train_data)

            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            train_loss.append(
                l2_loss.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            )

            test_loss.append(
                l2_loss.eval(feed_dict={x: test_data[0], y_: test_data[1]})
            )

            # randomise
            train_data = randomise_data(train_data)
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_loss)
    data.append(test_loss)
    data.append(per_epoch_time)

    return data


def nn_model_5(train_data, test_data):
    x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # hidden layer 1: relu
    layer_1_weights = tf.Variable(
        tf.truncated_normal(
            [FEATURE_INPUT, num_neurons], stddev=1.0 / math.sqrt(float(FEATURE_INPUT))
        ),
        name="one_weights",
    )
    layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
    layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

    layer_1_output = tf.nn.relu(layer_1_var)

    # hidden layer 2: relu
    layer_2_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, num_neurons], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="two_weights",
    )
    layer_2_biases = tf.Variable(tf.zeros([num_neurons]), name="two_biases")
    layer_2_var = tf.matmul(layer_1_output, layer_2_weights) + layer_2_biases

    layer_2_output = tf.nn.relu(layer_2_var)

    # hidden layer 3: relu
    layer_3_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, num_neurons], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="two_weights",
    )
    layer_3_biases = tf.Variable(tf.zeros([num_neurons]), name="two_biases")
    layer_3_var = tf.matmul(layer_2_output, layer_3_weights) + layer_3_biases

    layer_3_output = tf.nn.relu(layer_3_var)

    # final layer: linear
    layer_final_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, 1], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="final_weights",
    )
    layer_final_biases = tf.Variable(tf.zeros([1]), name="final_biases")
    logits = tf.matmul(layer_3_output, layer_final_weights) + layer_final_biases

    # regularise (l2)
    regularization = (
        tf.nn.l2_loss(layer_1_weights)
        + tf.nn.l2_loss(layer_2_weights)
        + tf.nn.l2_loss(layer_3_weights)
        + tf.nn.l2_loss(layer_final_weights)
    )
    loss = tf.reduce_mean(tf.square(y_ - logits))
    l2_loss = tf.reduce_mean(loss + decay * regularization)

    # minimise loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(l2_loss)

    train_loss, test_loss, per_epoch_time = [], [], []

    # run model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            # batching
            batched_data = process_data_batch(train_data)

            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            train_loss.append(
                l2_loss.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            )

            test_loss.append(
                l2_loss.eval(feed_dict={x: test_data[0], y_: test_data[1]})
            )

            # randomise
            train_data = randomise_data(train_data)
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_loss)
    data.append(test_loss)
    data.append(per_epoch_time)

    return data


def nn_model_5d(train_data, test_data):
    x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # hidden layer 1: relu
    layer_1_weights = tf.Variable(
        tf.truncated_normal(
            [FEATURE_INPUT, num_neurons], stddev=1.0 / math.sqrt(float(FEATURE_INPUT))
        ),
        name="one_weights",
    )
    layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
    layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

    layer_1_output = tf.nn.relu(layer_1_var)

    layer_1_dropout = tf.nn.dropout(layer_1_output, keep_probability)

    # hidden layer 2: relu
    layer_2_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, num_neurons], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="two_weights",
    )
    layer_2_biases = tf.Variable(tf.zeros([num_neurons]), name="two_biases")
    layer_2_var = tf.matmul(layer_1_dropout, layer_2_weights) + layer_2_biases

    layer_2_output = tf.nn.relu(layer_2_var)

    layer_2_dropout = tf.nn.dropout(layer_2_output, keep_probability)

    # hidden layer 3: relu
    layer_3_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, num_neurons], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="two_weights",
    )
    layer_3_biases = tf.Variable(tf.zeros([num_neurons]), name="two_biases")
    layer_3_var = tf.matmul(layer_2_dropout, layer_3_weights) + layer_3_biases

    layer_3_output = tf.nn.relu(layer_3_var)

    layer_3_dropout = tf.nn.dropout(layer_3_output, keep_probability)

    # final layer: linear
    layer_final_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, 1], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="final_weights",
    )
    layer_final_biases = tf.Variable(tf.zeros([1]), name="final_biases")
    logits = tf.matmul(layer_3_dropout, layer_final_weights) + layer_final_biases

    # regularise (l2)
    regularization = (
        tf.nn.l2_loss(layer_1_weights)
        + tf.nn.l2_loss(layer_2_weights)
        + tf.nn.l2_loss(layer_3_weights)
        + tf.nn.l2_loss(layer_final_weights)
    )
    loss = tf.reduce_mean(tf.square(y_ - logits))
    l2_loss = tf.reduce_mean(loss + decay * regularization)

    # minimise loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(l2_loss)

    train_loss, test_loss, per_epoch_time = [], [], []

    # run model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            # batching
            batched_data = process_data_batch(train_data)

            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            train_loss.append(
                l2_loss.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            )

            test_loss.append(
                l2_loss.eval(feed_dict={x: test_data[0], y_: test_data[1]})
            )

            # randomise
            train_data = randomise_data(train_data)
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_loss)
    data.append(test_loss)
    data.append(per_epoch_time)

    return data

def plot_all(dataset_zero,dataset_one,dataset_one_drop,dataset_two,dataset_two_drop):
    test_accuracies = []
    test_accuracies.append(dataset_zero)
    test_accuracies.append(dataset_one)
    test_accuracies.append(dataset_one_drop)
    test_accuracies.append(dataset_two)
    test_accuracies.append(dataset_two_drop)

    fig1 = plt.figure(figsize=(16, 8))
    plt.plot(range(epochs), test_accuracies[0], label = "3 Layer", color = "#FF0000")
    plt.plot(range(epochs), test_accuracies[1], label = "4 Layer", color = "#0000FF")
    plt.plot(range(epochs), test_accuracies[2], label = "4 Layer Dropout", color = "#008000")
    plt.plot(range(epochs), test_accuracies[3], label = "5 Layer", color = "#FFA500")
    plt.plot(range(epochs), test_accuracies[4], label = "5 Layer Dropout", color = "#FFC0CB")
    plt.xlabel(str(epochs) + " iterations")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.ylim(0.002,0.02)
    fig1.savefig("../Out/4.png")


def main():
    # setup multiprocessing
    num_threads = mp.cpu_count() - 1
    p = mp.Pool(processes=num_threads)

    # process data
    file_train = "../Data/train_data.csv"
    file_test = "../Data/test_data.csv"

    # train data contains 2 (x and y) X 280 X 7/1 (x or y inputs)
    train_data = process_data(file_train)
    test_data = process_data(file_test)

    # Remove 1st column: Research
    dropped_one_train, dropped_one_test = [], []
    dropped_one_train.append(process_drop_feature(train_data, 6))
    dropped_one_test.append(process_drop_feature(test_data, 6))

    # drop 2nd feature
    dropped_two_train, dropped_two_test = [], []
    dropped_two_train.append(process_drop_feature(dropped_one_train[0], 1))
    dropped_two_test.append(process_drop_feature(dropped_one_test[0], 1))
    # zipping dataset

    dataset_zero = p.starmap(
        nn_model_3, zip(dropped_two_train,dropped_two_test)
    )
    dataset_one = p.starmap(
        nn_model_4, zip(dropped_two_train, dropped_two_test)
    )
    
    dataset_one_drop = p.starmap(
        nn_model_4d, zip(dropped_two_train, dropped_two_test)
    )

    # execute RFE on 6 features
    dataset_two = p.starmap(
        nn_model_5, zip(dropped_two_train, dropped_two_test)
    )

    dataset_two_drop = p.starmap(
        nn_model_5d, zip(dropped_two_train, dropped_two_test)
    )

    plot_all(dataset_zero[0][1],dataset_one[0][1],dataset_one_drop[0][1],dataset_two[0][1],dataset_two_drop[0][1])



if __name__ == "__main__":
    main()
