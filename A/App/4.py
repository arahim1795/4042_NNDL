import math
import tensorflow as tf
import matplotlib.pylab as plt
import multiprocessing as mp
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.utils import gen_batches

# import matplotlib.pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# * Hyper-Parameters
FEATURE_INPUT = 21  # input: LB to Tendency
NUM_CLASSES = 3  # NSP = 1, 2, 3

batch = 32
decay = [0, math.pow(10, -3), math.pow(10, -6), math.pow(10, -9), math.pow(10, -12)]
epochs = 20000
learning_rate = 0.01
num_neurons = 15

k_value = 5


# * Functions
def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


def process_data(file):
    # parameters
    processed_data = []

    # import data
    raw_data = np.genfromtxt(file, delimiter=",")

    # process X
    X = raw_data[:, :FEATURE_INPUT]
    X = scale(X, np.min(X, axis=0), np.max(X, axis=0))

    # process Y
    Y = raw_data[:, -1].astype(int)
    Y_one_hot = np.zeros((Y.shape[0], NUM_CLASSES))
    Y_one_hot[np.arange(Y.shape[0]), Y - 1] = 1

    # append processed data to list
    processed_data.append(X)
    processed_data.append(Y_one_hot)

    return processed_data


def process_data_k(file):
    # parameters
    permutated_train, permutated_test = [], []
    train_indexes, test_indexes = [], []

    # import data
    raw_data = np.genfromtxt(file, delimiter=",")

    # create index splitter
    kf = KFold(n_splits=k_value)

    # get splitted index
    for train, test in kf.split(raw_data):
        train_indexes.append(train)
        test_indexes.append(test)

    # process X
    X = raw_data[:, :FEATURE_INPUT]
    X = scale(X, np.min(X, axis=0), np.max(X, axis=0))

    # process Y
    Y = raw_data[:, -1].astype(int)
    Y_one_hot = np.zeros((Y.shape[0], NUM_CLASSES))
    Y_one_hot[np.arange(Y.shape[0]), Y - 1] = 1

    # append processed data to list
    for a in range(k_value):
        train_data, test_data = [], []
        train_data.append(X[train_indexes[a]])
        train_data.append(Y_one_hot[train_indexes[a]])
        test_data.append(X[test_indexes[a]])
        test_data.append(Y_one_hot[test_indexes[a]])
        permutated_train.append(train_data)
        permutated_test.append(test_data)

    return permutated_train, permutated_test


def process_data_batch(data, batch):
    # parameters
    entries = len(data[0])
    batched_data = []

    # slicer
    slices = gen_batches(entries, batch)

    # batching
    for s in slices:
        data_store = []
        data_store.append(data[0][s])
        data_store.append(data[1][s])
        batched_data.append(data_store)

    return batched_data


def randomise_order(dataset):
    # generate indexes
    indexes = np.arange(len(dataset[0]))

    # randomise indexes
    np.random.seed(291)
    np.random.shuffle(indexes)

    # randomise dataset order
    dataset[0] = dataset[0][indexes]
    dataset[1] = dataset[1][indexes]

    return dataset


def nn_model(batched_train_data, train_data, test_data, decay):
    # * Create Model
    x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

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

    # final layer: softmax
    layer_final_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, NUM_CLASSES], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="final_weights",
    )
    layer_final_biases = tf.Variable(tf.zeros([NUM_CLASSES]), name="final_biases")
    logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases

    # regularise (l2)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    loss = tf.reduce_mean(loss + (decay * tf.nn.l2_loss(logits)))

    # minimise loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(
        tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32
    )
    accuracy = tf.reduce_mean(correct_prediction)

    # * Run Model
    train_acc, test_acc = [], []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            # train
            for data in batched_train_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            # evaluate
            train_acc.append(
                accuracy.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            )
            test_acc.append(
                accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1]})
            )

            if (i % 100) == 0:
                print("epoch: ", i, " tr-acc: ", train_acc[i], "te-acc: ", test_acc[i])

            # randomise dataset
            train_data = randomise_order(train_data)

            random.shuffle(batched_train_data)

            for data in batched_train_data:
                data = randomise_order(data)

    acc = []
    acc.append(train_acc)
    acc.append(test_acc)

    return acc


def export_data(acc, epochs):
    # * Export Accuracies
    with open("../Out/4_optimal.csv", "w") as f:
        f.write("iter,tr-acc,te-acc\n")
        for j in range(0, epochs):
            f.write("%s,%s,%s\n" % (str(j), acc[0][j], acc[1][j]))

    # * Plotting
    plt.figure(figsize=(16, 8))
    fig = plt.figure(1)
    plt.plot(range(epochs), acc[0], label="Train Accuracy", color="#ff0000")
    plt.plot(range(epochs), acc[1], label="Test Accuracy", color="#00ffff")
    plt.xlabel(str(epochs) + " iterations")
    plt.ylabel("Train/Test accuracy")
    plt.legend()
    fig.savefig("../Out/4_optimal_fig.png")
    plt.close()


def export_data_decay(acc, k_value, epochs):

    extracted_decay = [0, 3, 6, 9, 12]
    zipped_decay = []
    for i in range(len(extracted_decay)):
        for j in range(k_value):
            zipped_decay.append(extracted_decay[i])

    for i in range(0, len(acc), 5):
        #  mean validation
        mean_train = (
            np.array(acc[i][0])
            + np.array(acc[i + 1][0])
            + np.array(acc[i + 2][0])
            + np.array(acc[i + 3][0])
            + np.array(acc[i + 4][0])
        )

        mean_test = (
            np.array(acc[i][1])
            + np.array(acc[i + 1][1])
            + np.array(acc[i + 2][1])
            + np.array(acc[i + 3][1])
            + np.array(acc[i + 4][1])
        )

        mean_train = np.divide(mean_train, 5.0)
        mean_test = np.divide(mean_test, 5.0)

        # export accuracies
        with open("../Out/4_d" + str(zipped_decay[i]) + ".csv", "w") as f:
            f.write("iter,tr-acc,te-acc\n")
            for j in range(0, epochs):
                f.write("%s,%s,%s\n" % (str(j), mean_train[j], mean_test[j]))

        # plotting
        fig = plt.figure(1, figsize=(16, 8))
        plt.plot(range(epochs), mean_train, label="mean_train", color="#ff0000")
        plt.plot(range(epochs), mean_test, label="mean_test", color="#00ffff")

        plt.xlabel(str(epochs) + " iterations")
        plt.ylabel("Train/Test accuracy")
        plt.legend()
        fig.savefig("../Out/4_d" + str(zipped_decay[i]) + ".png")
        plt.close()


def main():
    # * Determining Optimal Batch
    # * Parameters
    train_file = "../Data/train_data.csv"

    # * Multiprocessing Setup
    num_threads = mp.cpu_count() - 1
    p = mp.Pool(processes=num_threads)

    # * Data Handling
    # * k-fold validation
    # import k-fold train data
    k_train_data, k_test_data = process_data_k(train_file)

    # batching
    batch_train_set = []
    for i in range(len(decay)):
        for j in range(k_value):
            batch_train_set.append(process_data_batch(k_train_data[j], batch))

    # copy data for zipping
    copy_train = k_train_data
    copy_test = k_test_data
    for i in range(len(decay) - 1):
        k_train_data += copy_train
        k_test_data += copy_test

    zipped_decay = []
    for i in range(len(decay)):
        for j in range(k_value):
            zipped_decay.append(decay[i])

    # * Execution (K-Fold)
    acc = p.starmap(
        nn_model, zip(batch_train_set, k_train_data, k_test_data, zipped_decay)
    )

    export_data_decay(acc, k_value, epochs)

    # * Optimal Decay (10^-3)
    # * Data Handling
    optimal_decay = math.pow(10, -3)
    test_file = "../Data/test_data.csv"

    # import train data
    train_70 = process_data(train_file)

    # import test data
    test_30 = process_data(test_file)

    optimal_batches_data = process_data_batch(train_70, batch)

    # * Execution (Optimal Size)
    optimal_acc = nn_model(optimal_batches_data, train_70, test_30, optimal_decay)

    export_data(optimal_acc, epochs)


if __name__ == "__main__":
    main()
