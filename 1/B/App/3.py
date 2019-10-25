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
FEATURE_INPUT = [7, 6]

learning_rate = math.pow(10, -3)
epochs = 50000
num_neurons = 10
batch_size = 8
decay = math.pow(10, -3)


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
    slices = gen_batches(entries, batch_size)

    # batching
    for s in slices:
        data_store = []
        data_store.append(data[0][s])
        data_store.append(data[1][s])
        batched_data.append(data_store)

    return batched_data


def nn_model(train_data, test_data, FEATURE_INPUT):
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
    regularization = tf.nn.l2_loss(layer_1_weights) + tf.nn.l2_loss(layer_final_weights)
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

            if (i % 1000) == 0:
                print(
                    "epoch: "
                    + str(i)
                    + " tr-loss: "
                    + str(train_loss[i])
                    + " te-less: "
                    + str(test_loss[i])
                )

            # randomise
            train_data = randomise_data(train_data)
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_loss)
    data.append(test_loss)
    data.append(per_epoch_time)

    return data


def export_data(dataset, feature_input):

    for i in range(feature_input):
        filename = "../Out/3_" + str(feature_input - 1) + "_drop_" + str(i) + ".csv"
        with open(filename, "w") as f:
            f.write("iter,tr-loss,te-loss,time\n")
            for j in range(0, epochs):
                f.write(
                    "%s,%s,%s,%s\n"
                    % (str(i), dataset[i][0][j], dataset[i][1][j], dataset[i][2][j])
                )

    colors = [
        "#FF0000",
        "#0000FF",
        "#008000",
        "#FFA500",
        "#800080",
        "#FFC0CB",
        "#00FFFF",
    ]

    fig1 = plt.figure(figsize=(16, 8))
    for i in range(feature_input):
        plt.plot(
            range(epochs),
            dataset[i][0],
            label="Loss w\\out Feature " + str(i),
            color=colors[i],
        )
    plt.xlabel(str(epochs) + " iterations")
    plt.ylabel("Train Loss (" + str(feature_input - 1) + ")")
    plt.legend()
    plt.ylim(0.002, 0.02)
    fig1.savefig("../Out/3_" + str(feature_input - 1) + "_feature_train_loss.png")
    plt.close()

    fig1 = plt.figure(figsize=(16, 8))
    for i in range(feature_input):
        plt.plot(
            range(epochs),
            dataset[i][1],
            label="Loss w\\out Feature " + str(i),
            color=colors[i],
        )
    plt.xlabel(str(epochs) + " iterations")
    plt.ylabel("Test Loss (" + str(feature_input - 1) + ")")
    plt.legend()
    plt.ylim(0.002, 0.02)
    fig1.savefig("../Out/3_" + str(feature_input - 1) + "_feature_test_loss.png")
    plt.close()


def main():
    # setup multiprocessing
    num_threads = mp.cpu_count() - 2
    p = mp.Pool(processes=num_threads)

    # process data
    file_train = "../Data/train_data.csv"
    file_test = "../Data/test_data.csv"

    # train data contains 2 (x and y) X 280 X 7/1 (x or y inputs)
    train_data = process_data(file_train)
    test_data = process_data(file_test)

    # drop one feature
    dropped_train_dataset, dropped_test_dataset = [], []
    for i in range(FEATURE_INPUT[0]):
        dropped_train_dataset.append(process_drop_feature(train_data, i))
        dropped_test_dataset.append(process_drop_feature(test_data, i))

    # zipping dataset
    zipped_feature = []
    for i in range(FEATURE_INPUT[0]):
        zipped_feature.append(FEATURE_INPUT[0] - 1)

    # execute RFE on 7 features
    dataset = p.starmap(
        nn_model, zip(dropped_train_dataset, dropped_test_dataset, zipped_feature)
    )

    export_data(dataset, FEATURE_INPUT[0])

    # process data
    # remove column 6: Research
    dropped_one_train = process_drop_feature(train_data, 6)
    dropped_one_test = process_drop_feature(test_data, 6)

    # drop one feature
    dropped_train_dataset, dropped_test_dataset = [], []
    for i in range(FEATURE_INPUT[1]):
        dropped_train_dataset.append(process_drop_feature(dropped_one_train, i))
        dropped_test_dataset.append(process_drop_feature(dropped_one_test, i))

    # zipping dataset
    zipped_feature = []
    for i in range(FEATURE_INPUT[1]):
        zipped_feature.append(FEATURE_INPUT[1] - 1)

    # execute RFE on 7 features
    dataset = p.starmap(
        nn_model, zip(dropped_train_dataset, dropped_test_dataset, zipped_feature)
    )

    export_data(dataset, FEATURE_INPUT[1])


if __name__ == "__main__":
    main()
