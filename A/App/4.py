import math
import tensorflow as tf
import matplotlib.pylab as plt
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import gen_batches
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# hyper-parameters
np.random.seed(291)

FEATURE_INPUT = 21  # input: LB to Tendency
NUM_CLASSES = 3  # NSP = 1, 2, 3

batch = 4
decay = [0, math.pow(10, -3), math.pow(10, -6), math.pow(10, -9), math.pow(10, -12)]
decay_exp = [0, 3, 6, 9, 12]
epochs = 20000
k_value = 5
learning_rate = 0.01
num_neurons = 25


# functions
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


def process_data_k(data):
    # parameters
    permutated_train, permutated_test = [], []
    train_indexes, test_indexes = [], []

    # create index splitter
    kf = KFold(n_splits=k_value)

    # get splitted index
    for train, test in kf.split(data[0]):
        train_indexes.append(train)
        test_indexes.append(test)

    # append processed data to list
    for a in range(k_value):
        train_data, test_data = [], []
        train_data.append(data[0][train_indexes[a]])
        train_data.append(data[1][train_indexes[a]])
        test_data.append(data[0][test_indexes[a]])
        test_data.append(data[1][test_indexes[a]])
        permutated_train.append(train_data)
        permutated_test.append(test_data)

    return permutated_train, permutated_test


def process_data_batch(data):
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
    np.random.shuffle(indexes)

    # randomise dataset order
    dataset[0] = dataset[0][indexes]
    dataset[1] = dataset[1][indexes]

    return dataset


def nn_model(train_data, test_data, decay):
    # create model
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

    # run model
    train_acc, test_acc, per_epoch_time = [], [], []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            # batching
            batched_data = process_data_batch(train_data)

            # train
            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            # evaluate
            train_acc.append(
                accuracy.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            )
            test_acc.append(
                accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1]})
            )

            if (i % 1000) == 0:
                print("epoch: ", i, " tr-acc: ", train_acc[i], "te-acc: ", test_acc[i])

            # randomise dataset
            train_data = randomise_order(train_data)
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_acc)
    data.append(test_acc)
    data.append(per_epoch_time)

    return data


def export_data(data):
    # export accuracies
    with open("../Out/4_optimal.csv", "w") as f:
        f.write("iter,tr-acc,te-acc,time\n")
        for j in range(0, epochs):
            f.write("%s,%s,%s,%s\n" % (str(j), data[0][j], data[1][j], data[2][j]))

    # plotting
    plt.figure(figsize=(16, 8))
    fig = plt.figure(1)
    plt.plot(range(epochs), data[0], label="Train Accuracy", color="#ff0000")
    plt.plot(range(epochs), data[1], label="Test Accuracy", color="#00ffff")
    plt.xlabel(str(epochs) + " iterations")
    plt.ylabel("Train/Test accuracy")
    plt.legend()
    fig.savefig("../Out/4_optimal_fig.png")
    plt.close()


def export_data_batch(data, decay_exp):
    for i in range(0, len(data), 5):
        #  mean cross-validation
        mean_train = (
            np.array(data[i][0])
            + np.array(data[i + 1][0])
            + np.array(data[i + 2][0])
            + np.array(data[i + 3][0])
            + np.array(data[i + 4][0])
        )

        mean_test = (
            np.array(data[i][1])
            + np.array(data[i + 1][1])
            + np.array(data[i + 2][1])
            + np.array(data[i + 3][1])
            + np.array(data[i + 4][1])
        )

        mean_time = (
            np.array(data[i][2])
            + np.array(data[i + 1][2])
            + np.array(data[i + 2][2])
            + np.array(data[i + 3][2])
            + np.array(data[i + 4][2])
        )

        mean_train = np.divide(mean_train, 5.0)
        mean_test = np.divide(mean_test, 5.0)
        mean_time = np.divide(mean_time, 5.0)

        # export data
        with open("../Out/4_d" + str(decay_exp[i]) + ".csv", "w") as f:
            f.write("iter,tr-acc,te-acc,time\n")
            for j in range(0, epochs):
                f.write(
                    "%s,%s,%s,%s\n"
                    % (str(j), mean_train[j], mean_test[j], mean_time[j])
                )

        # plotting
        fig = plt.figure(1, figsize=(16, 8))
        plt.plot(range(epochs), mean_train, label="mean_train", color="#ff0000")
        plt.plot(range(epochs), mean_test, label="mean_test", color="#00ffff")
        plt.xlabel(str(epochs) + " iterations")
        plt.ylabel("Train/Test accuracy")
        plt.legend()
        fig.savefig("../Out/4_d" + str(decay_exp[i]) + ".png")
        plt.close()


def extract_useful_data(file_1, file_2, file_3, file_4, file_5):
    # import raw data
    raw_data_1 = np.genfromtxt(file_1, delimiter=",")[1:]
    raw_data_2 = np.genfromtxt(file_2, delimiter=",")[1:]
    raw_data_3 = np.genfromtxt(file_3, delimiter=",")[1:]
    raw_data_4 = np.genfromtxt(file_4, delimiter=",")[1:]
    raw_data_5 = np.genfromtxt(file_5, delimiter=",")[1:]
    # print(raw_data_1[0])

    # get test accs
    test_acc = []
    test_acc.append(np.delete(raw_data_1, [0, 1, 3], axis=1).max())
    test_acc.append(np.delete(raw_data_2, [0, 1, 3], axis=1).max())
    test_acc.append(np.delete(raw_data_3, [0, 1, 3], axis=1).max())
    test_acc.append(np.delete(raw_data_4, [0, 1, 3], axis=1).max())
    test_acc.append(np.delete(raw_data_5, [0, 1, 3], axis=1).max())
    test_acc = np.array(test_acc)

    filename = "../Out/4_max_mean_test.csv"
    with open(filename, "w") as f:
        f.write("batch,mean test\n")
        for i in range(len(test_acc)):
            f.write("%s,%s\n" % (str(decay[i]), test_acc[i]))

    fig = plt.figure(figsize=(16, 8))
    plt.plot(decay_exp, test_acc, label="max_acc", color="#ff0000")
    plt.xticks(decay_exp)
    plt.legend()
    fig.savefig("../Out/4_max_mean_test.png")
    plt.close()


def main():
    # # process data
    file_train = "../Data/train_data.csv"
    file_test = "../Data/test_data.csv"

    train_data = process_data(file_train)
    test_data = process_data(file_test)

    k_train, k_test = process_data_k(train_data)

    # setup multiprocessing
    num_threads = mp.cpu_count() - 1
    p = mp.Pool(processes=num_threads)

    # zipping dataset
    zipped_decay, zipped_decay_exp = [], []
    zipped_k_train, zipped_k_test = [], []
    for i in range(len(decay)):
        for j in range(k_value):
            zipped_decay.append(decay[i])
            zipped_decay_exp.append(decay_exp[i])
            zipped_k_train.append(k_train[j])
            zipped_k_test.append(k_test[j])

    # # execute k-fold
    # dataset = p.starmap(nn_model, zip(zipped_k_train, zipped_k_test, zipped_decay))

    # # export data meaningfully
    # export_data_batch(dataset, zipped_decay_exp)

    file_1 = "../Out/4_d0.csv"
    file_2 = "../Out/4_d3.csv"
    file_3 = "../Out/4_d6.csv"
    file_4 = "../Out/4_d9.csv"
    file_5 = "../Out/4_d12.csv"
    extract_useful_data(file_1, file_2, file_3, file_4, file_5)

    # optimal decay (decay = 10^-3)
    optimal_decay = math.pow(10, -3)
    optimal_dataset = nn_model(train_data, test_data, optimal_decay)

    export_data(optimal_dataset)


if __name__ == "__main__":
    main()
