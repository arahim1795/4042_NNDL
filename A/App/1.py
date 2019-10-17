import math
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from sklearn.utils import gen_batches
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# hyper-parameters
np.random.seed(10)

FEATURE_INPUT = 21  # input: LB to Tendency
NUM_CLASSES = 3  # NSP = 1, 2, 3

batch = 32
decay = math.pow(10, -6)
epochs = 100000
learning_rate = 0.01
num_neurons = 10


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


def nn_model(train_data, test_data):
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
            # start record time
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

            # randomise dataset
            train_data = randomise_order(train_data)

            # end record time
            per_epoch_time.append(time.time() - start_time)

    data = []
    data.append(train_acc)
    data.append(test_acc)
    data.append(per_epoch_time)

    return data


def export_data(data, file):
    # export values
    with open(file, "w") as f:
        f.write("epoch,train accuracy,test accuracy,time per epoch\n")
        for i in range(0, epochs):
            f.write(
                "%s,%s,%s,%s\n"
                % (str(i), str(data[0][i]), str(data[1][i]), str(data[2][i]))
            )


def roundup_hundreds(value):
    return int(math.ceil(value / 100.00)) * 100


def extract_data(file):
    # find max test accuracy (i.e. test error convergence)
    raw_data = np.genfromtxt(file, delimiter=",")

    # remove header
    clean_data = np.array(raw_data[1:])

    # get max test index and value
    max_test_idx = clean_data.argmax(axis=0)[2]
    with open("../Out/csv/1_max.csv", "w") as f:
        f.write("epoch,train accuracy,test_accuracy\n")
        entry = clean_data[max_test_idx]
        f.write("%s,%s,%s\n" % (entry[0], entry[1], entry[2]))

    # process data
    train_data = np.delete(clean_data, [0, 2, 3], axis=1)
    test_data = np.delete(clean_data, [0, 1, 3], axis=1)

    # plot to nearest hundreds
    rounded_epoch = roundup_hundreds(max_test_idx)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.plot(
        range(rounded_epoch),
        train_data[0:rounded_epoch],
        label="Train Accuracy",
        color="#0000FF",
    )
    ax.plot(
        range(rounded_epoch),
        test_data[0:rounded_epoch],
        label="Test Accuracy",
        color="#FF0000",
    )
    ax.legend()
    fig.savefig("../Out/graph/1_epoch_" + str(rounded_epoch) + ".png")
    plt.close()

    # plot to max epoch
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.plot(range(epochs), train_data, label="Train Accuracy", color="#0000FF")
    ax.plot(range(epochs), test_data, label="Test Accuracy", color="#FF0000")
    ax.legend()
    fig.savefig("../Out/graph/1_epoch_" + str(epochs) + ".png")
    plt.close()


def main():
    # process data
    file_train = "../Data/train_data.csv"
    file_test = "../Data/test_data.csv"

    train_data = process_data(file_train)
    test_data = process_data(file_test)

    # execute model
    dataset = nn_model(train_data, test_data)

    file = "../Out/csv/1.csv"
    export_data(dataset, file)
    extract_data(file)


if __name__ == "__main__":
    main()
