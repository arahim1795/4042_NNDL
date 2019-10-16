import math
import matplotlib.pylab as plt
import numpy as np
from sklearn.utils import gen_batches
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# hyper-parameters
np.random.seed(291)

FEATURE_INPUT = 7

learning_rate = math.pow(10, -3)
epochs = 10000
num_neurons = 10
batch_size = 8
seed = 10
np.random.seed(seed)
decay = math.pow(10, -3)


# function
def scale(data):
    data_scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data_scaled


def process_data(file):
    data = np.genfromtxt(file, delimiter=",")

    X, Y = data[:, 1:8], data[:, -1]
    X = scale(X)
    Y = Y.reshape(Y.shape[0], 1)

    data = []
    data.append(X)
    data.append(Y)

    return data


def randomise_data(data):
    idx = np.arange(len(data[0]))
    np.random.shuffle(idx)

    data[0] = data[0][idx]
    data[1] = data[1][idx]

    return data


def data_sampling(data):
    output_data = []

    # randomise
    random_data = randomise_data(data)

    output_data.append(random_data[0][:50])
    output_data.append(random_data[1][:50])

    return output_data


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


def nn_model(train_data, test_data, predict_train_data):
    x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # layer 1: relu
    layer_1_weights = tf.Variable(
        tf.truncated_normal(
            [FEATURE_INPUT, num_neurons], stddev=1.0 / math.sqrt(float(FEATURE_INPUT))
        ),
        name="one_weights",
    )
    layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
    layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

    layer_1_output = tf.nn.relu(layer_1_var)

    # layer 2: linear
    layer_final_weights = tf.Variable(
        tf.truncated_normal(
            [num_neurons, 1], stddev=1.0 / math.sqrt(float(num_neurons))
        ),
        name="final_weights",
    )
    layer_final_biases = tf.Variable(tf.zeros([1]), name="final_biases")
    logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases

    # Regularisation (L2)
    regularization = tf.nn.l2_loss(layer_1_weights) + tf.nn.l2_loss(layer_final_weights)
    loss = tf.reduce_mean(tf.square(y_ - logits))
    l2_loss = tf.reduce_mean(loss + decay * regularization)

    # Minimising Loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(l2_loss)

    dataset, prediction_set, errors = [], [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_error_set, test_error_set = [], []

        for i in range(epochs):
            # batching
            batched_data = process_data_batch(train_data, batch_size)

            # train
            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            # test
            train_error = l2_loss.eval(feed_dict={x: train_data[0], y_: train_data[1]})
            test_error = l2_loss.eval(feed_dict={x: test_data[0], y_: test_data[1]})

            train_error_set.append(train_error)
            test_error_set.append(test_error)

            # randomise
            train_data = randomise_data(train_data)

        # predictions
        prediction_set = sess.run(logits, feed_dict={x: predict_train_data[0]})

    errors.append(train_error_set)
    errors.append(test_error_set)

    dataset.append(prediction_set)
    dataset.append(predict_train_data[1])
    dataset.append(errors)

    return dataset


def export_data(dataset):
    # export accuracies
    filename = "../Out/1-accuracy.csv"
    with open(filename, "w") as f:
        f.write("iter,tr-acc,te-acc\n")
        for i in range(0, epochs, 250):
            f.write(
                "%s,%s,%s\n" % (str(i), str(dataset[2][0][i]), str(dataset[2][1][i]))
            )

    fig1 = plt.figure(1)
    plt.plot(range(epochs), dataset[2][0], label="Train Loss")
    plt.plot(range(epochs), dataset[2][1], label="Test Loss")
    plt.xlabel(str(epochs) + " iterations")
    plt.ylabel("Train/Test Loss")
    plt.ylim(0, 0.03)
    plt.legend()
    fig1.savefig("../Out/1_a.png")

    fig2 = plt.figure(2)
    plt.scatter(range(50), dataset[0][0:50])
    plt.plot(range(50), dataset[0][0:50], label="prediction")
    plt.scatter(range(50), dataset[1][0:50])
    plt.plot(range(50), dataset[1][0:50], label="actual")
    plt.xlabel("Predicition Number")
    plt.ylabel("Admission chance")
    plt.legend()
    fig2.savefig("../Out/1_b.png")


def main():
    # process data
    file_train = "../Data/train_data.csv"
    file_test = "../Data/test_data.csv"

    train_data = process_data(file_train)
    test_data = process_data(file_test)

    sampled_data = data_sampling(train_data)

    # execute model
    dataset = nn_model(train_data, test_data, sampled_data)

    export_data(dataset)


if __name__ == "__main__":
    main()
