import math
import matplotlib.pylab as plt
import numpy as np
from sklearn.utils import gen_batches
import tensorflow as tf
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# hyper-parameters
np.random.seed(291)

FEATURE_INPUT = 7

learning_rate = math.pow(10, -3)
epochs = 50000
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

    train_loss, test_loss, per_epoch_time = [], [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            # batching
            batched_data = process_data_batch(train_data, batch_size)

            # train
            for data in batched_data:
                train_op.run(feed_dict={x: data[0], y_: data[1]})

            # test
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

        # predictions
        prediction_set = sess.run(logits, feed_dict={x: predict_train_data[0]})

    dataset = []
    dataset.append(train_loss)
    dataset.append(test_loss)
    dataset.append(per_epoch_time)
    dataset.append(prediction_set)

    return dataset


def export_data(dataset):
    # export accuracies
    filename = "../Out/1.csv"
    with open(filename, "w") as f:
        f.write("iter,tr-acc,te-acc,time\n")
        for i in range(0, epochs, 250):
            f.write(
                "%s,%s,%s,%s\n"
                % (str(i), str(dataset[0][i]), str(dataset[1][i]), str(dataset[2][i]))
            )

    fig1 = plt.figure(figsize=(16, 8))
    plt.plot(range(epochs), dataset[0], label="Train Loss")
    plt.plot(range(epochs), dataset[1], label="Test Loss")
    plt.xlabel(str(epochs) + " iterations")
    plt.ylabel("Train/Test Loss")
    plt.ylim(0, 0.03)
    plt.legend()
    fig1.savefig("../Out/1_loss.png")


def export_prediction_data(predicted_dataset, actual_dataset):
    # sort in ascending actual observation
    arr_sorter = np.argsort(actual_dataset, axis=0)
    sorted_actual = actual_dataset[arr_sorter]
    sorted_predicted = predicted_dataset[arr_sorter]

    fig2 = plt.figure(figsize=(16, 8))
    plt.scatter(range(50), sorted_predicted, label="prediction", color="#ff0000")
    plt.scatter(range(50), sorted_actual, label="actual", color="#00ffff")
    plt.xlabel("Predicition Number")
    plt.ylabel("Admission chance")
    plt.legend()
    fig2.savefig("../Out/1_predictions.png")


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
    export_prediction_data(dataset[3], sampled_data[1])


if __name__ == "__main__":
    main()
