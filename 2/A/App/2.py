import matplotlib.pylab as plt
import numpy as np
import pickle
from sklearn.utils import gen_batches
import tensorflow as tf
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_CLASSES = 10
IMG_SIZE = 32
CHANNELS = [[3], [40, 50, 60, 70], [50, 60, 70, 80]]
learning_rate = 0.001
epochs = 2000
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

num_sample = 2


def load_data(file):
    # parameters
    processed_data = []

    with open(file, "rb") as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding="latin1")

    data, labels = samples["data"], samples["labels"]

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1

    processed_data.append(data)
    processed_data.append(labels_)

    return processed_data


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


def cnn(images, two_channel, three_channel):
    # layer 1 (input): size * size * channels (32 * 32 * 3)
    images = tf.reshape(images, [-1, CHANNELS[0][0], IMG_SIZE, IMG_SIZE])
    images = tf.transpose(images, [0, 2, 3, 1])

    # layer 2: cnn 1
    two_weights = tf.Variable(
        tf.truncated_normal(
            [9, 9, CHANNELS[0][0], two_channel],
            stddev=1.0 / np.sqrt(CHANNELS[0][0] * 9 * 9),
        ),
        name="two_weights",
    )
    two_bias = tf.Variable(tf.zeros(two_channel), name="two_bias")

    two_conv = tf.nn.relu(
        tf.nn.conv2d(images, two_weights, [1, 1, 1, 1], padding="VALID") + two_bias
    )
    two_pool = tf.nn.max_pool(
        two_conv,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        name="two_pool",
    )

    # layer 3: cnn 2
    three_weights = tf.Variable(
        tf.truncated_normal(
            [5, 5, two_channel, three_channel],
            stddev=1.0 / np.sqrt(two_channel * 5 * 5),
        ),
        name="three_weights",
    )
    three_bias = tf.Variable(tf.zeros(three_channel), name="three_bias")

    three_conv = tf.nn.relu(
        tf.nn.conv2d(two_pool, three_weights, [1, 1, 1, 1], padding="VALID")
        + three_bias
    )
    three_pool = tf.nn.max_pool(
        three_conv,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        name="three_pool",
    )

    three_sum = (
        three_pool.get_shape()[1].value
        * three_pool.get_shape()[2].value
        * three_pool.get_shape()[3].value
    )

    # layer 4: fully connected
    four_weights = tf.Variable(
        tf.truncated_normal(
            [three_sum, 300], stddev=1.0 / np.sqrt(three_sum), name="four_weights"
        )
    )
    four_bias = tf.Variable(tf.zeros(300), name="four_bias")

    four_pool_flat = tf.reshape(three_pool, [-1, three_sum])
    # linear activation function from 1
    four_output = tf.matmul(four_pool_flat, four_weights) + four_bias

    # layer 5: output (softmax)
    five_weights = tf.Variable(
        tf.truncated_normal([300, 10], stddev=1.0 / np.sqrt(300)), name="five_weights"
    )
    five_bias = tf.Variable(tf.zeros([10]), name="five_bias")
    logits = tf.matmul(four_output, five_weights) + five_bias

    return (
        logits,
        two_weights,
        two_bias,
        three_weights,
        three_bias,
        four_weights,
        four_bias,
        five_weights,
        five_bias,
        two_conv,
        two_pool,
        three_conv,
        three_pool,
    )


def graph(train_data, test_data, channels):

    two_channel = channels[0]
    three_channel = channels[1]

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * CHANNELS[0][0]])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    logits, w2, b2, w3, b3, w4, b4, w5, b5, c2, p2, c3, p3 = cnn(
        x, two_channel, three_channel
    )

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.cast(
        tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32
    )
    accuracy = tf.reduce_mean(correct_prediction)

    # parameters for sess
    N = len(train_data[0])
    idx = np.arange(N)
    out = [[], [], [], []]
    highest_out = [[0], [0], [0], [0]]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in tqdm(range(epochs)):
            # randomise dataset order
            np.random.shuffle(idx)
            train_data[0], train_data[1] = train_data[0][idx], train_data[1][idx]

            # batching
            batched_data = process_data_batch(train_data)

            # train
            for data in batched_data:
                train_step.run(feed_dict={x: data[0], y_: data[1]})

            # evaluate
            current_train_acc = accuracy.eval(
                feed_dict={x: train_data[0], y_: train_data[1]}
            )
            current_train_loss = sess.run([loss], {x: train_data[0], y_: train_data[1]})
            current_test_acc = accuracy.eval(
                feed_dict={x: test_data[0], y_: test_data[1]}
            )

            out[0].append(e)
            out[1].append(current_train_acc)
            out[2].append(current_train_loss[0])
            out[3].append(current_test_acc)

            # higest accuracy
            if current_test_acc > highest_out[3][0]:
                highest_out[0][0] = e
                highest_out[1][0] = current_train_acc
                highest_out[2][0] = current_train_loss[0]
                highest_out[3][0] = current_test_acc

    return out, highest_out


def export_to_file(data, filename):
    with open("../Out/2_" + filename + "_output.csv", "w") as f:
        f.write("epoch,train_acc,train_cost,test_acc\n")
        for i in range(len(data[0])):
            f.write(str(data[0][i]) + ",")
            f.write(str(data[1][i]) + ",")
            f.write(str(data[2][i]) + ",")
            f.write(str(data[3][i]) + "\n")


def main():
    train_data = load_data("../Data/data_batch_1")
    # print(trainX.shape, trainY.shape)

    test_data = load_data("../Data/test_batch_trim")
    # print(testX.shape, testY.shape)

    # scale data
    train_data[0] = (train_data[0] - np.min(train_data[0], axis=0)) / np.max(
        train_data[0], axis=0
    )
    test_data[0] = (test_data[0] - np.min(test_data[0], axis=0)) / np.max(
        test_data[0], axis=0
    )

    all_out = []
    all_highest_acc = []
    for ch2 in CHANNELS[1]:  # conv/pool layer 2
        for ch3 in CHANNELS[2]:  # conv/pool layer 3
            raw_out, raw_high = graph(train_data, test_data, [ch2, ch3])
            all_out.append(raw_out)
            all_highest_acc.append(raw_high)

    # export data set
    for i in range(len(all_out)):
        export_to_file(all_out[i], str(i))
        export_to_file(all_highest_acc[i], str(i) + "_max")

    # read exported csv data
    num_params = len(CHANNELS[1]) * len(CHANNELS[2])

    max_data = [[], [], [], []]
    for i in range(num_params):
        with open("../Out/2_" + str(i) + "_max_output.csv", "r") as f:
            lines = f.read().split("\n")[1:-1]
            for j in range(len(lines)):
                line = lines[j].split(",")
            for i in range(len(line)):
                max_data[i].append(float(line[i]))

    # plot max graph
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title("Performance Comparison")

    ax.set_ylabel("Test Accuracy")

    labels = []
    for ch2 in CHANNELS[1]:  # conv/pool layer 2
        for ch3 in CHANNELS[2]:  # conv/pool layer 3
            labels.append(str(ch2) + " | " + str(ch3))

    ax.set_xlim([-1, num_params + 1])
    ax.set_xticks(np.arange(num_params))
    ax.set_xticklabels(labels)
    ax.set_xlabel("\nFM Conv/Pool 1 | FM Conv Pool 2\n")
    ax.plot(np.arange(num_params), max_data[3], label="Test Accuracy", color="#0000FF")
    # for i, value in enumerate(max_data[3]):
    #     ax.text(i, value, str(value), ha="center", va="bottom")
    ax.legend()
    fig.savefig("../Out/2_max_test_acc.png", bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # plot correlation matrix
    corr_max_test_acc = []

    for i in range(len(CHANNELS[1])):
        row_data = []
        for j in range(len(CHANNELS[2])):
            row_data.append(max_data[3][(i * len(CHANNELS[1])) + j])
        corr_max_test_acc.append(row_data)

    corr_max_test_acc = np.array(corr_max_test_acc)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_yticks(np.arange(len(CHANNELS[1])))
    ax.set_yticklabels(CHANNELS[1])
    ax.set_ylabel("\nConvolution / Pool Layer 1\nFeature Maps\n")

    ax.set_xticks(np.arange(len(CHANNELS[2])))
    ax.set_xticklabels(CHANNELS[2])
    ax.set_xlabel("\nConvolution / Pool Layer 2\nFeature Maps\n")
    im = ax.imshow(corr_max_test_acc, cmap="Oranges")
    fig.colorbar(im)

    # text
    for x in range(len(CHANNELS[1])):
        for y in range(len(CHANNELS[2])):
            text = str(corr_max_test_acc[x][y])
            ax.text(y, x, text, va="center", ha="center")

    fig.savefig("../Out/2_correlation.png", bbox_inches="tight", pad_inches=0.05)
    plt.close()


if __name__ == "__main__":
    main()
