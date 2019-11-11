import matplotlib.pylab as plt
import numpy as np
import pickle
from sklearn.utils import gen_batches
import tensorflow as tf
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = [3, 50, 60]
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


def cnn(images, activation_type):
    # layer 1 (input): size * size * channels (32 * 32 * 3)
    images = tf.reshape(images, [-1, NUM_CHANNELS[0], IMG_SIZE, IMG_SIZE])
    images = tf.transpose(images, [0, 2, 3, 1])

    # layer 2: cnn 1
    two_weights = tf.Variable(
        tf.truncated_normal(
            [9, 9, NUM_CHANNELS[0], NUM_CHANNELS[1]],
            stddev=1.0 / np.sqrt(NUM_CHANNELS[0] * 9 * 9),
        ),
        name="two_weights",
    )
    two_bias = tf.Variable(tf.zeros(NUM_CHANNELS[1]), name="two_bias")

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
            [5, 5, NUM_CHANNELS[1], NUM_CHANNELS[2]],
            stddev=1.0 / np.sqrt(NUM_CHANNELS[1] * 5 * 5),
        ),
        name="three_weights",
    )
    three_bias = tf.Variable(tf.zeros(NUM_CHANNELS[2]), name="three_bias")

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

    four_calc = tf.matmul(four_pool_flat, four_weights) + four_bias

    if activation_type == 4:
        four_output = tf.nn.tanh(four_calc)
    elif activation_type == 3:
        four_output = tf.nn.relu6(four_calc)
    elif activation_type == 2:
        four_output = tf.nn.relu(four_calc)
    elif activation_type == 1:
        four_output = tf.nn.sigmoid(four_calc)
    else:
        four_output = four_calc

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


def plot_feature_maps(layer, num_features, filename):
    plt.figure()
    plt.gray()
    layer_array = np.array(layer)
    for i in range(num_features):
        plt.subplot(num_features / 10, 10, i + 1)
        plt.axis("off")
        plt.imshow(layer_array[0, :, :, i])
    plt.savefig("../Out/" + filename + ".png")
    plt.close()


def graph(train_data, test_data, activation_type, patterns):
    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS[0]])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    logits, w2, b2, w3, b3, w4, b4, w5, b5, c2, p2, c3, p3 = cnn(x, activation_type)

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

            # early stopping
            if current_test_acc > highest_out[3][0]:
                highest_out[0][0] = e
                highest_out[1][0] = current_train_acc
                highest_out[2][0] = current_train_loss[0]
                highest_out[3][0] = current_test_acc
                w2_, b2_, w3_, b3_, w4_, b4_, w5_, b5_, = sess.run(
                    [w2, b2, w3, b3, w4, b4, w5, b5]
                )

        # evaluate with best train accuracy epoch
        w2.load(w2_, sess)
        b2.load(b2_, sess)
        w3.load(w3_, sess)
        b3.load(b3_, sess)
        w4.load(w4_, sess)
        b4.load(b4_, sess)
        w5.load(w5_, sess)
        b5.load(b5_, sess)

        for i in range(len(patterns)):
            two_conv_, two_pool_, three_conv_, three_pool_ = sess.run(
                [c2, p2, c3, p3], {x: patterns[i]}
            )

            # export feature map
            plot_feature_maps(
                two_conv_,
                NUM_CHANNELS[1],
                "1_" + str(activation_type) + "_" + str(i) + "_two_conv_fm",
            )
            plot_feature_maps(
                two_pool_,
                NUM_CHANNELS[1],
                "1_" + str(activation_type) + "_" + str(i) + "_two_pool_fm",
            )
            plot_feature_maps(
                three_conv_,
                NUM_CHANNELS[2],
                "1_" + str(activation_type) + "_" + str(i) + "_three_conv_fm",
            )
            plot_feature_maps(
                three_pool_,
                NUM_CHANNELS[2],
                "1_" + str(activation_type) + "_" + str(i) + "_three_pool_fm",
            )

    return out, highest_out


def export_to_file(data, filename):
    with open("../Out/1_" + filename + "_output.csv", "w") as f:
        f.write("epoch,train_acc,train_cost,test_acc\n")
        for i in range(len(data[0])):
            f.write(str(data[0][i]) + ",")
            f.write(str(data[1][i]) + ",")
            f.write(str(data[2][i]) + ",")
            f.write(str(data[3][i]) + "\n")


def import_data_export_graph(filenumber):
    data_epoch = []
    data_train_acc = []
    data_train_cost = []
    data_test_acc = []

    with open("../Out/1_" + str(filenumber) + "_output.csv", "r") as f:
        data = f.read().split("\n")[1:-1]
        for i in range(len(data)):
            data[i] = data[i].split(",")
            for j in range(len(data[i])):
                if j == 0:
                    data_epoch.append(int(data[i][j]))
                elif j == 1:
                    data_train_acc.append(float(data[i][j]))
                elif j == 2:
                    data_train_cost.append(float(data[i][j]))
                else:
                    data_test_acc.append(float(data[i][j]))

    fig, ax = plt.subplots(figsize=(16, 8))
    title = ""
    if filenumber == 0:
        title = "Linear"
    elif filenumber == 1:
        title = "Sigmoid"
    elif filenumber == 2:
        title = "ReLu"
    elif filenumber == 3:
        title = "ReLu6"
    else:
        title = "Hyperbolic Tangent"

    ax.set_title(title)
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Epochs")

    ax.plot(data_epoch, data_test_acc, label="Test Accuracy", color="#0000FF")
    # for i, value in enumerate(max_data[3]):
    #     ax.text(i, value, str(value), ha="center", va="bottom")
    ax.legend()
    fig.savefig(
        "../Out/1_" + str(filenumber) + "_plot.png",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()


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

    # randomise patterns for test
    patterns = []
    indexes = []
    for i in range(2):
        current_index = np.random.randint(low=0, high=len(test_data[0]))
        indexes.append(current_index)
        original = test_data[0][current_index]
        patterns.append(np.reshape(original, [1, -1]))

        # export picture
        plt.figure()
        plt.gray()
        original_show = original.reshape(NUM_CHANNELS[0], IMG_SIZE, IMG_SIZE).transpose(
            1, 2, 0
        )
        plt.axis("off")
        plt.imshow(original_show)
        plt.savefig("../Out/1_pattern_" + str(i) + ".png")
        plt.close()

    with open("../Out/1_indexes.csv", "w") as f:
        for index in indexes:
            f.write(str(index) + "\n")

    num_activations = 5
    all_out = []
    all_highest_acc = []
    # activation functions
    # 0 - Linear
    # 1 - Sigmoid
    # 2 - ReLu
    # 3 - ReLu6
    # 4 - Hyperbolic Tangent
    for i in range(num_activations):
        raw_out, raw_high = graph(train_data, test_data, i, patterns)
        all_out.append(raw_out)
        all_highest_acc.append(raw_high)

    # export data set
    for i in range(len(all_out)):
        export_to_file(all_out[i], str(i))
        export_to_file(all_highest_acc[i], str(i) + "_max")

    for i in range(num_activations):
        import_data_export_graph(i)

    # plot max graph
    max_data = [[], [], [], []]
    for i in range(num_activations):
        with open("../Out/1_" + str(i) + "_max_output.csv", "r") as f:
            lines = f.read().split("\n")[1:-1]
            for j in range(len(lines)):
                line = lines[j].split(",")
            for i in range(len(line)):
                max_data[i].append(float(line[i]))

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title("Performance Comparison")

    ax.set_ylabel("Test Accuracy")

    labels = ["Linear", "Sigmoid", "ReLu", "ReLu6", "Hyperbolic Tangent"]

    ax.set_xlim([-1, num_activations])
    ax.set_xticks(np.arange(num_activations))
    ax.set_xticklabels(labels)
    ax.set_xlabel("\nActivation Functions\n")
    ax.plot(
        np.arange(num_activations), max_data[3], label="Test Accuracy", color="#0000FF"
    )
    # for i, value in enumerate(max_data[3]):
    #     ax.text(i, value, str(value), ha="center", va="bottom")
    ax.legend()
    fig.savefig("../Out/1_max_test_acc.png", bbox_inches="tight", pad_inches=0.05)
    plt.close()


if __name__ == "__main__":
    main()
