import math
import matplotlib.pylab as plt
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# * Parameters
FEATURE_INPUT = 21  # input: LB to Tendency
NUM_CLASSES = 3  # NSP = 1, 2, 3

learning_rate = 0.01
epochs = 100000
num_neurons = 10
seed = 10
np.random.seed(seed)
decay = math.pow(10, -6)


# * Data Handling
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


def process_data(file):
    # parameters
    processed_data = []

    # import data
    raw_data = np.genfromtxt(file, delimiter=',')

    # process X
    X = raw_data[:, :FEATURE_INPUT]
    X = scale(X, np.min(X, axis=0), np.max(X, axis=0))

    # process Y
    Y = raw_data[:, -1].astype(int)
    Y_one_hot = np.zeros((Y.shape[0], NUM_CLASSES))
    Y_one_hot[np.arange(Y.shape[0]), Y-1] = 1

    # append processed data to list
    processed_data.append(X)
    processed_data.append(Y_one_hot)

    return processed_data


# import train data
file = '../Data/train_data.csv'
train_data = process_data(file)

# import test data
file = '../Data/test_data.csv'
test_data = process_data(file)

# * Create Model
x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# hidden layer 1: relu
layer_1_weights = tf.Variable(tf.truncated_normal(
    [FEATURE_INPUT, num_neurons], stddev=1.0/math.sqrt(float(FEATURE_INPUT))), name='one_weights')
layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name='one_biases')
layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

layer_1_output = tf.nn.relu(layer_1_var)

# final layer: softmax
layer_final_weights = tf.Variable(tf.truncated_normal(
    [num_neurons, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_neurons))), name='final_weights')
layer_final_biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='final_biases')
logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases

# regularise (l2)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y_, logits=logits)
loss = tf.reduce_mean(cross_entropy)

loss = tf.reduce_mean(loss + (decay*tf.nn.l2_loss(logits)))

# minimise loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

train_acc_set, test_acc_set = [], []


# * Execution
def randomise_dataset(dataset):
    # parameters
    randomised_data = []

    # column-wise combine data
    combined_data = np.concatenate((dataset[0], dataset[1]), axis=1)
    
    # randomise data
    random.seed(291)
    random.shuffle(combined_data)

    # append split randomised data to list
    randomised_data.append(combined_data[:, :FEATURE_INPUT])
    randomised_data.append(combined_data[:, FEATURE_INPUT:])
    
    return randomised_data


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc, test_acc = [], []
    for j in tqdm(range(epochs)):
        # train
        train_op.run(feed_dict={x: train_data[0], y_: train_data[1]})
        
        # evaluate
        train_acc.append(
            accuracy.eval(feed_dict={x: train_data[0], y_: train_data[1]})
        )
        test_acc.append(
            accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1]})
        )

        # randomise order
        train_data = randomise_dataset(train_data)
    train_acc_set.append(train_acc)
    test_acc_set.append(test_acc)

# * plot graph
plt.figure(1)
plt.plot(range(epochs), train_acc, label='Train Accuracy')
plt.plot(range(epochs), test_acc, label='Test Accuracy')
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train/Test accuracy')
plt.legend()
plt.show()
