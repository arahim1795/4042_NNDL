import math
import tensorflow as tf
import numpy as np
import pylab as plt
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(data):
    data_scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data_scaled


# Parameters
# - input: LB to Tendency
FEATURE_INPUT = 8

learning_rate = math.pow(10, -3)
epochs = 100000
num_neurons = 50
batch_size = 8
seed = 10
np.random.seed(seed)
decay = math.pow(10, -3)
keep_probability = 0.8

# Data Pre-Processing / Handler
# X_: inputs, Y_: NSP
# index 0-4: train, 5: test
X_, Y_ = [], []

data = np.genfromtxt("../Data/train_data.csv", delimiter=",")
# process X and Y
X_temp, Y_temp = data[:, :8], data[:, -1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

# add to list
X_.append(X_temp)
Y_.append(Y_temp)

data = np.genfromtxt("../Data/test_data.csv", delimiter=",")
# process X and Y
X_temp, Y_temp = data[:, :8], data[:, -1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

# add to list
X_.append(X_temp)
Y_.append(Y_temp)

# shuffled outputs
actual = Y_[1][idx]
actual_set = np.squeeze(np.asarray(actual))

# for Qn
# - experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

# n = trainX.shape[0]

# Graph Start

# Create the model
x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
y_ = tf.placeholder(tf.float32, [None, 1])

# Hidden Layer
layer_1_weights = tf.Variable(
    tf.truncated_normal(
        [FEATURE_INPUT, num_neurons], stddev=1.0 / math.sqrt(float(FEATURE_INPUT))
    ),
    name="one_weights",
)
layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

# Relu for layer 1
layer_1_output = tf.nn.relu(layer_1_var)

# Dropout for layer 1
layer_1_dropout = tf.nn.dropout(layer_1_output, keep_probability)

# Hidden Layer 2
layer_2_weights = tf.Variable(
    tf.truncated_normal(
        [num_neurons, num_neurons], stddev=1.0 / math.sqrt(float(num_neurons))
    ),
    name="two_weights",
)
layer_2_biases = tf.Variable(tf.zeros([num_neurons]), name="one_biases")
layer_2_var = tf.matmul(layer_1_dropout, layer_2_weights) + layer_2_biases

# layer 2 relu
layer_2_output = tf.nn.relu(layer_2_var)

# layer 2 dropout
layer_2_dropout = tf.nn.dropout(layer_2_output, keep_probability)

# Final layer
layer_final_weights = tf.Variable(
    tf.truncated_normal([num_neurons, 1], stddev=1.0 / math.sqrt(float(num_neurons))),
    name="final_weights",
)
layer_final_biases = tf.Variable(tf.zeros([1]), name="final_biases")
logits = tf.matmul(layer_2_dropout, layer_final_weights) + layer_final_biases

# Regularisation (L2)
loss = tf.reduce_mean(tf.square(y_ - logits))
loss = tf.reduce_mean(loss + (decay * tf.nn.l2_loss(logits)))

# Minimising Lossz`
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32
)
accuracy = tf.reduce_mean(correct_prediction)

train_loss_set, test_loss_set = [], []
prediction_set = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_error_set, test_error_set = [], []

    for i in tqdm(range(epochs)):
        # Batch
        for start, end in zip(
            range(0, len(X_[0]), batch_size), range(batch_size, len(X_[0]), batch_size)
        ):
            if start + batch_size < len(X_[0]):
                train_op.run(feed_dict={x: X_[0][start:end], y_: Y_[0][start:end]})
            else:
                train_op.run(
                    feed_dict={
                        x: X_[0][start : len(X_[0])],
                        y_: Y_[0][start : len(Y_[0])],
                    }
                )
        # calculate loss
        train_error = loss.eval(feed_dict={x: X_[0], y_: Y_[0]})
        train_error_set.append(train_error)
        test_error = loss.eval(feed_dict={x: X_[1], y_: Y_[1]})
        test_error_set.append(test_error)
    # predictions
    prediction_set = sess.run(logits, feed_dict={x: prediciton})


# print(train_acc_set)
# print('-')
# print(test_acc_set)

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_error_set[i], label="Train Loss")
plt.plot(range(epochs), test_error_set[i], label="Test Loss")
plt.xlabel(str(epochs) + " iterations")
plt.ylabel("Train/Test Loss")
plt.legend()
