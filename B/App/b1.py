#
# Project 1, starter code part b
#
import random
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from tqdm import tqdm
import pandas

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(data):
    data_scaled = (data- np.mean(data, axis=0))/ np.std(data, axis=0)
    return data_scaled 

# Parameters
# - input: LB to Tendency
FEATURE_INPUT = 7

learning_rate = math.pow(10,-3)
epochs = 1000
num_neurons = 10
batch_size = 8
seed = 10
np.random.seed(seed)
decay = math.pow(10, -3)

# Data Pre-Processing / Handler
# X_: inputs, Y_: NSP
# index 0-4: train, 5: test
X_, Y_ = [], []

data = np.genfromtxt('../Data/train_data.csv', delimiter=',')
# process X and Y
X_temp, Y_temp = data[:,1:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

#add to list
X_.append(X_temp)
Y_.append(Y_temp)

data = np.genfromtxt('../Data/test_data.csv', delimiter=',')
#process X and Y
X_temp, Y_temp = data[:,1:8], data[:,-1]
Y_temp = Y_temp.reshape(Y_temp.shape[0], 1)
X_temp = scale(X_temp)

#add to list
X_.append(X_temp)
Y_.append(Y_temp)

idx = np.arange(X_[1].shape[0])
np.random.shuffle(idx)
# shuffled inputs
prediciton = X_[1][idx]
 
#shuffled outputs
actual = Y_[1][idx] 
actual_set = np.squeeze(np.asarray(actual))

# Graph Start

# Create the model
x = tf.placeholder(tf.float32, [None, FEATURE_INPUT])
y_ = tf.placeholder(tf.float32, [None, 1])

# Hidden Layer
layer_1_weights = tf.Variable(tf.truncated_normal(
    [FEATURE_INPUT, num_neurons], stddev=1.0/math.sqrt(float(FEATURE_INPUT))), name='one_weights')
layer_1_biases = tf.Variable(tf.zeros([num_neurons]), name='one_biases')
layer_1_var = tf.matmul(x, layer_1_weights) + layer_1_biases

layer_1_output = tf.nn.relu(layer_1_var)

# Final layer
layer_final_weights = tf.Variable(tf.truncated_normal(
    [num_neurons, 1], stddev=1.0/math.sqrt(float(num_neurons))), name='final_weights')
layer_final_biases = tf.Variable(tf.zeros([1]), name='final_biases')
logits = tf.matmul(layer_1_output, layer_final_weights) + layer_final_biases

# Regularisation (L2)
regularization = tf.nn.l2_loss(layer_1_weights) + tf.nn.l2_loss(layer_final_weights)
loss = tf.reduce_mean(tf.square(y_ - logits))
l2_loss = tf.reduce_mean(loss + decay*regularization)

# Minimising Loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(l2_loss)

correct_prediction = tf.cast(
    tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

train_loss_set,test_loss_set = [], []
prediction_set = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_error_set, test_error_set = [], []

    for i in tqdm(range(epochs)):
        # Batch
        for start, end in zip(range(0, len(X_[0]), batch_size), range(batch_size, len(X_[0]), batch_size)):
            if start+batch_size < len(X_[0]):
                train_op.run(feed_dict={x: X_[0][start:end], y_: Y_[0][start:end]})
            else: 
                train_op.run(feed_dict={x: X_[0][start:len(X_[0])], y_: Y_[0][start:len(Y_[0])]})
        # calculate loss
        train_error = l2_loss.eval(feed_dict={x: X_[0], y_: Y_[0]})
        train_error_set.append(train_error)
        test_error = l2_loss.eval(feed_dict={x: X_[1], y_: Y_[1]})
        test_error_set.append(test_error)
    #predictions
    prediction_set=(sess.run(logits,feed_dict={x:prediciton}))


# print(train_acc_set)
# print('-')
# print(test_acc_set)

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_error_set, label ='Train Loss')
plt.plot(range(epochs), test_error_set, label = 'Test Loss')
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train/Test Loss')
plt.legend()

plt.figure(2)
plt.scatter(range(50), prediction_set[0:50])
plt.plot(range(50), prediction_set[0:50], label="prediction")
plt.scatter(range(50), actual_set[0:50])
plt.plot(range(50), actual_set[0:50], label="actual")
plt.xlabel("Predicition Number")
plt.ylabel("Admission chance")
plt.legend()

corr_data =  np.genfromtxt('../Data/train_data.csv', delimiter=',')
corr_data =  data[:,1:9]

test_dataframe = pandas.DataFrame(corr_data,columns=["GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research","Chance of Admit"])
plt.matshow(test_dataframe.corr())
# use shape 1 for x-axis
plt.xticks(range(test_dataframe.shape[1]), test_dataframe.columns, fontsize=7, rotation=90)
plt.yticks(range(test_dataframe.shape[1]), test_dataframe.columns, fontsize=7)
plt.colorbar()
plt.legend()
plt.show()