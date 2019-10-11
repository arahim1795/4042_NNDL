#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 7

learning_rate = 0.01
epochs = 1000
batch_size = 8
num_neuron = 30
seed = 10
np.random.seed(seed)

#read and divide data into test and train sets 
admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
Y_data = Y_data.reshape(Y_data.shape[0], 1)

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

# experiment with small datasets
trainX = X_data[:100]
trainY = Y_data[:100]

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, 1], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights')
biases = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
y = tf.matmul(x, weights) + biases



#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
loss = tf.reduce_mean(tf.square(y_ - y))
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_err = []
	for i in range(epochs):
		train_op.run(feed_dict={x: trainX, y_: trainY})
		err = loss.eval(feed_dict={x: trainX, y_: trainY})
		train_err.append(err)

		if i % 100 == 0:
			print('iter %d: train error %g'%(i, train_err[i]))

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train Error')
plt.show()
