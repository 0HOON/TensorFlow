#Same as MultiVariableLinearRegression. This used Matrix.

import tensorflow as tf
import numpy as np

'''
x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]
'''
'''
Loading data from file

xy = np.loadtxt('data-test-score.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
'''

#Using filename queue and batch
filename_queue = tf.train.string_input_producer(
    ['data-test-score.csv'], shuffle = False, name = 'filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#Default values, in case of empty columns. Also specifies the type of the decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults = record_defaults)

#collect batches of csv 
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size = 10)

#place holders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#Our hypothesis (used matrix multiplication)
hypothesis = tf.matmul(X, W) + b

#Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#Start populating th efilename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

for step in range(5001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, b_val, _ = sess.run([cost, hypothesis, b, train], feed_dict = {X: x_batch, Y: y_batch})

    if step % 10 == 0:
        print(step, "cost: ", cost_val, "b: ", b_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

#Asking score
print("Your score will be ", sess.run(hypothesis, feed_dict = {X: [[100, 70, 101]]}))