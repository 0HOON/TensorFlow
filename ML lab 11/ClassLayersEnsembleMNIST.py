import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
training_epochs = 15
batch_size = 100
learning_rate = 0.001
num_models = 7

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.training = True
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs = X_img, filters = 32, kernel_size = [3,3], padding = "SAME", activation = tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2, padding = "SAME")
            dropout1 = tf.layers.dropout(inputs = pool1, rate = 0.7, training = self.training)

            conv2 = tf.layers.conv2d(inputs = dropout1, filters = 64, kernel_size = [3,3], padding = "SAME", activation = tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2, padding = "SAME")
            dropout2 = tf.layers.dropout(inputs = pool2, rate = 0.7, training = self.training)

            flat = tf.reshape(dropout2, [-1, 64 * 7 * 7])
            dense = tf.layers.dense(inputs = flat, units = 512, activation = tf.nn.relu)
            
            self.logits = tf.layers.dense(inputs = dense, units = 10, activation = None)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        self.training = False
        return self.sess.run(self.logits, feed_dict = {self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        self.training = False
        return self.sess.run(self.accuracy, feed_dict = {self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        self.training = True
        return self.sess.run([self.cost, self.optimizer], feed_dict = {self.X: x_data, self.Y: y_data})


sess = tf.Session()

models = []
for n in range(num_models):
    models.append(Model(sess, "model"+str(n)))


sess.run(tf.global_variables_initializer())
print('Learning started')

for epoch in range(training_epochs):
    avg_cost = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        for m_index, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost[m_index] += c / total_batch
        
    print('Epoch: ', epoch+1, ' Cost: ', avg_cost)

print('Learning finished')

test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

for m_index, m in enumerate(models):
    print(m_index, 'Accuracy: ', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

print('Ensemble Accuracy: ', sess.run(ensemble_accuracy))

'''
0 Accuracy:  0.9787
1 Accuracy:  0.9774
2 Accuracy:  0.977
3 Accuracy:  0.9801
4 Accuracy:  0.9794
5 Accuracy:  0.9776
6 Accuracy:  0.9816
Ensemble Accuracy:  0.9927
'''