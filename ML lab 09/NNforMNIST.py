import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([784, 100]))
    b1 = tf.Variable(tf.random_normal([100]))
    layer1 = tf.nn.softmax(tf.matmul(X,W1) + b1)

    w1_hist = tf.summary.histogram("weight1", W1)
    b1_hist = tf.summary.histogram("weight1", b1)
    layer_hist = tf.summary.histogram("weight1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([100, 10]))
    b2 = tf.Variable(tf.random_normal([10]))
    layer2 = tf.nn.softmax(tf.matmul(layer1,W2) + b2)

    w2_hist = tf.summary.histogram("weight2", W2)
    b2_hist = tf.summary.histogram("weight1", b2)
    layer2_hist = tf.summary.histogram("weight1", layer2)

with tf.name_scope("hypothesis") as scope:
    W3 = tf.Variable(tf.random_normal([10, nb_classes]))
    b3 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.nn.softmax(tf.matmul(layer2,W3) + b3)

    w3_hist = tf.summary.histogram("weight3", W3)
    b3_hist = tf.summary.histogram("weight1", b3)
    hypothesis_hist = tf.summary.histogram("weight1", hypothesis)


cost = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
cost_summ = tf.summary.scalar("cost", cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_summ = tf.summary.scalar("accuraccy", accuracy)

summary = tf.summary.merge_all()

training_epochs = 100
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, s, _ = sess.run([cost, summary, train], feed_dict = {X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
            writer.add_summary(s, global_step = epoch * total_batch + i)
        
        print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    
    

    print("Learning Finished")
    print("Accuracy:\n", accuracy.eval(session = sess, feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict = {X: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap = 'Greys', interpolation = 'nearest')

