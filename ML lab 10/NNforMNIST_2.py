import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("layer1") as scope:
    W1 = tf.get_variable("W1", shape = [784, 512], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([512]))
    layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob = keep_prob)

    w1_hist = tf.summary.histogram("weight1", W1)
    b1_hist = tf.summary.histogram("bias1", b1)
    layer_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.get_variable("W2", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([512]))
    layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
    layer2 = tf.nn.dropout(layer2, keep_prob = keep_prob)

    w2_hist = tf.summary.histogram("weight2", W2)
    b2_hist = tf.summary.histogram("bias2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

with tf.name_scope("layer3") as scope:
    W3 = tf.get_variable("W3", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([512]))
    layer3 = tf.nn.relu(tf.matmul(layer1,W3) + b3)
    layer3 = tf.nn.dropout(layer3, keep_prob = keep_prob)

    w3_hist = tf.summary.histogram("weight3", W3)
    b3_hist = tf.summary.histogram("bias3", b3)
    layer3_hist = tf.summary.histogram("layer3", layer3)

with tf.name_scope("layer4") as scope:
    W4 = tf.get_variable("W4", shape = [512, 512], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([512]))
    layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
    layer4 = tf.nn.dropout(layer4, keep_prob = keep_prob)

    w4_hist = tf.summary.histogram("weight4", W4)
    b4_hist = tf.summary.histogram("bias4", b4)
    layer4_hist = tf.summary.histogram("layer4", layer4)

with tf.name_scope("hypothesis") as scope:
    W5 = tf.get_variable("W5", shape = [512, nb_classes], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(layer4,W5) + b5

    w5_hist = tf.summary.histogram("weight5", W5)
    b5_hist = tf.summary.histogram("bias5", b5)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
train = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
cost_summ = tf.summary.scalar("cost", cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_summ = tf.summary.scalar("accuraccy", accuracy)

summary = tf.summary.merge_all()

training_epochs = 15
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
            c, s,  _ = sess.run([cost, summary, train], feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch
            writer.add_summary(s, global_step = epoch * total_batch + i)
        
        print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    
    

    print("Learning Finished")
    print("Accuracy:\n", accuracy.eval(session = sess, feed_dict = {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict = {X: mnist.test.images[r:r+1], keep_prob: 1}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap = 'Greys', interpolation = 'nearest')
    
'''
Epoch:  0001 cost =  0.411695754
Epoch:  0002 cost =  0.158077145
Epoch:  0003 cost =  0.115302339
Epoch:  0004 cost =  0.092440161
Epoch:  0005 cost =  0.080749021
Epoch:  0006 cost =  0.072651395
Epoch:  0007 cost =  0.062079404
Epoch:  0008 cost =  0.059043657
Epoch:  0009 cost =  0.054176884
Epoch:  0010 cost =  0.049291687
Epoch:  0011 cost =  0.043601990
Epoch:  0012 cost =  0.046984426
Epoch:  0013 cost =  0.044421965
Epoch:  0014 cost =  0.036697221
Epoch:  0015 cost =  0.041609486
Learning Finished
Accuracy:
 0.9807
Label:  [1]
Prediction:  [1]
'''