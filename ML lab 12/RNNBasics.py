import tensorflow as tf
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent = 4)
hidden_size = 2
sess = tf.InteractiveSession()

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

x_data = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype = np.float32)

cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype = tf.float32)

sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
