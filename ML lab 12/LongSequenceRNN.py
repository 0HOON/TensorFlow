import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to collect wood and don't assign tem tasks and work, but rather teach them to long for the endless immensity of the sea.")
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}
learning_rate = 0.001
seq_length = 9

x_data = []
y_data = []
for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i: i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]
    #print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    x_data.append(x)
    y_data.append(y)

#RNN parameters
data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
batch_size = len(x_data)

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])
X_one_hot = tf.one_hot(X, num_classes)

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, seq_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(seq_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis = 2)

print(char_set)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        _ = sess.run(train, feed_dict = {X:x_data, Y:y_data})
        if(i%500 == 499):
            result = sess.run(prediction, feed_dict = {X:x_data})
            l = sess.run(loss, feed_dict = {X:x_data, Y:y_data})
            print(i, "loss: ", l)
            for n, k in enumerate(result):
                result_str = [char_set[c] for c in np.squeeze(k)]
                print(n, "prediction: ", ''.join(result_str))
                
    print("learning finished")
    final_result = sess.run(prediction, feed_dict = {X:x_data})
    final_prediction = np.zeros(len(sentence) - 1, dtype = np.int8)
    for i, c in enumerate(final_result):
        for n, k in enumerate(np.squeeze(c)):
            if (i + n) <= 8: 
                final_prediction[i + n] += k/(i + n + 1)
            elif (i + n) >= 169:
                final_prediction[i + n] += k/(178 - i - n)
            else:
                final_prediction[i + n] += k/9
    for i, c in enumerate(final_prediction):
        final_prediction[i] = int(round(c))
        
    print(final_prediction)
    final_str = [char_set[c] for c in final_prediction]
    print("final_prediction: ", ''.join(final_str))