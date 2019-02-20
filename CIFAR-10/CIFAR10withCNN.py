#AlexNet for CIFAR-10 by 0HOON
import tensorflow as tf
import numpy as np


training_epochs = 30
learning_rate = 0.001
num_models = 7
batch_size = 100
total_batch = 5

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.training = True
        self._build_net()
    
    def _build_net(self):
        #build AlexNet
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3072])
            X_img = tf.reshape(self.X, [-1, 3, 32, 32])
            X_img = tf.transpose(X_img, perm = [0, 3, 2, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs = X_img, filters = 96, kernel_size = [3, 3], padding = 'SAME', activation = tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
            pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = (2, 2), strides = (2, 2), padding = "SAME")
            #dropout1 = tf.layers.dropout(inputs = pool1, rate = 0.7, training = self.training)

            conv2 = tf.layers.conv2d(inputs = pool1, filters = 256, kernel_size = [3, 3], padding = 'SAME', activation = tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
            pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = (2, 2), strides = (2, 2), padding = "SAME")
            #dropout2 = tf.layers.dropout(inputs = pool2, rate = 0.7, training = self.training)

            conv3 = tf.layers.conv2d(inputs = pool2, filters = 384, kernel_size = [3, 3], padding = 'SAME', activation = tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())

            conv4 = tf.layers.conv2d(inputs = conv3, filters = 384, kernel_size = [3, 3], padding = 'SAME', activation = tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())

            conv5 = tf.layers.conv2d(inputs = conv4, filters = 256, kernel_size = [3, 3], padding = 'SAME', activation = tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
            pool5 = tf.layers.max_pooling2d(inputs = conv5, pool_size = (2, 2), strides = (2, 2), padding = "SAME")

            flat = tf.reshape(pool5, [-1, 64 * 8 * 8])
            dense6 = tf.layers.dense(inputs = flat, units = 512, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
            #dropout3 = tf.layers.dropout(inputs = dense3, training = self.training)
            
            dense7 = tf.layers.dense(inputs = dense6, units = 128, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
            #dropout4 = tf.layers.dropout(inputs = dense4, training = self.training)
            
            self.hypothesis = tf.layers.dense(inputs = dense7, units = 10, activation = None, kernel_initializer = tf.contrib.layers.xavier_initializer())
           
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.hypothesis, labels = self.Y))
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

            correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        #get prediction of a model
        self.training = False
        return self.sess.run(self.hypothesis, feed_dict = {self.X: x_test})

    def getAccuracy(self, x_test, y_test):
        #get accuracy of a model
        self.training = False
        return self.sess.run(self.accuracy, feed_dict = {self.X: x_test, self.Y: y_test})
    
    def train(self, x_train, y_train):
        #train a model
        self.training = True
        return self.sess.run([self.cost, self.optimizer], feed_dict = {self.X: x_train, self.Y: y_train})



sess = tf.Session()

#add models
models = []
for i in range(num_models):
    models.append(Model(sess,'model'+str(i)))

sess.run(tf.global_variables_initializer())
print("Learning Started")

for epoch in range(training_epochs):
    avg_cost = np.zeros(len(models))

    for i in range(total_batch):
        #unpickle a batch
        batch = unpickle('cifar-10-batches-py/data_batch_'+str(i+1))

        for k in range(int(10000/batch_size)):       
            x_training = batch[b'data'][k*batch_size: (k+1)*batch_size]   
            #one-hot encoding          
            y_training = np.zeros((batch_size,10))
            y_training[np.arange(batch_size), batch[b'labels'][k*batch_size: (k+1)*batch_size]] = 1    
               
            for m_index, m in enumerate(models):
                c, _ = m.train(x_training, y_training)
                avg_cost[m_index] += c / (batch_size * total_batch)
            
    
    print("Epoch: ", epoch+1, "Cost: ", avg_cost)

print("Learning Finished")

test_batch = unpickle('cifar-10-batches-py/test_batch')


ensemble_accuracy = 0

for k in range(int(10000/batch_size)):     
    predictions = np.zeros(100 * 10).reshape(100, 10)  
    x_test = test_batch[b'data'][k*batch_size: (k+1)*batch_size]     
    #one-hot encoding       
    y_test = np.zeros((batch_size,10))
    y_test[np.arange(batch_size), test_batch[b'labels'][k*batch_size: (k+1)*batch_size]] = 1    
        
    for m_index, m in enumerate(models):
        print(m_index, 'Accuracy: ',m.getAccuracy(x_test, y_test))
        p = m.predict(x_test)
        predictions += p
    #calculate ensemble accuracy
    ensemble_correct_prediction =  tf.equal(tf.argmax(predictions, 1), tf.argmax(y_test, 1))
    ensemble_accuracy += sess.run(tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))/100)
print("Ensemble Accuracy: ", ensemble_accuracy)
