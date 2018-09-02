import tensorflow as tf

X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])

X_data = []
Y_data = []

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#Our hypothesis XW+b
hypothesis = X * W + b

#cost function
cost = tf.reduce_mean(tf.square(hypothesis -Y))

#Minimize cose
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

#Get data
while 1:
    x = input("X data:")
    if x == 'q':
        break
    X_data.append(x)

while 1:
    y = input("Y data:")
    if y == 'q':
        break
    Y_data.append(y)

#Fit the line
for step in range(5001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X: X_data, Y: Y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)