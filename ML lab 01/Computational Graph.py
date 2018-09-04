import tensorflow as tf

#Build graph (tensors) 
node1 = tf.constant(3.0, tf.float32) #(value, dtype)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1,node2)

#print nodes. Just print its information. Do not print results.
print("node1:", node1, "node2:", node2)
print("node3:", node3)

#print results of nodes
sess = tf.Session() #feed data and run graph
print("sess.run(node1, node2):", sess.run([node1, node2]))
print("sess.runs(node3):", sess.run(node3))