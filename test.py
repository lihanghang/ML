import tensorflow as tf

a = tf.constant(12)
b = tf.constant(54)
#limit work filed
with tf.Session() as sess:
#sess = tf.Session()
    print sess.run(a+b)
c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)
add = tf.add(c, d)
mul = tf.multiply(c, d)
with tf.Session() as sess:
    print(sess.run(add, feed_dict={c:2, d:3}))
    print(sess.run(mul, feed_dict={c:5, d:6}))

#sava graph of compute
writer = tf.summary.FileWriter(logdir="logs", graph=tf.get_default_graph())
writer.flush()
