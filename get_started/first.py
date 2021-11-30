import tensorflow as tf

tf.enable_eager_execution()

a = tf.add(1, 2)
print(a)
print(a.numpy())

b = tf.constant('Hello, TensorFlow!')
print(b)
print(b.numpy())
