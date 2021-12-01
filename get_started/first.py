from utils import tf_settings

tf = tf_settings(silent=True)
tf.enable_eager_execution()

a = tf.add(1, 2)
print(a)
print(a.numpy())

b = tf.constant('Hello, TensorFlow!')
print(b)
print(b.numpy())
