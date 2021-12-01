from utils import tf_settings

tf = tf_settings(silent=True)
print(tf.constant(3))
print(tf.constant([1., 2., 3.]))
print(tf.constant([[1., 2., 3.], [4., 5., 6.]]))
print(tf.constant([[[1., 2., 3.]], [[7., 8., 9.]]]))
