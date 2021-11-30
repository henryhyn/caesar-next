import tensorflow as tf

from utils import tf_settings, logger


def define_graph():
    g = tf.Graph()
    with g.as_default():
        x = tf.Variable(3, name='x')
        y = tf.Variable(4, name='y')
        f = x * x * y + y + 2
    return [g, x, y, f]


def main():
    [g, x, y, f] = define_graph()
    tf.summary.FileWriter('/tmp/demo_mul', g)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)
    with tf.Session(graph=g, config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        logger.info(f'result = {sess.run(f)}')


if __name__ == '__main__':
    tf_settings()
    main()
