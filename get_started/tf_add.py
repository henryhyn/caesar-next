from utils import tf_settings, logger


def define_graph():
    g = tf.Graph()
    with g.as_default():
        a = tf.constant([1.0, 2.0], name='a')
        b = tf.constant([2.0, 3.0], name='b')
        result = tf.add(a, b)
    return [g, result]


def main():
    # 定义计算图
    [g, result] = define_graph()
    tf.summary.FileWriter('/tmp/demo_add', g)

    # 在会话中执行定义好的计算图
    with tf.Session(graph=g) as sess:
        logger.info(sess.run(result))


if __name__ == '__main__':
    tf = tf_settings()
    main()
