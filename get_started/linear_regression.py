import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from absl import app

from utils import tf_settings, logger


@dataclass
class Context:
    epochs: int
    step: int
    learning_rate: float


def parse_args():
    app.define_help_flags()
    app.flags.DEFINE_integer('epochs', 20, 'epoch number')
    app.flags.DEFINE_integer('step', 2, 'print loss per step')
    app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
    app.parse_flags_with_usage(sys.argv)
    args = app.FLAGS
    return Context(
        epochs=args.epochs,
        step=args.step,
        learning_rate=args.learning_rate
    )


def load_data():
    data_x = np.linspace(-1, 1, 100)
    data_y = 2 * data_x + 1 + .3 * np.random.randn(*data_x.shape)
    return [data_x, data_y]


def define_graph():
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        w = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='w')
        b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='b')
        y_ = tf.add(tf.multiply(w, x), b)
        loss = tf.reduce_mean(tf.square(y_ - y))
        optimizer = tf.train.GradientDescentOptimizer(ctx.learning_rate)
        train_op = optimizer.minimize(loss)
    return [g, x, y, y_, w, b, loss, train_op]


def main():
    logger.info(ctx)

    [data_x, data_y] = load_data()
    plt.plot(data_x, data_y, 'ro', label='Original data')

    [g, x, y, y_, w, b, loss, train_op] = define_graph()
    with tf.Session(graph=g) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for epoch in range(ctx.epochs):
            for [point_x, point_y] in zip(data_x, data_y):
                sess.run(train_op, feed_dict={x: point_x, y: point_y})
            if epoch % ctx.step == 0:
                [curr_w, curr_b] = sess.run([w, b])
                curr_loss = sess.run(loss, feed_dict={x: data_x, y: data_y})
                logger.info(f'Epoch: {epoch}, w: {curr_w}, b: {curr_b}, loss: {curr_loss}')
        pred_y = sess.run(y_, feed_dict={x: data_x})
    plt.plot(data_x, pred_y, label='Fitted line')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ctx = parse_args()
    tf = tf_settings()
    main()
