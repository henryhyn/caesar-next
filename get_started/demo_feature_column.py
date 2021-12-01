"""
参考资料: https://juejin.cn/post/6844904201001271310
"""
from utils import tf_settings


def main():
    features = {
        'price1': [[2.0], [30.0], [5.0], [100.0]],
        'price2': [[2.0, 30], [30.0, 12], [2, 5.0], [3, 100.0]],
        'b': [[0], [1], [2], [3]],
        'title': [['asdc'], ['thkj'], ['y0bn'], ['12gb']]
    }

    price1 = tf.feature_column.numeric_column(key='price1')
    price2 = tf.feature_column.numeric_column(key='price2', shape=(2,))
    # 连续特征离散化
    price3 = tf.feature_column.bucketized_column(source_column=price1, boundaries=[1, 10, 100])
    # 类别标识列
    b1 = tf.feature_column.categorical_column_with_identity(key='b', num_buckets=4, default_value=0)
    # 用于把 sparse 特征进行 onehot 变换，用于把 categorical_column_with_* 工具生成的特征变成 onehot 编码
    b2 = tf.feature_column.indicator_column(b1)
    t1 = tf.feature_column.categorical_column_with_hash_bucket(key='title', hash_bucket_size=5, dtype=tf.string)
    t2 = tf.feature_column.indicator_column(t1)

    g = tf.Graph()
    with g.as_default():
        inputs1 = tf.feature_column.input_layer(features=features, feature_columns=[price1])
        inputs2 = tf.feature_column.input_layer(features=features, feature_columns=[price2])
        inputs3 = tf.feature_column.input_layer(features=features, feature_columns=[price1, price2, price3])
        inputs4 = tf.feature_column.input_layer(features=features, feature_columns=[b2, t2])

    with tf.Session(graph=g) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(sess.run(inputs1))
        print(sess.run(inputs2))
        print(sess.run(inputs3))
        print(sess.run(inputs4))


if __name__ == '__main__':
    tf = tf_settings()
    main()
