import numpy as np
import tensorflow as tf

from utils import ensure_dir


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(feature0, feature1, feature2, label):
    feature = {
        'feature0': _int64_feature([feature0]),
        'feature1': _int64_feature([feature1]),
        'feature2': _int64_feature([feature2]),
        'label': _int64_feature([label]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def prepare(filename='/tmp/demo_data/1.gz'):
    n_observations = 100000
    feature0 = np.random.randint(0, 100, n_observations)
    feature1 = np.random.randint(0, 100, n_observations)
    feature2 = np.random.randint(0, 100, n_observations)
    label = np.random.randint(0, 1, n_observations)

    options = tf.io.TFRecordOptions(compression_type=tf.compat.v1.io.TFRecordCompressionType.GZIP)

    ensure_dir(filename)
    with tf.io.TFRecordWriter(filename, options) as writer:
        for i in range(n_observations):
            example = serialize_example(feature0[i], feature1[i], feature2[i], label[i])
            writer.write(example)


if __name__ == '__main__':
    prepare()
