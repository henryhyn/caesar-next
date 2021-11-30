"""
鸢尾花分类问题
参考: https://github.com/tensorflow/models/blob/r1.8.1/samples/core/get_started/premade_estimator.py
python -m get_started.premade_estimator --help
"""

import os.path as osp
import shutil
import sys
from dataclasses import dataclass

import pandas as pd
import tensorflow as tf
from absl import app
from tensorflow import keras

from utils import logger, tf_settings

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


@dataclass
class Context:
    batch_size: int
    train_steps: int
    model_dir: str = '/data/models/iris_model'
    base_url: str = 'http://download.tensorflow.org/data'
    train_file: str = 'iris_training.csv'
    test_file: str = 'iris_test.csv'


def parse_args():
    app.define_help_flags()
    app.flags.DEFINE_integer('batch_size', 32, 'Batch size for training and evaluation.')
    app.flags.DEFINE_integer('train_steps', 1000, 'The number of training steps.')
    app.parse_flags_with_usage(sys.argv)
    args = app.FLAGS
    return Context(
        batch_size=args.batch_size,
        train_steps=args.train_steps
    )


def maybe_download():
    train_path = keras.utils.get_file(ctx.train_file, osp.join(ctx.base_url, ctx.train_file))
    test_path = keras.utils.get_file(ctx.test_file, osp.join(ctx.base_url, ctx.test_file))
    return train_path, test_path


def load_data(y_name=CSV_COLUMN_NAMES[-1]):
    train_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x = dict(train)
    train_y = train_x.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x = dict(test)
    test_y = test_x.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


def input_fn(inputs, num_epochs, shuffle, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def build_feature_columns():
    return [tf.feature_column.numeric_column(key=key) for key in CSV_COLUMN_NAMES[:-1]]


def build_model():
    feature_columns = build_feature_columns()
    hidden_units = [10, 10]
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        n_classes=3,
        model_dir=ctx.model_dir)


def main():
    logger.info(ctx)
    shutil.rmtree(ctx.model_dir, ignore_errors=True)
    (train_x, train_y), (test_x, test_y) = load_data()

    def train_input_fn():
        return input_fn((train_x, train_y), None, True, ctx.batch_size)

    def eval_input_fn():
        return input_fn((test_x, test_y), 1, False, ctx.batch_size)

    model = build_model()
    model.train(input_fn=train_input_fn, steps=ctx.train_steps)
    eval_result = model.evaluate(input_fn=eval_input_fn)
    logger.info('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))

    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    def predict_input_fn():
        return input_fn(predict_x, 1, False, ctx.batch_size)

    predictions = model.predict(input_fn=predict_input_fn)
    template = 'Prediction is "{}" ({:.2f}%), expected "{}"'
    for predict, expect in zip(predictions, expected):
        class_id = predict['class_ids'][0]
        probability = predict['probabilities'][class_id]
        logger.info(template.format(SPECIES[class_id], 100 * probability, expect))


if __name__ == '__main__':
    ctx = parse_args()
    tf_settings()
    main()
