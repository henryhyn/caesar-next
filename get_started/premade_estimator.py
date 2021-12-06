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
from absl import app
from tensorflow import keras

from utils import logger, tf_settings

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
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


def input_fn(inputs, epochs, shuffle, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return _next_batch(dataset, epochs, shuffle, batch_size)


def csv_input_fn(file_path, epochs, shuffle, batch_size):
    dataset = tf.data.TextLineDataset(file_path).skip(1).map(_parse_line)
    return _next_batch(dataset, epochs, shuffle, batch_size)


def _next_batch(dataset, epochs, shuffle, batch_size):
    if shuffle:
        dataset = dataset.shuffle(1024)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def _parse_line(line):
    fields = tf.io.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    return features, features.pop(CSV_COLUMN_NAMES[-1])


def build_feature_columns():
    return [tf.feature_column.numeric_column(key=key) for key in CSV_COLUMN_NAMES[:-1]]


def build_model():
    feature_columns = build_feature_columns()
    hidden_units = [10, 10]
    model = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        n_classes=len(SPECIES),
        model_dir=ctx.model_dir)
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return model, receiver_fn


def main():
    logger.info(ctx)
    shutil.rmtree(ctx.model_dir, ignore_errors=True)
    # (train_x, train_y), (test_x, test_y) = load_data()
    train_path, test_path = maybe_download()

    def train_input_fn():
        # return input_fn((train_x, train_y), None, True, ctx.batch_size)
        return csv_input_fn(train_path, None, True, ctx.batch_size)

    def eval_input_fn():
        # return input_fn((test_x, test_y), 1, False, ctx.batch_size)
        return csv_input_fn(test_path, 1, False, ctx.batch_size)

    model, receiver_fn = build_model()
    model.train(input_fn=train_input_fn, steps=ctx.train_steps)
    eval_result = model.evaluate(input_fn=eval_input_fn)
    logger.info('Test set accuracy: {accuracy:.2%}'.format(**eval_result))

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
    template = 'Prediction is "{}" ({:.2%}), expected "{}"'
    for predict, expect in zip(predictions, expected):
        class_id = predict['class_ids'][0]
        probability = predict['probabilities'][class_id]
        logger.info(template.format(SPECIES[class_id], probability, expect))

    model.export_saved_model(export_dir_base=ctx.model_dir, serving_input_receiver_fn=receiver_fn)


if __name__ == '__main__':
    ctx = parse_args()
    tf = tf_settings()
    main()
