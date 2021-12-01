"""
预测年收入是否超过 5W 美元
参考: https://github.com/tensorflow/models/blob/r1.8.1/official/wide_deep/wide_deep.py
python -m tutorials.wide_deep --help
"""
import os.path as osp
import shutil
import sys
from dataclasses import dataclass

import pandas as pd
from absl import app
from tensorflow import keras

from utils import logger, tf_settings

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]


@dataclass
class Context:
    batch_size: int
    train_epochs: int
    epochs_between_evals: int
    model_type: str
    model_dir: str = '/data/models/census_model'
    base_url: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
    train_file: str = 'adult.data'
    test_file: str = 'adult.test'


def parse_args():
    app.define_help_flags()
    app.flags.DEFINE_integer('batch_size', 32, 'Batch size for training and evaluation.')
    app.flags.DEFINE_integer('train_epochs', 40, 'The number of epochs used to train.')
    app.flags.DEFINE_integer('epochs_between_evals', 2, 'The number of training epochs to run between evaluations.')
    app.flags.DEFINE_enum('model_type', 'wide_deep', ['wide', 'deep', 'wide_deep'], 'Select model topology.')
    app.parse_flags_with_usage(sys.argv)
    args = app.FLAGS
    return Context(
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        epochs_between_evals=args.epochs_between_evals,
        model_type=args.model_type
    )


def preprocess_data(input_file):
    output_file = input_file + '.md'
    with open(output_file, mode='w', encoding='utf-8') as writer:
        with open(input_file, mode='r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip().replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line.endswith('.'):
                    line = line[:-1]
                writer.write(line + '\n')
    return output_file


def maybe_download():
    train_path = keras.utils.get_file(ctx.train_file, osp.join(ctx.base_url, ctx.train_file))
    test_path = keras.utils.get_file(ctx.test_file, osp.join(ctx.base_url, ctx.test_file))
    return preprocess_data(train_path), preprocess_data(test_path)


def load_data(y_name=_CSV_COLUMNS[-1]):
    train_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=_CSV_COLUMNS)
    train_x = dict(train)
    train_y = train_x.pop(y_name)

    test = pd.read_csv(test_path, names=_CSV_COLUMNS)
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
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)

    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            [education, 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, education, 'occupation'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def build_model():
    wide_columns, deep_columns = build_feature_columns()
    hidden_units = [100, 75, 50, 25]
    if ctx.model_type == 'wide':
        return tf.estimator.LinearClassifier(
            feature_columns=wide_columns,
            model_dir=ctx.model_dir)
    elif ctx.model_type == 'deep':
        return tf.estimator.DNNClassifier(
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            model_dir=ctx.model_dir)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            model_dir=ctx.model_dir)


def main():
    logger.info(ctx)
    shutil.rmtree(ctx.model_dir, ignore_errors=True)
    (train_x, train_y), (test_x, test_y) = load_data()

    def train_input_fn():
        label = tf.equal(train_y, '>50K')
        return input_fn((train_x, label), ctx.epochs_between_evals, True, ctx.batch_size)

    def eval_input_fn():
        label = tf.equal(test_y, '>50K')
        return input_fn((test_x, label), 1, False, ctx.batch_size)

    model = build_model()
    template = 'step: {global_step:d}, accuracy: {accuracy:0.4f}, precision: {precision:0.4f}, recall: {recall:0.4f}, auc: {auc:0.4f}, loss: {loss:0.2f}'
    for n in range(ctx.train_epochs // ctx.epochs_between_evals):
        model.train(input_fn=train_input_fn)
        eval_result = model.evaluate(input_fn=eval_input_fn)
        logger.info(template.format(**eval_result))


if __name__ == '__main__':
    ctx = parse_args()
    tf = tf_settings()
    main()
