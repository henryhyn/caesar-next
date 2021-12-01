import shutil
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from absl import app
from sklearn.model_selection import train_test_split

from utils import tf_settings, logger


@dataclass
class Context:
    batch_size: int
    train_epochs: int
    epochs_between_evals: int
    model_dir: str = '/data/models/job_card_model'
    data_path: str = '/data/datasets/job_card_data_20211130.tsv'


def parse_args():
    app.define_help_flags()
    app.flags.DEFINE_integer('batch_size', 128, 'Batch size for training and evaluation.')
    app.flags.DEFINE_integer('train_epochs', 40, 'The number of epochs used to train.')
    app.flags.DEFINE_integer('epochs_between_evals', 2, 'The number of training epochs to run between evaluations.')
    app.parse_flags_with_usage(sys.argv)
    args = app.FLAGS
    return Context(
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        epochs_between_evals=args.epochs_between_evals
    )


def load_data():
    data = pd.read_csv(ctx.data_path, sep='\t').fillna('UK')
    data = data[data.apply(lambda row: row['y'] == 1 or np.random.uniform() < 0.2, axis=1)]
    train, test = train_test_split(data, test_size=0.3)
    logger.info(data['y'].value_counts())
    logger.info(data.head())
    logger.info(f'data size: {len(data)}, train size: {len(train)}, test size: {len(test)}')

    train_x = dict(train)
    train_y = train_x.pop('y')
    test_x = dict(test)
    test_y = test_x.pop('y')
    return (train_x, train_y), (test_x, test_y)


def input_fn(inputs, num_epochs, shuffle, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if shuffle:
        dataset = dataset.shuffle(100000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def build_feature_columns():
    gender = tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', vocabulary_list=['UK', '男', '女'], dtype=tf.string)

    school_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'school_type', vocabulary_list=['未知', '其他', '海外', '海外QS_TOP100', '双一流', '985', '211', '一本', '二本', '初高中'],
        dtype=tf.string)

    edu_level = tf.feature_column.categorical_column_with_vocabulary_list(
        'edu_level', vocabulary_list=['UK', '博士后', '博士', '硕士', '工程硕士', '研究生', '本科', '专科', '大学', '大专', '高中', '初中', '小学'],
        dtype=tf.string)

    honor_level = tf.feature_column.categorical_column_with_identity('honor_level', num_buckets=9)

    job_city = tf.feature_column.categorical_column_with_hash_bucket(
        'job_city', hash_bucket_size=100, dtype=tf.string)

    career_job1_1 = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job1_1', hash_bucket_size=100, dtype=tf.string)

    career_job1_2 = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job1_2', hash_bucket_size=100, dtype=tf.string)

    career_job1_3 = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job1_3', hash_bucket_size=100, dtype=tf.string)

    career_job_id = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job_id', hash_bucket_size=100, dtype=tf.int32)

    company_id = tf.feature_column.categorical_column_with_hash_bucket(
        'company_id', hash_bucket_size=100, dtype=tf.int32)

    job_id = tf.feature_column.categorical_column_with_hash_bucket(
        'job_id', hash_bucket_size=100, dtype=tf.int32)

    base_columns = [
        gender,
        school_type,
        edu_level,
        honor_level,
        job_city,
        career_job1_1,
        career_job1_2,
        career_job1_3,
        career_job_id,
        company_id,
        job_id
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(['school_type', 'edu_level'], hash_bucket_size=100)
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        tf.feature_column.indicator_column(gender),
        tf.feature_column.indicator_column(school_type),
        tf.feature_column.indicator_column(edu_level),
        tf.feature_column.indicator_column(honor_level),
        tf.feature_column.embedding_column(job_city, dimension=8),
        tf.feature_column.embedding_column(career_job1_1, dimension=8),
        tf.feature_column.embedding_column(career_job1_2, dimension=8),
        tf.feature_column.embedding_column(career_job1_3, dimension=8),
        tf.feature_column.embedding_column(career_job_id, dimension=8),
        tf.feature_column.embedding_column(company_id, dimension=8),
        tf.feature_column.embedding_column(job_id, dimension=8)
    ]

    return wide_columns, deep_columns


def build_model():
    wide_columns, deep_columns = build_feature_columns()
    hidden_units = [100, 80, 60, 40, 20]
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
        return input_fn((train_x, train_y), ctx.epochs_between_evals, True, ctx.batch_size)

    def eval_input_fn():
        return input_fn((test_x, test_y), 1, False, ctx.batch_size)

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
