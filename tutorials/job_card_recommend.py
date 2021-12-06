import glob
import os.path as osp
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from absl import app
from sklearn.model_selection import train_test_split

from tutorials.job_card_examples import examples
from utils import tf_settings, logger, delete_dir
from utils.time_util import date_time


@dataclass
class Context:
    train: bool
    batch_size: int
    train_epochs: int
    epochs_between_evals: int
    model_dir: str
    data_path: str = '/data/datasets/job_card_data_20211116_20211206_android_intern.tsv'


def parse_args():
    app.define_help_flags()
    app.flags.DEFINE_boolean('train', False, 'train or predict')
    app.flags.DEFINE_integer('batch_size', 128, 'Batch size for training and evaluation.')
    app.flags.DEFINE_integer('train_epochs', 40, 'The number of epochs used to train.')
    app.flags.DEFINE_integer('epochs_between_evals', 2, 'The number of training epochs to run between evaluations.')
    app.flags.DEFINE_string('root_path', '/data/models/job_card', 'root path')
    app.flags.DEFINE_string('version', None, 'model version')
    app.parse_flags_with_usage(sys.argv)
    args = app.FLAGS

    if args.train:
        out_path = osp.join(args.root_path, date_time())
        delete_dir(out_path)
    else:
        if args.version:
            files = glob.glob(args.root_path + '/' + args.version)
        else:
            files = glob.glob(args.root_path + '/202*')
        files.sort()
        out_path = files[-1]

    return Context(
        train=args.train,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        epochs_between_evals=args.epochs_between_evals,
        model_dir=out_path
    )


def load_data():
    data = pd.read_csv(ctx.data_path, sep='\t').fillna('UK')
    # data = data[data.apply(lambda row: row['y'] == 1 or np.random.uniform() < 0.2, axis=1)]
    train, test = train_test_split(data, test_size=50000)
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
    from_create = tf.feature_column.numeric_column('from_create', dtype=tf.float32)
    avg_process_rate = tf.feature_column.numeric_column('avg_process_rate', dtype=tf.float32)
    avg_process_sec = tf.feature_column.numeric_column('avg_process_sec', dtype=tf.float32)
    day_salary_min = tf.feature_column.numeric_column('day_salary_min', dtype=tf.float32)
    day_salary_max = tf.feature_column.numeric_column('day_salary_max', dtype=tf.float32)

    from_create_buckets = tf.feature_column.bucketized_column(
        from_create, boundaries=[0.5, 1, 3, 6, 12, 24, 48, 72, 96, 120, 240, 480, 720])

    avg_process_rate_buckets = tf.feature_column.bucketized_column(
        avg_process_rate, boundaries=[10, 20, 30, 40, 50, 60, 70, 80, 90])

    b_minute = np.array([5, 10, 20, 30]).dot(60)
    b_hour = np.array([1, 3, 6, 12, 24, 48, 72, 96]).dot(3600)
    avg_process_sec_buckets = tf.feature_column.bucketized_column(
        avg_process_sec, boundaries=list(b_minute) + list(b_hour))

    b_salary = [10, 20, 50, 100, 150, 200, 400, 600, 800, 1000, 1500, 2000, 4000]
    day_salary_min_buckets = tf.feature_column.bucketized_column(
        day_salary_min, boundaries=b_salary)
    day_salary_max_buckets = tf.feature_column.bucketized_column(
        day_salary_max, boundaries=b_salary)

    gender = tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', vocabulary_list=['UK', '男', '女'], dtype=tf.string)

    school_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'school_type', vocabulary_list=['未知', '其他', '海外', '海外QS_TOP100', '双一流', '985', '211', '一本', '二本', '初高中'],
        dtype=tf.string)

    edu_level = tf.feature_column.categorical_column_with_vocabulary_list(
        'edu_level', vocabulary_list=['UK', '博士后', '博士', '硕士', '工程硕士', '研究生', '本科', '专科', '大学', '大专', '高中', '初中', '小学'],
        dtype=tf.string)

    i_edu_level = tf.feature_column.categorical_column_with_vocabulary_list(
        'i_edu_level', vocabulary_list=[0, 4000, 5000, 6000, 7000],
        dtype=tf.int64)

    honor_level = tf.feature_column.categorical_column_with_identity('honor_level', num_buckets=9)

    job_city = tf.feature_column.categorical_column_with_hash_bucket(
        'job_city', hash_bucket_size=100, dtype=tf.string)

    live_place = tf.feature_column.categorical_column_with_hash_bucket(
        'live_place', hash_bucket_size=100, dtype=tf.string)

    work_want_place = tf.feature_column.categorical_column_with_hash_bucket(
        'work_want_place', hash_bucket_size=100, dtype=tf.string)

    career_job1_1 = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job1_1', hash_bucket_size=100, dtype=tf.string)

    career_job1_2 = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job1_2', hash_bucket_size=100, dtype=tf.string)

    career_job1_3 = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job1_3', hash_bucket_size=100, dtype=tf.string)

    career_job_id = tf.feature_column.categorical_column_with_hash_bucket(
        'career_job_id', hash_bucket_size=100, dtype=tf.int64)

    company_id = tf.feature_column.categorical_column_with_hash_bucket(
        'company_id', hash_bucket_size=100, dtype=tf.int64)

    job_id = tf.feature_column.categorical_column_with_hash_bucket(
        'job_id', hash_bucket_size=100, dtype=tf.int64)

    base_columns = [
        from_create_buckets,
        avg_process_rate_buckets,
        avg_process_sec_buckets,
        day_salary_min_buckets,
        day_salary_max_buckets,
        gender,
        school_type,
        edu_level,
        i_edu_level,
        honor_level,
        job_city,
        live_place,
        work_want_place,
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
        tf.feature_column.indicator_column(from_create_buckets),
        tf.feature_column.indicator_column(avg_process_rate_buckets),
        tf.feature_column.indicator_column(avg_process_sec_buckets),
        tf.feature_column.indicator_column(day_salary_min_buckets),
        tf.feature_column.indicator_column(day_salary_max_buckets),
        tf.feature_column.indicator_column(gender),
        tf.feature_column.indicator_column(school_type),
        tf.feature_column.indicator_column(edu_level),
        tf.feature_column.indicator_column(i_edu_level),
        tf.feature_column.indicator_column(honor_level),
        tf.feature_column.embedding_column(job_city, dimension=8),
        tf.feature_column.embedding_column(live_place, dimension=8),
        tf.feature_column.embedding_column(work_want_place, dimension=8),
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
    hidden_units = [200, 200, 150, 150]
    model = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        model_dir=ctx.model_dir)
    feature_spec = tf.feature_column.make_parse_example_spec(wide_columns + deep_columns)
    receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return model, receiver_fn


def train():
    logger.info(ctx)
    (train_x, train_y), (test_x, test_y) = load_data()

    def train_input_fn():
        return input_fn((train_x, train_y), ctx.epochs_between_evals, True, ctx.batch_size)

    def eval_input_fn():
        return input_fn((test_x, test_y), 1, False, ctx.batch_size)

    model, receiver_fn = build_model()
    template = 'step: {global_step:d}, accuracy: {accuracy:.2%}, precision: {precision:.2%}, recall: {recall:.2%}, auc: {auc:.4f}, loss: {loss:.2f}'
    for n in range(ctx.train_epochs // ctx.epochs_between_evals):
        model.train(input_fn=train_input_fn)
        eval_result = model.evaluate(input_fn=eval_input_fn)
        logger.info(template.format(**eval_result))
        logger.info(f'==> batch predict: {batch_predict(model)}')
    model.export_saved_model(export_dir_base=ctx.model_dir, serving_input_receiver_fn=receiver_fn)


def predict():
    model, receiver_fn = build_model()
    logger.info(f'==> batch predict: {batch_predict(model)}')


def batch_predict(model):
    def predict_input_fn():
        return input_fn(examples, 1, False, ctx.batch_size)

    predictions = model.predict(input_fn=predict_input_fn)
    return [predict['probabilities'][1] for predict in predictions]


if __name__ == '__main__':
    ctx = parse_args()
    tf = tf_settings()
    if ctx.train:
        train()
    else:
        predict()
