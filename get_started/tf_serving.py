"""
参考资料: http://shzhangji.com/cnblogs/2018/05/14/serve-tensorflow-estimator-with-savedmodel/
"""
import numpy as np
from tensorflow.contrib import predictor

from get_started.prepare_data import _int64_feature
from utils import tf_settings
from utils.time_util import timeit


def iris():
    # 从导出目录中加载模型，并生成预测函数。
    model = predictor.from_saved_model(export_dir='/data/models/iris_model/1638781680')

    # 测试数据。
    inputs = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    keys = inputs.keys()
    records = [dict(zip(keys, vals)) for vals in zip(*(inputs[k] for k in keys))]

    # 将输入数据转换成序列化后的 Example 字符串。
    examples = []
    for record in records:
        feature = {}
        for key, val in record.items():
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[val]))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(example.SerializeToString())

    # 开始预测
    predictions = model({'inputs': examples})
    scores = predictions['scores']
    classes = predictions['classes']
    idx = np.argmax(scores, axis=-1)
    for i, s, c in zip(idx, scores, classes):
        print(f'class: {c[i]}, prob: {s[i]:.2%}')


def serialize_example(honor_level):
    feature = {
        'honor_level': _int64_feature([honor_level])
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


@timeit
def main():
    # 从导出目录中加载模型，并生成预测函数。
    model = predictor.from_saved_model(export_dir='/data/models/job_card_model/1638850482')

    # 测试数据。
    inputs = {
        'honor_level': [4]
    }
    keys = inputs.keys()
    records = [dict(zip(keys, vals)) for vals in zip(*(inputs[k] for k in keys))]

    # 将输入数据转换成序列化后的 Example 字符串。
    inputs = [serialize_example(**record) for record in records]
    predict(model, inputs)


@timeit
def predict(model, inputs):
    # 开始预测
    predictions = model({'inputs': inputs})
    scores = predictions['scores']
    classes = predictions['classes']
    idx = np.argmax(scores, axis=-1)
    for i, s, c in zip(idx, scores, classes):
        print(f'class: {c[i]}, prob: {s[i]:.2%}')


if __name__ == '__main__':
    tf = tf_settings()
    main()
