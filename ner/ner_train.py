"""
通用命名实体识别 (NER) TF1 版本
python -m ner.ner_train
"""
import os.path as osp
import pickle
import shutil
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
from absl import app
from keras import preprocessing, layers, models, optimizers
from keras_contrib.layers import CRF

from ner.ner_utils import pre_process, post_process
from utils import tf_settings, logger, ensure_dir


@dataclass
class Context:
    train: bool
    batch_size: int
    epochs: int
    epochs_between_evals: int
    embedding_size: int = 64
    bi_rnn_units: int = 100
    model_dir: str = '/data/models/ner'
    model_path: str = '/data/models/ner/crf.h5'
    train_path: str = '/data/warehouse/ner/train_data.data'
    test_path: str = '/data/warehouse/ner/test_data.data'


def parse_args():
    app.define_help_flags()
    app.flags.DEFINE_boolean('train', False, 'train or validate')
    app.flags.DEFINE_integer('batch_size', 256, 'Batch size for training and evaluation.')
    app.flags.DEFINE_integer('epochs', 4, 'The number of epochs used to train.')
    app.flags.DEFINE_integer('epochs_between_evals', 2, 'The number of training epochs to run between evaluations.')
    app.parse_flags_with_usage(sys.argv)
    args = app.FLAGS
    return Context(
        train=args.train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        epochs_between_evals=args.epochs_between_evals
    )


def load_data():
    train_data = [pre_process(line.strip()) for line in open(ctx.train_path, 'r', encoding='utf-8')]
    test_data = [pre_process(line.strip()) for line in open(ctx.test_path, 'r', encoding='utf-8')]

    word_counts = Counter(char[0].lower() for line in train_data for char in line)
    vocab = [w for w, f in word_counts.items() if f > 2]
    vocab.insert(0, '<unk>')
    tags = sorted(list(set([char[1] for line in train_data for char in line])))
    logger.info(f'tags: {tags}')

    word2idx = dict((w, i) for i, w in enumerate(vocab))
    tag2idx = dict((t, i) for i, t in enumerate(tags))

    train_x, train_y = process_data(train_data, word2idx, tag2idx, max_len=400)
    test_x, test_y = process_data(test_data, word2idx, tag2idx, max_len=400)
    return (train_x, train_y), (test_x, test_y), (vocab, tags)


def process_data(data, word2idx, tag2idx, max_len=None, one_hot=False):
    if max_len is None:
        max_len = max(len(line) for line in data)
    x = [[word2idx.get(char[0].lower(), 0) for char in line] for line in data]
    y_chunk = [[tag2idx.get(char[1]) for char in line] for line in data]
    x = preprocessing.sequence.pad_sequences(x, max_len)
    y_chunk = preprocessing.sequence.pad_sequences(y_chunk, max_len, value=-1)
    if one_hot:
        y_chunk = np.eye(len(tag2idx), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, -1)
    logger.info(f'x shape: {x.shape}, dtype: {x.dtype}, type: {type(x)}')
    logger.info(f'y shape: {y_chunk.shape}, dtype: {y_chunk.dtype}, type: {type(y_chunk)}')
    return x, y_chunk


def process_line(line, word2idx, max_len=100):
    x = [word2idx.get(char[0].lower(), 0) for char in line]
    length = len(x)
    x = preprocessing.sequence.pad_sequences([x], max_len)
    return x, length


def build_model(input_dim, crf_units):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim, ctx.embedding_size, mask_zero=True))
    model.add(layers.Bidirectional(layers.LSTM(ctx.bi_rnn_units // 2, return_sequences=True)))
    crf = CRF(crf_units, sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile(optimizer=optimizers.Adam(), loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def train():
    logger.info(ctx)
    shutil.rmtree(ctx.model_dir, ignore_errors=True)
    ensure_dir(ctx.model_path)
    (train_x, train_y), (test_x, test_y), (vocab, tags) = load_data()
    with open(osp.join(ctx.model_dir, 'config.pkl'), 'wb') as writer:
        pickle.dump((vocab, tags), writer)
    model = build_model(input_dim=len(vocab), crf_units=len(tags))
    model.fit(train_x, train_y, batch_size=ctx.batch_size, epochs=ctx.epochs, validation_data=[test_x, test_y])
    model.save(ctx.model_path)


def validate():
    with open(osp.join(ctx.model_dir, 'config.pkl'), 'rb') as reader:
        (vocab, tags) = pickle.load(reader)
    model = build_model(input_dim=len(vocab), crf_units=len(tags))
    model.load_weights(ctx.model_path)
    predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅，副部长王东的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    chars, length = process_line(predict_text, word2idx)
    raw = model.predict(chars)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [tags[i] for i in result]
    labeled_str = post_process(predict_text, result_tags)
    logger.info(labeled_str)


if __name__ == '__main__':
    ctx = parse_args()
    tf = tf_settings()
    if ctx.train:
        train()
    else:
        validate()
