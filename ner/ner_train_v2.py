"""
通用命名实体识别 (NER) TF2 版本
python -m ner.ner_train_v2
"""

import glob
import os.path as osp
import pickle
import sys
from collections import Counter

from absl import app
from tensorflow.keras import preprocessing, optimizers
from tensorflow_addons import text
from tqdm import tqdm

from ner.ner_model_v2 import Context, BiLSTM_CRFModel
from ner.ner_utils import pre_process, post_process
from utils import logger, tf_settings, ensure_dir, delete_dir
from utils.time_util import date_time


def parse_args():
    app.define_help_flags()
    app.flags.DEFINE_boolean('train', False, 'train or validate')
    app.flags.DEFINE_integer('epochs', 4, 'The number of epochs used to train.')
    app.flags.DEFINE_integer('batch_size', 256, 'Batch size for training and evaluation.')
    app.flags.DEFINE_string('root_path', '/data/models/ner', 'root path')
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=out_path,
        config_path=osp.join(out_path, 'config.pkl')
    )


def load_data():
    train_data = [pre_process(line.strip()) for line in open(ctx.train_path, 'r', encoding='utf-8')]
    test_data = [pre_process(line.strip()) for line in open(ctx.test_path, 'r', encoding='utf-8')]

    word_counts = Counter(char[0].lower() for line in train_data for char in line)
    vocab = [w for w, f in word_counts.items() if f > 2]
    vocab.insert(0, '<UNK>')
    vocab.insert(0, '<PAD>')
    tags = sorted(list(set([char[1] for line in train_data for char in line])))
    logger.info(f'tags: {tags}')

    word2idx = dict((w, i) for i, w in enumerate(vocab))
    tag2idx = dict((t, i) for i, t in enumerate(tags))

    train_x, train_y = process_data(train_data, word2idx, tag2idx, max_len=400)
    test_x, test_y = process_data(test_data, word2idx, tag2idx, max_len=400)
    return (train_x, train_y), (test_x, test_y), (vocab, tags)


def process_data(data, word2idx, tag2idx, max_len=None):
    if max_len is None:
        max_len = max(len(line) for line in data)
    x = [[word2idx.get(char[0].lower(), 0) for char in line] for line in data]
    y_chunk = [[tag2idx.get(char[1]) for char in line] for line in data]
    x = preprocessing.sequence.pad_sequences(x, max_len, padding='post')
    y_chunk = preprocessing.sequence.pad_sequences(y_chunk, max_len, padding='post')
    logger.info(f'x shape: {x.shape}, dtype: {x.dtype}, type: {type(x)}')
    logger.info(f'y shape: {y_chunk.shape}, dtype: {y_chunk.dtype}, type: {type(y_chunk)}')
    return x, y_chunk


def process_texts(data, vocab, max_len=None):
    if max_len is None:
        max_len = max(len(line) for line in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(char, 1) for char in line.lower()] for line in data]
    return preprocessing.sequence.pad_sequences(x, max_len, padding='post')


def predict():
    with open(ctx.config_path, 'rb') as inp:
        (vocab, chunk_tags) = pickle.load(inp)
    bilstm_crf_model = BiLSTM_CRFModel(ctx, len(vocab), len(chunk_tags))
    bilstm_crf_model.load_weights(ctx.model_path)
    batch_predict(bilstm_crf_model, vocab, chunk_tags)


def batch_predict(model, vocab, tags, max_len=None):
    predict_texts = ['中华人民共和国国务院总理周恩来在外交部长陈毅，副部长王东的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚']
    eval_x = process_texts(predict_texts, vocab, max_len=max_len)
    decoded_sequences, potentials, sequence_lengths, chain_kernel = model(eval_x, training=False)
    for predict_text, length, sequence in zip(predict_texts, sequence_lengths.numpy(), decoded_sequences.numpy()):
        result_tags = [tags[i] for i in sequence[:length]]
        labeled_str = post_process(predict_text, result_tags)
        logger.info(labeled_str)


def train():
    logger.info(ctx)
    ensure_dir(ctx.config_path)
    (train_x, train_y), (test_x, test_y), (vocab, tags) = load_data()
    with open(ctx.config_path, 'wb') as writer:
        pickle.dump((vocab, tags), writer)

    model = BiLSTM_CRFModel(ctx, vocab_size=len(vocab), tags_size=len(tags))
    model.build(input_shape=(None, ctx.max_len))
    model.summary()
    optimizer = optimizers.Adam(ctx.learning_rate)

    dataset = tf.data.Dataset.from_tensor_slices({
        'x': train_x,
        'y': train_y
    }).shuffle(buffer_size=1024).batch(ctx.batch_size).repeat(ctx.epochs)

    eval_dataset = tf.data.Dataset.from_tensor_slices({
        'x': test_x,
        'y': test_y
    }).batch(ctx.batch_size)

    iteration = 0
    total = len(dataset)
    for batch in tqdm(dataset):
        iteration += 1
        batch_train_x = batch['x']
        batch_train_y = batch['y']
        with tf.GradientTape() as tape:
            [decoded_sequences, potentials, sequence_lengths, chain_kernel] = model(batch_train_x)
            log_likelihood, _ = text.crf_log_likelihood(
                inputs=potentials,
                tag_indices=tf.convert_to_tensor(batch_train_y, dtype=tf.int32),
                sequence_lengths=sequence_lengths,
                transition_params=chain_kernel)
            loss = -tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if iteration % ctx.print_per_batch == 0:
            batch_predict(model, vocab, tags, max_len=ctx.max_len)
            measures, _ = metrics(batch_train_y.numpy(), decoded_sequences.numpy(), sequence_lengths.numpy())
            acc = measures['accuracy']
            val_acc = validate(model, eval_dataset)
            msg = f"batch {iteration}/{total}, loss: {loss:.2f}, accuracy: {acc:.2%}, eval accuracy: {val_acc:.2%}"
            logger.info(msg)


def validate(model, val_dataset):
    correct = 0
    total = 0
    for batch in val_dataset:
        batch_test_x = batch['x']
        batch_test_y = batch['y']
        decoded_sequences, potentials, sequence_lengths, chain_kernel = model(batch_test_x, training=False)
        val_measures, _ = metrics(batch_test_y.numpy(), decoded_sequences.numpy(), sequence_lengths.numpy())
        correct += val_measures['correct']
        total += val_measures['total']
    return 1.0 * correct / total


def metrics(y_true, y_pred, lengths):
    measures = dict()
    correct_label_num = 0
    total_label_num = 0
    for y, y_hat, length in zip(y_true, y_pred, lengths):
        match = [a == b for a, b in zip(y[:length], y_hat[:length])]
        correct_label_num += sum(match)
        total_label_num += length
    measures.update({'correct': correct_label_num})
    measures.update({'total': total_label_num})
    measures.update({'accuracy': 1.0 * correct_label_num / total_label_num})
    res_str = ', '.join([(k + ': %.3f' % v) for k, v in measures.items()])
    return measures, res_str


if __name__ == '__main__':
    ctx = parse_args()
    tf = tf_settings()
    if ctx.train:
        train()
    else:
        predict()
