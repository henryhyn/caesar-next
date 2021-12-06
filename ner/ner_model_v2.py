from abc import ABC
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow_addons.layers.crf import CRF


@dataclass
class Context:
    train: bool = False
    epochs: int = 10
    batch_size: int = 256
    embedding_dim: int = 64
    dropout_rate: float = 0.5
    learning_rate: float = 0.001
    bi_rnn_units: int = 100
    max_len: int = 400
    print_per_batch: int = 2
    model_path: str = ''
    config_path: str = ''
    train_path: str = '/data/warehouse/ner/train_data.data'
    test_path: str = '/data/warehouse/ner/test_data.data'


class BiLSTM_CRFModel(models.Model, ABC):
    def __init__(self, ctx: Context, vocab_size, tags_size):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=ctx.embedding_dim, mask_zero=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(units=ctx.bi_rnn_units // 2, return_sequences=True))
        self.dense = layers.Dense(units=tags_size)
        self.crf = CRF(units=tags_size)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        inputs = self.embedding(inputs)
        inputs = self.bilstm(inputs)
        inputs = self.dense(inputs)
        return self.crf(inputs)
