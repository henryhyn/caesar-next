import os
import platform

import tensorflow

from utils.logger_helper import logger

__all__ = ['tf_settings']


def tf_settings(silent=False):
    tf = tensorflow.compat.v1
    tf_logger = tf.get_logger()
    [tf_logger.removeHandler(h) for h in tf_logger.handlers]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if silent:
        return tf
    logger.info(f'Python Version: {platform.python_version()}')
    logger.info(f'TensorFlow Version: {tf.__version__}')
    return tf
