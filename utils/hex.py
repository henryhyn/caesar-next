import os
import platform

import tensorflow as tf

from utils.logger_helper import logger

__all__ = ['tf_settings']


def tf_settings(silent=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if silent:
        return
    logger.info(f'Python Version: {platform.python_version()}')
    logger.info(f'TensorFlow Version: {tf.__version__}')
