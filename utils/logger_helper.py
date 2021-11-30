import logging
import os
from logging.handlers import TimedRotatingFileHandler


class GroupWriteTimedRotatingFileHandler(TimedRotatingFileHandler):
    def _open(self):
        prevumask = os.umask(0o002)
        rtv = TimedRotatingFileHandler._open(self)
        os.umask(prevumask)
        return rtv


def getLogger(env, port):
    log_dir = '/data/applogs/ut-caesar'

    fmt = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s (%(filename)s:%(lineno)d) - %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger('nacos').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if env == 'local':
        logger.addHandler(ch)

    fh = GroupWriteTimedRotatingFileHandler(
        '{}/ut-caesar-{}.log'.format(log_dir, port),
        when='D', interval=1, backupCount=30, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    rh = GroupWriteTimedRotatingFileHandler(
        '{}/ut-caesar-{}.error.log'.format(log_dir, port),
        when='D', interval=1, backupCount=30, encoding='utf-8')
    rh.setLevel(logging.ERROR)
    rh.setFormatter(fmt)
    logger.addHandler(rh)

    return logger


logger = getLogger(env='local', port=9090)
