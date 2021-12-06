import time
from functools import wraps

from utils.logger_helper import logger


def date_time():
    return time.strftime('%Y%m%d%H%M%S', time.localtime())


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logger.info('execute %s%s => %s (cost: %dms)', func.__name__, args, res, 1000 * (end_time - start_time))
        return res

    return wrapper


def timeit_no_res(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logger.info('execute %s%s (cost: %dms)', func.__name__, args, 1000 * (end_time - start_time))
        return res

    return wrapper


def timeit_no_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logger.info('execute %s => %s (cost: %dms)', func.__name__, res, 1000 * (end_time - start_time))
        return res

    return wrapper
