import sys
import os
import logging
from functools import *
import datetime


def log(func):
    @wraps(func)
    def wrap_log(*args, **kwargs):
        name = func.__name__
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler('anomaly.log')
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info('Running function: %s' % name)
        result = func(*args, **kwargs)
        logger.info('Result: %s' % result)
        return result
    return wrap_log
