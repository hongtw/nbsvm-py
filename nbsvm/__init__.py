import logging
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
ch.setFormatter(formatter)
logging.getLogger('nbsvm_logger').addHandler(ch)
logging.getLogger('nbsvm_logger').setLevel(logging.INFO)

from .nbsvm import NBSVM

