import logging

filename = 'test_log'
logger = logging.getLogger(__name__)

f_handler = logging.FileHandler(filename)
f_handler.setLevel(logging.DEBUG())
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.WARNING)

f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_format = logging.Formatter('%(name)s - %(levelname) - %(message)s')
f_handler.setFormatter(f_format)
c_handler.setFormatter(c_format)

logger.addHandler(f_handler)
logger.addHandler(c_handler)