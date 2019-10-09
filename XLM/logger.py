# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os

def create_logger(file=None):
    """
    Create a logger.
    One for console (info) one for file (debug).
    """
    # create log formatter
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler and set level to debug
    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create root logger and set level to debug
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logger.propagate = False
    if file is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
