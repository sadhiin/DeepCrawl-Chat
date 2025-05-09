import logging
import os
from datetime import datetime


def create_logger(name="deepcrawl_chat", filename=f"{datetime.now().strftime('%Y-%m-%d')}.log"):
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger