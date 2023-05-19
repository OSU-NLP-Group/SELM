import logging


def init(name: str, verbose: bool = False, date=True) -> logging.Logger:
    if date:
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    else:
        log_format = "[%(levelname)s] [%(name)s] %(message)s"

    if not verbose:
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    return logging.getLogger(name)
