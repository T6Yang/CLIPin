import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

from torch.multiprocessing import Queue

logging.raiseExceptions = False

def setup_primary_logging(log_file, level):
    log_queue = Queue(-1)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%Y-%m-%d,%H:%M:%S')

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    listener = QueueListener(log_queue, file_handler, stream_handler)

    listener.start()

    return log_queue


class WorkerLogFilter(Filter):
    def __init__(self):
        super().__init__()

    def filter(self, record):
        record.msg = f"{record.msg}"
        return True


def setup_worker_logging(log_queue, level):
    queue_handler = QueueHandler(log_queue)

    worker_filter = WorkerLogFilter()
    queue_handler.addFilter(worker_filter)

    queue_handler.setLevel(level)

    root_logger = logging.getLogger()
    if len(root_logger.handlers) > 0:
        root_logger.removeHandler(root_logger.handlers[0])
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(level)
