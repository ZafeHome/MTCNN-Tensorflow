import logging

def splunkable_log(message='', **kwargs):
    extra = ' '.join(['{0}="{1}"'.format(k,v) for k,v in kwargs.items()])
    log = 'message="{}"'.format(message)
    if extra:
        log += " " + extra
    return log


class SentinelLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def critical(self, message='', **kwargs):
        self.logger.critical(splunkable_log(message=message, **kwargs))

    def error(self, message='', **kwargs):
        self.logger.error(splunkable_log(message=message, **kwargs))

    def warning(self, message='', **kwargs):
        self.logger.warning(splunkable_log(message=message, **kwargs))

    def info(self, message='', **kwargs):
        self.logger.info(splunkable_log(message=message, **kwargs))

    def debug(self, message='', **kwargs):
        self.logger.debug(splunkable_log(message=message, **kwargs))

    def exception(self, message='', **kwargs):
        self.logger.exception(splunkable_log(message=message, **kwargs))

    def addHandler(self, handler):
        self.logger.addHandler(handler)

    def setLevel(self, logging_level):
        self.logger.setLevel(logging_level)
