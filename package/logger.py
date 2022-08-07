import logging.handlers
import sys
import warnings
from pathlib import Path
from typing import Union, Optional


class AdvancedLoggerManager(logging.Logger):
    """
    Extended logger, supports automatic log file configuration, used RAM logging
    """
    SYS_LOG_ADDRESS = '/dev/log'
    MAX_BYTES = 100 * 1024 * 1024
    BACKUP_COUNT = 10
    LOG_NAME_LEVEL = 2

    def __init__(
            self,
            logger_path: Union[str, Path],
            logger_name: Optional[str] = None,
            debug_enabled=True,
            sys_log=False,
            raise_on_error=False):
        """
        Parameters
        ----------
        logger_path                 path of the logger
        sentry_dsn                  web address for sentry handler
        logger_name                 name of the logger
        debug_enabled               include debug messages in the output
        sys_log                     add sys_log handler
        override_sys_exceptions     intercept and log system exceptions
        raise_on_error              raise an exception if failed to create log file
        """
        super().__init__(str(logger_path))

        self.logger_path = logger_path
        self.logger_name = logger_name
        self.debug_enabled = debug_enabled
        self.sys_log = sys_log
        self.raise_on_error = raise_on_error

        self._init_logger()

    def _init_logger(self) -> logging.Logger:
        logger_path = Path(self.logger_path).resolve()

        logger = self
        logger.setLevel(logging.DEBUG if self.debug_enabled else logging.INFO)

        handlers = list()

        try:
            logfile = logging.handlers.RotatingFileHandler(
                logger_path,
                maxBytes=self.MAX_BYTES,
                backupCount=self.BACKUP_COUNT,
            )
            logfile.setLevel(logging.DEBUG if self.debug_enabled
                             else logging.INFO)
            handlers.append(logfile)

        except OSError:
            if self.raise_on_error:
                raise
            else:
                warnings.warn(f'Failed to setup file logger at:\n{logger_path}')

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.DEBUG if self.debug_enabled
                                else logging.ERROR)
        handlers.append(stream_handler)

        if self.sys_log:
            syslog_handler = logging.handlers.SysLogHandler(
                address=self.SYS_LOG_ADDRESS
            )
            syslog_handler.setLevel(logging.DEBUG if self.debug_enabled
                                    else logging.INFO)
            handlers.append(syslog_handler)

        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        logger.handlers = list()

        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self.logger = self._init_logger()
