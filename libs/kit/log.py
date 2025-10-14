import logging
import inspect
import sys

from kit.time import now

LOGGER = None  # python logger
LOG_INDENT_BASE = len(inspect.stack())  # base for logging indentation
START = None
MSG = None


def setup_logger(logging_level=logging.INFO, log_file=None):
    """sets up the python logging system

    :param logging_level: logging level
    """

    global LOGGER

    if LOGGER is None:
        # initialize the logger
        LOGGER = logging.getLogger(__name__)
        LOGGER.propagate = False
        LOGGER.setLevel(logging_level)
        LOGGER.addHandler(logging.StreamHandler(sys.stdout))
        if log_file is not None:
            LOGGER.addHandler(logging.FileHandler(log_file))


def log_info(msg, start=False, stop=False):
    log(msg, start=start, stop=stop, logging_level=logging.INFO)


def log_debug(msg, start=False, stop=False):
    log(msg, start=start, stop=stop, logging_level=logging.DEBUG)


def log(msg, start=False, stop=False, logging_level=logging.INFO):
    if start:
        global START, MSG
        jetzt, START = now(return_obj=True)
        MSG = msg
        msg = f"START: {MSG}"
    elif stop:
        jetzt, _stop = now(return_obj=True)
        total_seconds = (_stop - START).total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        msg = f"STOP: {MSG} ({int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}) {msg}"
    else:
        jetzt = now()
    indentation = f"{get_indent()}"

    if isinstance(msg, list):
        text = f"{jetzt}: {indentation} {msg[0]}"
        if len(msg) > 1:
            len_jetzt = len(jetzt)
            len_indent = len(indentation)
            for nachricht in msg[1:]:
                text += f"\n{' '*len_jetzt}  {' ' * len_indent} {nachricht}"
    else:
        text = f"{jetzt}: {indentation} {msg}"

    if LOGGER is not None:
        if logging_level == logging.INFO:
            LOGGER.info(text)
        elif logging_level == logging.WARNING:
            LOGGER.warning(text)
        elif logging_level == logging.ERROR:
            LOGGER.error(text)
        elif logging_level == logging.DEBUG:
            LOGGER.debug(text)
    else:
        print(text)


def log_warning(msg):
    text = f"{now()}: {get_indent()} WARNING: {msg}"
    if LOGGER is not None:
        LOGGER.warning(text)
    else:
        print(text)


def log_error(msg):
    text = f"{now()}: {get_indent()} ERROR: {msg}"
    if LOGGER is not None:
        LOGGER.error(text)
    else:
        print(text)


def log_caller(obj=None):
    """logs the name of the calling function

    :param obj: object holding the calling function
    """

    c = caller(obj, indent=False)
    log_info(c)


def get_indent():
    """returns the indentation for logging purposes

    :return: indentation as blank spaces string
    """

    return f"{' '*(len(inspect.stack()) - LOG_INDENT_BASE)}"


def caller(obj=None, indent=True):
    """returns the name of the calling function for logging purposes

    :param obj: object holding the calling function
    :return: name of the calling function in log format
    """

    c = (
        f"{obj.__class__.__name__}.{inspect.stack()[1][3]}"
        if obj is not None
        else f"{inspect.stack()[1][3]}"
    )

    if indent:
        idt = get_indent()
        c = f"{idt} {c}"

    return c
