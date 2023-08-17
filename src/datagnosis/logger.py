# stdlib
import logging
import os
from typing import Any, Callable, NoReturn, TextIO, Union

# third party
from loguru import logger

LOG_FORMAT = "[{time}][{process.id}][{level}] {message}"

logger.remove()
DEFAULT_SINK = "datagnosis_{time}.log"


def remove() -> None:
    """
    This function removes all sinks from the logger.
    """
    logger.remove()


def add(
    sink: Union[None, str, os.PathLike, TextIO, logging.Handler] = None,
    level: str = "ERROR",
) -> None:
    """
    This function adds a sink to the logger.

    Args:
        sink (Union[None, str, os.PathLike, TextIO, logging.Handler], optional): The sink to add. Defaults to None.
        level (str, optional): The level to log at. Defaults to "ERROR".
    """
    sink = DEFAULT_SINK if sink is None else sink
    try:
        logger.add(
            sink=sink,  # pyright: ignore
            format=LOG_FORMAT,
            enqueue=True,
            colorize=False,
            diagnose=True,
            backtrace=True,
            rotation="10 MB",
            retention="1 day",
            level=level,
        )
    except BaseException:
        logger.add(
            sink=sink,
            format=LOG_FORMAT,
            colorize=False,
            diagnose=True,
            backtrace=True,
            level=level,
        )


def traceback_and_raise(e: Any, verbose: bool = False) -> NoReturn:
    """
    This function prints the traceback and raises the exception.

    Args:
        e (Any): The exception to raise.
        verbose (bool, optional): A Flag to indicate whether to print the traceback. Defaults to False.

    Raises:
        e: The exception to raise.

    Returns:
        NoReturn: This function does not return.
    """
    try:
        if verbose:
            logger.opt(lazy=True).exception(e)
        else:
            logger.opt(lazy=True).critical(e)
    except BaseException as ex:
        print("failed to print exception", ex)
    if not issubclass(type(e), Exception):
        e = Exception(e)
    raise e


def create_log_and_print_function(level: str) -> Callable:
    """
    This function creates a log and print function.

    Args:
        level (str): The level to log at.

    Returns:
        Callable: The log and print function.
    """

    def log_and_print(*args: Any, **kwargs: Any) -> None:
        try:
            method = getattr(logger.opt(lazy=True), level, None)
            if method is not None:
                method(*args, **kwargs)
            else:
                logger.debug(*args, **kwargs)
        except BaseException as e:
            msg = f"failed to log exception. {e}"
            try:
                logger.debug(msg)
            except Exception as e:
                print(f"{msg}. {e}")

    return log_and_print


# create_log_and_print_function's called at the relevant log level
def traceback(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="exception")(*args, **kwargs)


def critical(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="critical")(*args, **kwargs)


def error(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="error")(*args, **kwargs)


def warning(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="warning")(*args, **kwargs)


def info(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="info")(*args, **kwargs)


def debug(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="debug")(*args, **kwargs)


def trace(*args: Any, **kwargs: Any) -> None:
    return create_log_and_print_function(level="trace")(*args, **kwargs)
