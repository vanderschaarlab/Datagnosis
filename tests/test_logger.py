# datagnosis absolute
import datagnosis.logger as log


def test_logger_sanity() -> None:
    assert log is not None
    assert log.__name__ == "datagnosis.logger"
    assert log.critical is not None
    assert log.error is not None
    assert log.warning is not None
    assert log.info is not None
    assert log.debug is not None
    assert log.trace is not None
    assert log.traceback is not None
    assert log.traceback_and_raise is not None
