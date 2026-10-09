"""Unit tests for mesa_logging."""

import logging

import pytest

from mesa import mesa_logging


@pytest.fixture(autouse=True)
def tear_down():
    """Pytest fixture to ensure all logging state is reset after testing."""
    yield
    mesa_logging._logger = None
    mesa_logging._rootlogger = None
    mesa_logger = logging.getLogger(mesa_logging.LOGGER_NAME)
    mesa_logger.handlers = []
    mesa_logger.propagate = True


def test_get_logger():
    """Test get_logger."""
    mesa_logging._rootlogger = None
    logger = mesa_logging.get_rootlogger()
    assert logger == logging.getLogger(mesa_logging.LOGGER_NAME)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)

    logger = mesa_logging.get_rootlogger()
    assert logger, logging.getLogger(mesa_logging.LOGGER_NAME)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_log_to_stderr():
    """Test log_to_stderr."""
    mesa_logging._rootlogger = None
    logger = mesa_logging.log_to_stderr(mesa_logging.DEBUG)
    assert len(logger.handlers) == 2
    assert logger.level == mesa_logging.DEBUG

    mesa_logging._rootlogger = None
    logger = mesa_logging.log_to_stderr()
    assert len(logger.handlers) == 2
    assert logger.level == mesa_logging.DEFAULT_LEVEL

    logger = mesa_logging.log_to_stderr()
    assert len(logger.handlers) == 2
    assert logger.level == mesa_logging.DEFAULT_LEVEL


def test_log_to_stderr_propagation():
    """Test log_to_stderr with propagate parameter."""
    mesa_logging._rootlogger = None
    logger = mesa_logging.log_to_stderr(propagate=True)
    assert logger.propagate is True

    mesa_logging._rootlogger = None
    logger = mesa_logging.log_to_stderr(propagate=False)
    assert logger.propagate is False


def test_caplog_capture_after_teardown(caplog):
    """Verify that logger propagation allows caplog to capture records."""
    logger = mesa_logging.get_module_logger("test_propagation")
    logger.setLevel(logging.INFO)
    logger.info("testing caplog propagation")
    assert "testing caplog propagation" in caplog.text
