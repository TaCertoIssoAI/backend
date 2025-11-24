"""
tests for the centralized logging system.

verifies logger configuration, pipeline step context, output destinations,
and log level filtering.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.observability.logger import (
    PipelineStep,
    get_logger,
    get_logger_config,
    get_request_logger,
    setup_logging,
)
from app.observability.logger.config import LoggerConfig


class TestLoggerConfig:
    """test logger configuration from environment variables"""

    def test_default_config(self):
        """test default configuration values"""
        with patch.dict(os.environ, {}, clear=True):
            config = LoggerConfig()

            assert config.log_level == "DEBUG"
            assert config.log_output == "BOTH"  # default is now BOTH
            assert config.log_dir == "logs"
            assert config.log_file_max_bytes == 10485760  # 10MB
            assert config.log_file_backup_count == 5
            assert config.organize_by_pipeline_step is True
            assert config.create_session_folder is True

    def test_custom_config_from_env(self):
        """test configuration from environment variables"""
        env_vars = {
            "LOG_LEVEL": "DEBUG",
            "LOG_OUTPUT": "FILE",
            "LOG_DIR": "/tmp/custom_logs",
            "LOG_FILE_MAX_BYTES": "5242880",  # 5MB
            "LOG_FILE_BACKUP_COUNT": "3",
            "LOG_ORGANIZE_BY_STEP": "false",
            "LOG_CREATE_SESSION_FOLDER": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = LoggerConfig()

            assert config.log_level == "DEBUG"
            assert config.log_output == "FILE"
            assert config.log_dir == "/tmp/custom_logs"
            assert config.log_file_max_bytes == 5242880
            assert config.log_file_backup_count == 3
            assert config.organize_by_pipeline_step is False
            assert config.create_session_folder is False

    def test_invalid_log_level_raises_error(self):
        """test that invalid log level raises ValueError"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}, clear=True):
            with pytest.raises(ValueError, match="invalid LOG_LEVEL"):
                LoggerConfig()

    def test_invalid_log_output_raises_error(self):
        """test that invalid log output raises ValueError"""
        with patch.dict(os.environ, {"LOG_OUTPUT": "INVALID"}, clear=True):
            with pytest.raises(ValueError, match="invalid LOG_OUTPUT"):
                LoggerConfig()

    def test_case_insensitive_config(self):
        """test that config values are case-insensitive"""
        env_vars = {
            "LOG_LEVEL": "debug",
            "LOG_OUTPUT": "both",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = LoggerConfig()

            assert config.log_level == "DEBUG"
            assert config.log_output == "BOTH"


class TestPipelineStep:
    """test pipeline step enumeration"""

    def test_pipeline_step_values(self):
        """test that all expected pipeline steps exist"""
        assert PipelineStep.LINK_EXPANSION.value == "link_expansion"
        assert PipelineStep.CLAIM_EXTRACTION.value == "claim_extraction"
        assert PipelineStep.EVIDENCE_RETRIEVAL.value == "evidence_retrieval"
        assert PipelineStep.ADJUDICATION.value == "adjudication"
        assert PipelineStep.API_INTAKE.value == "api_intake"
        assert PipelineStep.UNKNOWN.value == "unknown"

    def test_pipeline_step_string_conversion(self):
        """test that pipeline steps convert to strings correctly"""
        step = PipelineStep.CLAIM_EXTRACTION
        assert str(step) == "claim_extraction"


class TestLoggerFactory:
    """test logger factory functions"""

    def setup_method(self):
        """reset logging state before each test"""
        # reset global initialization flag
        import app.observability.logger.logger as logger_module
        logger_module._logging_initialized = False

        # clear all handlers from root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def test_get_logger_with_pipeline_step(self):
        """test getting logger with pipeline step context"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)

            assert logger.pipeline_step == PipelineStep.CLAIM_EXTRACTION
            assert logger.logger.name == __name__

    def test_get_logger_default_pipeline_step(self):
        """test that logger defaults to UNKNOWN pipeline step"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__)

            assert logger.pipeline_step == PipelineStep.UNKNOWN

    def test_get_request_logger_with_request_id(self):
        """test getting logger with request id context"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_request_logger(
                __name__,
                PipelineStep.EVIDENCE_RETRIEVAL,
                request_id="req-123"
            )

            assert logger.pipeline_step == PipelineStep.EVIDENCE_RETRIEVAL
            assert "req-123" in logger.logger.name

    def test_setup_logging_is_idempotent(self):
        """test that setup_logging can be called multiple times safely"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            setup_logging()
            initial_handler_count = len(logging.getLogger().handlers)

            setup_logging()
            final_handler_count = len(logging.getLogger().handlers)

            assert initial_handler_count == final_handler_count

    def test_logger_outputs_to_stdout(self, capsys):
        """test that logger outputs to stdout when configured"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)
            logger.info("test message")

            captured = capsys.readouterr()
            assert "test message" in captured.err
            assert "claim_extraction" in captured.err

    def test_logger_outputs_to_file(self):
        """test that logger outputs to file when configured"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "INFO",
                "LOG_OUTPUT": "FILE",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "true",
                "LOG_CREATE_SESSION_FOLDER": "false",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                logger = get_logger(__name__, PipelineStep.EVIDENCE_RETRIEVAL)
                logger.info("file test message")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # check that file was created in evidence_retrieval.log
                log_file = Path(tmp_dir) / "evidence_retrieval.log"
                assert log_file.exists()
                content = log_file.read_text()
                assert "file test message" in content
                assert "evidence_retrieval" in content

    def test_logger_outputs_to_both(self, capsys):
        """test that logger outputs to both stdout and file when configured"""
        # reset global state before this test
        import app.observability.logger.logger as logger_module
        logger_module._logging_initialized = False
        logger_module._session_timestamp = None
        logger_module._pipeline_handlers.clear()

        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "INFO",
                "LOG_OUTPUT": "BOTH",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "true",
                "LOG_CREATE_SESSION_FOLDER": "false",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                logger = get_logger(__name__, PipelineStep.ADJUDICATION)
                logger.info("both output test")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # check stdout
                captured = capsys.readouterr()
                assert "both output test" in captured.err

                # check file
                log_file = Path(tmp_dir) / "adjudication.log"
                assert log_file.exists()
                content = log_file.read_text()
                assert "both output test" in content

    def test_log_level_filtering(self, capsys):
        """test that log levels are filtered correctly"""
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.SYSTEM)

            logger.debug("debug message")
            logger.info("info message")
            logger.warning("warning message")
            logger.error("error message")

            captured = capsys.readouterr()

            # only error should appear
            assert "debug message" not in captured.err
            assert "info message" not in captured.err
            assert "warning message" not in captured.err
            assert "error message" in captured.err

    def test_logger_with_exception_info(self, capsys):
        """test that logger can log exception info"""
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.WEB_SCRAPING)

            try:
                raise ValueError("test exception")
            except ValueError:
                logger.error("caught exception", exc_info=True)

            captured = capsys.readouterr()
            assert "caught exception" in captured.err
            assert "ValueError" in captured.err
            assert "test exception" in captured.err


class TestLoggerIntegration:
    """integration tests for logger usage in pipeline context"""

    def setup_method(self):
        """reset logging state before each test"""
        import app.observability.logger.logger as logger_module
        logger_module._logging_initialized = False
        logger_module._session_timestamp = None
        logger_module._pipeline_handlers.clear()

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def test_multiple_loggers_different_steps(self, capsys):
        """test that multiple loggers with different steps work correctly"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger1 = get_logger("module1", PipelineStep.LINK_EXPANSION)
            logger2 = get_logger("module2", PipelineStep.CLAIM_EXTRACTION)

            logger1.info("expanding links")
            logger2.info("extracting claims")

            captured = capsys.readouterr()

            assert "link_expansion" in captured.err
            assert "expanding links" in captured.err
            assert "claim_extraction" in captured.err
            assert "extracting claims" in captured.err

    def test_logger_format_includes_all_fields(self, capsys):
        """test that log format includes all expected fields"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger("test.module", PipelineStep.EVIDENCE_RETRIEVAL)
            logger.info("test format")

            captured = capsys.readouterr()
            output = captured.err

            # should include timestamp, level, pipeline step, module name, message
            assert "INFO" in output
            assert "evidence_retrieval" in output
            assert "test.module" in output
            assert "test format" in output


class TestPipelineStepFileOrganization:
    """test file organization by pipeline step and session"""

    def setup_method(self):
        """reset logging state before each test"""
        import app.observability.logger.logger as logger_module
        logger_module._logging_initialized = False
        logger_module._session_timestamp = None
        logger_module._pipeline_handlers.clear()

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def test_session_folder_creation(self):
        """test that session folder is created with timestamp"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "INFO",
                "LOG_OUTPUT": "FILE",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "true",
                "LOG_CREATE_SESSION_FOLDER": "true",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                from app.observability.logger import get_session_log_dir

                logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)
                logger.info("test message")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # check session folder exists
                session_dir = get_session_log_dir()
                assert session_dir is not None
                assert session_dir.exists()

                # verify folder name has timestamp format
                folder_name = session_dir.name
                assert len(folder_name.split("_")) == 2  # date_time
                assert len(folder_name.split("-")) == 5  # YYYY-MM-DD_HH-MM-SS

    def test_separate_files_per_pipeline_step(self):
        """test that logs are written to separate files per pipeline step"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "INFO",
                "LOG_OUTPUT": "FILE",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "true",
                "LOG_CREATE_SESSION_FOLDER": "false",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                logger1 = get_logger("module1", PipelineStep.CLAIM_EXTRACTION)
                logger2 = get_logger("module2", PipelineStep.EVIDENCE_RETRIEVAL)
                logger3 = get_logger("module3", PipelineStep.ADJUDICATION)

                logger1.info("claim message")
                logger2.info("evidence message")
                logger3.info("adjudication message")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # check files exist
                log_dir = Path(tmp_dir)
                claim_file = log_dir / "claim_extraction.log"
                evidence_file = log_dir / "evidence_retrieval.log"
                adjudication_file = log_dir / "adjudication.log"

                assert claim_file.exists()
                assert evidence_file.exists()
                assert adjudication_file.exists()

                # verify content
                assert "claim message" in claim_file.read_text()
                assert "claim message" not in evidence_file.read_text()
                assert "claim message" not in adjudication_file.read_text()

                assert "evidence message" in evidence_file.read_text()
                assert "evidence message" not in claim_file.read_text()

                assert "adjudication message" in adjudication_file.read_text()
                assert "adjudication message" not in claim_file.read_text()

    def test_logs_filtered_by_pipeline_step(self):
        """test that logs are properly filtered to their respective files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "DEBUG",
                "LOG_OUTPUT": "FILE",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "true",
                "LOG_CREATE_SESSION_FOLDER": "false",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                link_logger = get_logger("link_module", PipelineStep.LINK_EXPANSION)
                claim_logger = get_logger("claim_module", PipelineStep.CLAIM_EXTRACTION)

                # log multiple messages
                link_logger.debug("link debug")
                link_logger.info("link info")
                claim_logger.info("claim info")
                claim_logger.warning("claim warning")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # verify link_expansion.log only has link messages
                link_file = Path(tmp_dir) / "link_expansion.log"
                link_content = link_file.read_text()

                assert "link debug" in link_content
                assert "link info" in link_content
                assert "claim info" not in link_content
                assert "claim warning" not in link_content

                # verify claim_extraction.log only has claim messages
                claim_file = Path(tmp_dir) / "claim_extraction.log"
                claim_content = claim_file.read_text()

                assert "claim info" in claim_content
                assert "claim warning" in claim_content
                assert "link debug" not in claim_content
                assert "link info" not in claim_content

    def test_session_folder_with_pipeline_steps(self):
        """test combination of session folder and pipeline step files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "INFO",
                "LOG_OUTPUT": "FILE",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "true",
                "LOG_CREATE_SESSION_FOLDER": "true",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                from app.observability.logger import get_session_log_dir

                logger = get_logger(__name__, PipelineStep.SYSTEM)
                logger.info("system message")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # check structure: logs/{timestamp}/system.log
                session_dir = get_session_log_dir()
                assert session_dir is not None

                system_file = session_dir / "system.log"
                assert system_file.exists()
                assert "system message" in system_file.read_text()

    def test_disable_organize_by_step(self):
        """test that single log file is created when organize_by_step is disabled"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "INFO",
                "LOG_OUTPUT": "FILE",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "false",
                "LOG_CREATE_SESSION_FOLDER": "false",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                logger1 = get_logger("module1", PipelineStep.CLAIM_EXTRACTION)
                logger2 = get_logger("module2", PipelineStep.EVIDENCE_RETRIEVAL)

                logger1.info("claim message")
                logger2.info("evidence message")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # check that app.log exists and contains both messages
                app_file = Path(tmp_dir) / "app.log"
                assert app_file.exists()

                content = app_file.read_text()
                assert "claim message" in content
                assert "evidence message" in content

                # check that separate files do NOT exist
                assert not (Path(tmp_dir) / "claim_extraction.log").exists()
                assert not (Path(tmp_dir) / "evidence_retrieval.log").exists()


class TestLoggerPrefix:
    """test logger prefix functionality"""

    def setup_method(self):
        """reset logging state before each test"""
        import app.observability.logger.logger as logger_module
        logger_module._logging_initialized = False
        logger_module._session_timestamp = None
        logger_module._pipeline_handlers.clear()

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def test_set_prefix(self, capsys):
        """test setting a prefix on log messages"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)

            # set prefix
            logger.set_prefix("[req-123]")
            logger.info("processing claim")

            captured = capsys.readouterr()
            assert "[req-123] processing claim" in captured.err

    def test_clear_prefix(self, capsys):
        """test clearing a prefix"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.CLAIM_EXTRACTION)

            # set and use prefix
            logger.set_prefix("[batch-5]")
            logger.info("batch started")

            # clear prefix
            logger.clear_prefix()
            logger.info("batch completed")

            captured = capsys.readouterr()
            assert "[batch-5] batch started" in captured.err
            assert "[batch-5] batch completed" not in captured.err
            assert "batch completed" in captured.err

    def test_prefix_with_file_output(self):
        """test that prefix works with file output"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = {
                "LOG_LEVEL": "INFO",
                "LOG_OUTPUT": "FILE",
                "LOG_DIR": tmp_dir,
                "LOG_ORGANIZE_BY_STEP": "true",
                "LOG_CREATE_SESSION_FOLDER": "false",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                logger = get_logger(__name__, PipelineStep.EVIDENCE_RETRIEVAL)

                logger.set_prefix("[user-456]")
                logger.info("gathering evidence")
                logger.info("found 5 citations")

                logger.clear_prefix()
                logger.info("completed")

                # force flush
                for handler in logging.getLogger().handlers:
                    handler.flush()

                log_file = Path(tmp_dir) / "evidence_retrieval.log"
                content = log_file.read_text()

                assert "[user-456] gathering evidence" in content
                assert "[user-456] found 5 citations" in content
                assert "[user-456] completed" not in content
                assert "completed" in content

    def test_multiple_prefix_changes(self, capsys):
        """test changing prefix multiple times"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.SYSTEM)

            logger.set_prefix("[op-1]")
            logger.info("operation 1")

            logger.set_prefix("[op-2]")
            logger.info("operation 2")

            logger.clear_prefix()
            logger.info("operation 3")

            captured = capsys.readouterr()
            assert "[op-1] operation 1" in captured.err
            assert "[op-2] operation 2" in captured.err
            assert "operation 3" in captured.err
            assert "[op-" not in captured.err.split("operation 3")[0].split("operation 2")[1]

    def test_prefix_with_different_log_levels(self, capsys):
        """test that prefix works with all log levels"""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG", "LOG_OUTPUT": "STDOUT"}, clear=True):
            logger = get_logger(__name__, PipelineStep.LINK_EXPANSION)

            logger.set_prefix("[ctx]")
            logger.debug("debug message")
            logger.info("info message")
            logger.warning("warning message")
            logger.error("error message")

            captured = capsys.readouterr()
            assert "[ctx] debug message" in captured.err
            assert "[ctx] info message" in captured.err
            assert "[ctx] warning message" in captured.err
            assert "[ctx] error message" in captured.err
