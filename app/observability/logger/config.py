"""
logger configuration using environment variables.

provides centralized settings for log level, output destinations, and formatting.
follows the project's environment-based configuration pattern without pydantic.
"""

import os
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
LogOutput = Literal["STDOUT", "FILE", "BOTH"]


class LoggerConfig:
    """environment-based logger configuration"""

    def __init__(self):
        # log level: DEBUG, INFO, WARN, ERROR (default: DEBUG)
        self.log_level: LogLevel = os.getenv("LOG_LEVEL", "DEBUG").upper()  # type: ignore

        # output destination: STDOUT, FILE, or BOTH (default: BOTH)
        self.log_output: LogOutput = os.getenv("LOG_OUTPUT", "BOTH").upper()  # type: ignore

        # file output settings
        self.log_dir: str = os.getenv("LOG_DIR", "logs")
        self.log_file_max_bytes: int = int(os.getenv("LOG_FILE_MAX_BYTES", 10485760))  # 10MB default
        self.log_file_backup_count: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", 5))

        # organize logs by pipeline step and session
        self.organize_by_pipeline_step: bool = os.getenv("LOG_ORGANIZE_BY_STEP", "true").lower() == "true"
        self.create_session_folder: bool = os.getenv("LOG_CREATE_SESSION_FOLDER", "true").lower() == "true"

        # format settings
        self.log_format: str = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s | %(levelname)-5s | %(pipeline_step)-20s | %(name)s | %(message)s"
        )
        self.log_date_format: str = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

        # validate log level
        valid_levels = ["DEBUG", "INFO", "WARN", "ERROR"]
        if self.log_level not in valid_levels:
            raise ValueError(f"invalid LOG_LEVEL: {self.log_level}. must be one of {valid_levels}")

        # validate log output
        valid_outputs = ["STDOUT", "FILE", "BOTH"]
        if self.log_output not in valid_outputs:
            raise ValueError(f"invalid LOG_OUTPUT: {self.log_output}. must be one of {valid_outputs}")


def get_logger_config() -> LoggerConfig:
    """get singleton logger configuration instance"""
    return LoggerConfig()
