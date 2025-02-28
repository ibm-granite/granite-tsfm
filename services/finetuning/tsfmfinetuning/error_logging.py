import logging
import os


def write_termination_log(text, log_file="error.log"):
    """Writes text to termination log.

    Args:
        text: str
        log_file: Optional[str]
    """
    log_file = os.environ.get("TERMINATION_LOG_FILE", log_file)
    try:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(text)
    except Exception as e:  # pylint: disable=broad-except
        logging.warning(f"Unable to write termination log due to {e}")
