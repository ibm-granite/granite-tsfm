from enum import IntEnum


class ErrorType(IntEnum):
    TYPE_ERROR = 100
    INCORRECT_PARAMETER_ERROR = 101
    # leave room between codes for sub-categories
    UNKNOWN_ERROR = 200


_ex2ErrorType = {TypeError: ErrorType.TYPE_ERROR, ValueError: ErrorType.INCORRECT_PARAMETER_ERROR}


US_ENG_MESSAGES = {
    ErrorType.TYPE_ERROR: "Invalid data type, if you specified a timestamp column, confirm that all elements are convertable to a datetime type. For other columns like targets or features, confirm that they are numeric.",
    ErrorType.INCORRECT_PARAMETER_ERROR: "A specified parameter is invalid.",
    ErrorType.UNKNOWN_ERROR: "Unknown error.",
}

MESSAGES = US_ENG_MESSAGES


def error_message(ex: Exception) -> ErrorType:
    for k, v in _ex2ErrorType.items():
        if isinstance(ex, k):
            return f"{MESSAGES[v]}:{str(ex)}"
    return str(ex)
