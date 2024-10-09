from enum import IntEnum

from fastapi import HTTPException
from starlette import status


class ErrorType(IntEnum):
    TYPE_ERROR = 100
    # leave room between codes for sub-categories
    SOMETHING_ELSE = 200


US_ENG_MESSAGES = {
    ErrorType.TYPE_ERROR: "Invalid data type, if you specified a timestamp column, confirm that all elements are convertable to a datetime type. For other columns like targets or features, confirm that they are numeric."
}

_messages = US_ENG_MESSAGES  # later localization if necessary

EXCEPTION_TABLE = {
    ErrorType.TYPE_ERROR: HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"TSFM_ERROR:{ErrorType.TYPE_ERROR.value}:{_messages[ErrorType.TYPE_ERROR]}",
    )
}
