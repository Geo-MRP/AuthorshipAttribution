"""Stable error codes and privacy-safe contract validation errors."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """The complete set of machine-readable failure categories."""

    MINIMUMS_NOT_MET = "MINIMUMS_NOT_MET"
    ENGINE_TIMEOUT = "ENGINE_TIMEOUT"
    ENGINE_FAILURE = "ENGINE_FAILURE"
    BASELINE_CONFIG_MISMATCH = "BASELINE_CONFIG_MISMATCH"
    BASELINE_INVALID = "BASELINE_INVALID"
    LOCAL_MODEL_UNAVAILABLE = "LOCAL_MODEL_UNAVAILABLE"
    LOCAL_MODEL_INVALID = "LOCAL_MODEL_INVALID"
    EXTERNAL_NOT_ALLOWED = "EXTERNAL_NOT_ALLOWED"
    INPUT_INVALID = "INPUT_INVALID"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"


class ContractValidationError(ValueError):
    """Validation failure containing only a stable field name and error code."""

    field_name: str
    code: ErrorCode

    def __init__(self, field_name: str, code: ErrorCode) -> None:
        self.field_name = field_name
        self.code = code
        super().__init__(field_name, code.value)

    def __str__(self) -> str:
        return f"{self.field_name}:{self.code.value}"


def error_code_for(exc: BaseException) -> ErrorCode:
    """Return a stable code without inspecting exception message text."""

    if isinstance(exc, ContractValidationError):
        return exc.code
    try:
        code = getattr(exc, "code", None)
    except BaseException:
        return ErrorCode.ENGINE_FAILURE
    if isinstance(code, ErrorCode):
        return code
    return ErrorCode.ENGINE_FAILURE
