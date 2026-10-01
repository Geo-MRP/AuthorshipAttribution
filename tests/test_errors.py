from __future__ import annotations

from authorship_attribution.errors import (
    ContractValidationError,
    ErrorCode,
    error_code_for,
)


def test_error_code_has_exact_values() -> None:
    assert {code.value for code in ErrorCode} == {
        "MINIMUMS_NOT_MET",
        "ENGINE_TIMEOUT",
        "ENGINE_FAILURE",
        "BASELINE_CONFIG_MISMATCH",
        "BASELINE_INVALID",
        "LOCAL_MODEL_UNAVAILABLE",
        "LOCAL_MODEL_INVALID",
        "EXTERNAL_NOT_ALLOWED",
        "INPUT_INVALID",
        "ARTIFACT_INVALID",
    }


def test_validation_error_is_stable_and_privacy_safe() -> None:
    error = ContractValidationError("text", ErrorCode.INPUT_INVALID)

    assert error.field_name == "text"
    assert error.code is ErrorCode.INPUT_INVALID
    assert str(error) == "text:INPUT_INVALID"


def test_error_code_for_uses_code_attribute_without_messages() -> None:
    class CodedFailure(Exception):
        code = ErrorCode.ENGINE_TIMEOUT

        def __str__(self) -> str:
            raise AssertionError("error_code_for must not inspect this")

    assert error_code_for(ContractValidationError("field", ErrorCode.BASELINE_INVALID)) is ErrorCode.BASELINE_INVALID
    assert error_code_for(CodedFailure()) is ErrorCode.ENGINE_TIMEOUT
    assert error_code_for(RuntimeError("private value")) is ErrorCode.ENGINE_FAILURE


def test_error_code_for_handles_hostile_code_properties() -> None:
    class HostileFailure(Exception):
        @property
        def code(self) -> ErrorCode:
            raise KeyboardInterrupt

    assert error_code_for(HostileFailure()) is ErrorCode.ENGINE_FAILURE
