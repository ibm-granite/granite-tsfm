# Copyright contributors to the TSFM project
#
"""Tests for security validation in service handler module loading"""

import os
import sys
from pathlib import Path

import pytest


# Add services directory to path for testing so we can import as a package
services_path = Path(__file__).parent.parent.parent / "services"
sys.path.insert(0, str(services_path))

from boilerplate.service_handler import (  # noqa: E402
    _validate_handler_class_name,
    _validate_handler_module_path,
)


class TestHandlerModuleValidation:
    """Test suite for handler module path validation"""

    def test_allowed_tsfm_public_module(self):
        """Test that tsfm_public.* modules are allowed"""
        # Should not raise
        _validate_handler_module_path("tsfm_public.toolkit.some_handler")
        _validate_handler_module_path("tsfm_public.models.patchtst_fm")

    def test_allowed_tsfminference_module(self):
        """Test that tsfminference.* modules are allowed"""
        # Should not raise
        _validate_handler_module_path("tsfminference.tsfm_inference_handler")
        _validate_handler_module_path("tsfminference.custom_handler")

    def test_allowed_tsfmfinetuning_module(self):
        """Test that tsfmfinetuning.* modules are allowed"""
        # Should not raise
        _validate_handler_module_path("tsfmfinetuning.tsfm_tuning_handler")
        _validate_handler_module_path("tsfmfinetuning.custom_handler")

    def test_blocked_subprocess_module(self):
        """Test that subprocess module is blocked (PoC attack vector)"""
        with pytest.raises(ValueError, match="Security: Handler module path 'subprocess' is not allowed"):
            _validate_handler_module_path("subprocess")

    def test_blocked_os_module(self):
        """Test that os module is blocked"""
        with pytest.raises(ValueError, match="Security: Handler module path 'os' is not allowed"):
            _validate_handler_module_path("os")

    def test_blocked_arbitrary_module(self):
        """Test that arbitrary modules are blocked"""
        with pytest.raises(ValueError, match="Security: Handler module path 'malicious.module' is not allowed"):
            _validate_handler_module_path("malicious.module")

    def test_blocked_builtin_module(self):
        """Test that builtin modules like sys are blocked"""
        with pytest.raises(ValueError, match="Security: Handler module path 'sys' is not allowed"):
            _validate_handler_module_path("sys")

    def test_error_message_includes_allowed_prefixes(self):
        """Test that error message includes the allowed prefixes"""
        with pytest.raises(ValueError) as exc_info:
            _validate_handler_module_path("attacker.evil")

        error_msg = str(exc_info.value)
        assert "tsfm_public." in error_msg
        assert "tsfminference." in error_msg
        assert "tsfmfinetuning." in error_msg
        assert "TSFM_TRUST_REMOTE_CODE" in error_msg

    def test_case_sensitive_validation(self):
        """Test that validation is case-sensitive"""
        # Uppercase should be blocked
        with pytest.raises(ValueError):
            _validate_handler_module_path("TSFM_PUBLIC.toolkit.handler")

        # Mixed case should be blocked
        with pytest.raises(ValueError):
            _validate_handler_module_path("Tsfm_Public.toolkit.handler")


class TestAdditionalHandlerModules:
    """Test suite for TSFM_ADDITIONAL_HANDLER_MODULES environment variable"""

    def test_additional_modules_via_env_var(self):
        """Test that additional module prefixes can be added via environment variable"""
        env_backup = os.environ.get("TSFM_ADDITIONAL_HANDLER_MODULES")

        try:
            # Set additional allowed modules with proper dot endings
            os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = "mycompany.models.,custom.handlers."

            # Re-import to get fresh configuration
            import importlib

            importlib.reload(sys.modules["boilerplate.service_handler"])
            from boilerplate.service_handler import ALLOWED_HANDLER_MODULE_PREFIXES, _validate_handler_module_path

            # Should include both base and additional prefixes
            assert "tsfm_public." in ALLOWED_HANDLER_MODULE_PREFIXES
            assert "mycompany.models." in ALLOWED_HANDLER_MODULE_PREFIXES
            assert "custom.handlers." in ALLOWED_HANDLER_MODULE_PREFIXES

            # Should allow the additional modules
            _validate_handler_module_path("mycompany.models.forecasting_handler")
            _validate_handler_module_path("custom.handlers.my_handler")

        finally:
            # Restore env var
            if env_backup is not None:
                os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = env_backup
            elif "TSFM_ADDITIONAL_HANDLER_MODULES" in os.environ:
                del os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"]

    def test_additional_modules_must_end_with_dot(self):
        """Test that additional module prefixes must end with a dot"""
        env_backup = os.environ.get("TSFM_ADDITIONAL_HANDLER_MODULES")

        try:
            # Set without dot - should raise ValueError during import
            os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = "mycompany.models"

            import importlib

            with pytest.raises(ValueError, match="must end with a dot"):
                importlib.reload(sys.modules["boilerplate.service_handler"])

        finally:
            if env_backup is not None:
                os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = env_backup
            elif "TSFM_ADDITIONAL_HANDLER_MODULES" in os.environ:
                del os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"]

    def test_mixed_valid_and_invalid_prefixes(self):
        """Test that one invalid prefix causes the entire configuration to fail"""
        env_backup = os.environ.get("TSFM_ADDITIONAL_HANDLER_MODULES")

        try:
            # Mix valid and invalid - should fail
            os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = "mycompany.models.,invalid_no_dot"

            import importlib

            with pytest.raises(ValueError, match="must end with a dot"):
                importlib.reload(sys.modules["boilerplate.service_handler"])

        finally:
            if env_backup is not None:
                os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = env_backup
            elif "TSFM_ADDITIONAL_HANDLER_MODULES" in os.environ:
                del os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"]

    def test_dot_enforcement_prevents_broad_matches(self):
        """Test that dot enforcement prevents overly broad module matching"""
        env_backup = os.environ.get("TSFM_ADDITIONAL_HANDLER_MODULES")

        try:
            # Without dot enforcement, "mycompany" would match "mycompany_evil"
            # With dot enforcement, only "mycompany." matches "mycompany.X"
            os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = "mycompany."

            import importlib

            importlib.reload(sys.modules["boilerplate.service_handler"])
            from boilerplate.service_handler import _validate_handler_module_path

            # Should allow exact match
            _validate_handler_module_path("mycompany.models.handler")

            # Should NOT allow similar but different module
            with pytest.raises(ValueError):
                _validate_handler_module_path("mycompany_evil.models.handler")

        finally:
            if env_backup is not None:
                os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = env_backup
            elif "TSFM_ADDITIONAL_HANDLER_MODULES" in os.environ:
                del os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"]

    def test_additional_modules_with_spaces(self):
        """Test that spaces in the environment variable are handled correctly"""
        env_backup = os.environ.get("TSFM_ADDITIONAL_HANDLER_MODULES")

        try:
            # Set with spaces around commas
            os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = "mycompany.models. , custom.handlers. "

            import importlib

            importlib.reload(sys.modules["boilerplate.service_handler"])
            from boilerplate.service_handler import _validate_handler_module_path

            # Should still work after stripping spaces
            _validate_handler_module_path("mycompany.models.handler")
            _validate_handler_module_path("custom.handlers.handler")

        finally:
            if env_backup is not None:
                os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = env_backup
            elif "TSFM_ADDITIONAL_HANDLER_MODULES" in os.environ:
                del os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"]

    def test_empty_additional_modules(self):
        """Test that empty environment variable doesn't break validation"""
        env_backup = os.environ.get("TSFM_ADDITIONAL_HANDLER_MODULES")

        try:
            os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = ""

            import importlib

            importlib.reload(sys.modules["boilerplate.service_handler"])
            from boilerplate.service_handler import ALLOWED_HANDLER_MODULE_PREFIXES

            # Should only have base prefixes
            assert len(ALLOWED_HANDLER_MODULE_PREFIXES) == 3
            assert "tsfm_public." in ALLOWED_HANDLER_MODULE_PREFIXES

        finally:
            if env_backup is not None:
                os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = env_backup
            elif "TSFM_ADDITIONAL_HANDLER_MODULES" in os.environ:
                del os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"]


class TestEnvironmentVariables:
    """Test environment variable defaults"""

    def test_tsfm_allow_load_from_hf_hub_default(self):
        """Test that TSFM_ALLOW_LOAD_FROM_HF_HUB defaults to 0 (disabled)"""
        # Remove the env var if it exists
        env_backup = os.environ.get("TSFM_ALLOW_LOAD_FROM_HF_HUB")
        if "TSFM_ALLOW_LOAD_FROM_HF_HUB" in os.environ:
            del os.environ["TSFM_ALLOW_LOAD_FROM_HF_HUB"]

        try:
            # Re-import to get fresh default
            import importlib

            import services.inference.tsfminference as tsfminference_module

            importlib.reload(tsfminference_module)

            # Should default to False (0)
            assert not tsfminference_module.TSFM_ALLOW_LOAD_FROM_HF_HUB
        finally:
            # Restore env var
            if env_backup is not None:
                os.environ["TSFM_ALLOW_LOAD_FROM_HF_HUB"] = env_backup

    def test_tsfm_trust_remote_code_default(self):
        """Test that TSFM_TRUST_REMOTE_CODE defaults to 0 (disabled)"""
        # Remove the env var if it exists
        env_backup = os.environ.get("TSFM_TRUST_REMOTE_CODE")
        if "TSFM_TRUST_REMOTE_CODE" in os.environ:
            del os.environ["TSFM_TRUST_REMOTE_CODE"]

        try:
            # Re-import to get fresh default
            import importlib

            importlib.reload(sys.modules["boilerplate.service_handler"])
            from boilerplate.service_handler import TSFM_TRUST_REMOTE_CODE

            # Should default to False (0)
            assert not TSFM_TRUST_REMOTE_CODE
        finally:
            # Restore env var
            if env_backup is not None:
                os.environ["TSFM_TRUST_REMOTE_CODE"] = env_backup


class TestIntegrationScenarios:
    """Integration tests for realistic attack scenarios"""

    def test_poc_attack_blocked(self):
        """Test that the PoC attack from the vulnerability report is blocked"""
        # The PoC uses subprocess.Popen
        with pytest.raises(ValueError, match="subprocess"):
            _validate_handler_module_path("subprocess")

    def test_similar_name_attack_blocked(self):
        """Test that modules with similar names to allowed ones are blocked"""
        # Attacker might try to use similar-looking names
        with pytest.raises(ValueError):
            _validate_handler_module_path("tsfm_public_evil.toolkit.handler")

        with pytest.raises(ValueError):
            _validate_handler_module_path("tsfmpublic.toolkit.handler")  # Missing underscore

    def test_path_traversal_attempt_blocked(self):
        """Test that path traversal attempts are blocked"""
        with pytest.raises(ValueError):
            _validate_handler_module_path("../../../etc/passwd")

        with pytest.raises(ValueError):
            _validate_handler_module_path("....tsfm_public.handler")


class TestHandlerClassNameValidation:
    """Test suite for handler class name validation"""

    def test_allowed_handler_suffix(self):
        """Test that class names ending with 'Handler' are allowed"""
        # Should not raise
        _validate_handler_class_name("ForecastingHandler")
        _validate_handler_class_name("TinyTimeMixerForecastingInferenceHandler")
        _validate_handler_class_name("CustomHandler")

    def test_allowed_service_handler_suffix(self):
        """Test that class names ending with 'ServiceHandler' are allowed"""
        # Should not raise
        _validate_handler_class_name("TSFMServiceHandler")
        _validate_handler_class_name("CustomServiceHandler")

    def test_blocked_subprocess_popen(self):
        """Test that 'Popen' class name is blocked (PoC attack vector)"""
        with pytest.raises(ValueError, match="Security: Handler class name 'Popen' does not match allowed pattern"):
            _validate_handler_class_name("Popen")

    def test_blocked_arbitrary_class_names(self):
        """Test that arbitrary class names without proper suffix are blocked"""
        with pytest.raises(ValueError, match="does not match allowed pattern"):
            _validate_handler_class_name("subprocess")

        with pytest.raises(ValueError, match="does not match allowed pattern"):
            _validate_handler_class_name("os")

        with pytest.raises(ValueError, match="does not match allowed pattern"):
            _validate_handler_class_name("system")

    def test_blocked_module_level_objects(self):
        """Test that common module-level dangerous objects are blocked"""
        dangerous_names = ["Popen", "call", "run", "system", "exec", "eval"]
        for name in dangerous_names:
            with pytest.raises(ValueError):
                _validate_handler_class_name(name)

    def test_error_message_includes_allowed_suffixes(self):
        """Test that error message includes the allowed suffixes"""
        with pytest.raises(ValueError) as exc_info:
            _validate_handler_class_name("MaliciousClass")

        error_msg = str(exc_info.value)
        assert "Handler" in error_msg
        assert "ServiceHandler" in error_msg
        assert "Contact your security team" in error_msg

    def test_case_sensitive_validation(self):
        """Test that validation is case-sensitive for suffixes"""
        # Lowercase should be blocked
        with pytest.raises(ValueError):
            _validate_handler_class_name("Forecastinghandler")

        # Mixed case in suffix should be blocked
        with pytest.raises(ValueError):
            _validate_handler_class_name("ForecastingHANDLER")

    def test_partial_suffix_match_blocked(self):
        """Test that partial matches of suffix are blocked"""
        # "Hand" is not "Handler"
        with pytest.raises(ValueError):
            _validate_handler_class_name("ForecastingHand")

        # "Handle" is not "Handler"
        with pytest.raises(ValueError):
            _validate_handler_class_name("ForecastingHandle")

    def test_suffix_in_middle_blocked(self):
        """Test that suffix must be at the end, not in the middle"""
        with pytest.raises(ValueError):
            _validate_handler_class_name("HandlerForecasting")

        with pytest.raises(ValueError):
            _validate_handler_class_name("ServiceHandlerExtra")


class TestIntegrationWithClassNameValidation:
    """Integration tests including class name validation"""

    def test_poc_attack_with_class_name_blocked(self):
        """Test that the PoC attack is blocked by class name validation"""
        # Even if module path validation is bypassed, class name validation should block
        with pytest.raises(ValueError, match="Popen"):
            _validate_handler_class_name("Popen")

    def test_re_exported_dangerous_objects_blocked(self):
        """Test that re-exported dangerous objects are blocked by class name validation"""
        # If an allowed module re-exports subprocess.Popen, the class name check blocks it
        dangerous_class_names = [
            "Popen",
            "call",
            "check_output",
            "system",
            "exec",
            "eval",
            "compile",
        ]
        for class_name in dangerous_class_names:
            with pytest.raises(ValueError):
                _validate_handler_class_name(class_name)

    def test_legitimate_handlers_pass_both_validations(self):
        """Test that legitimate handlers pass both module and class name validation"""
        # Module path validation
        _validate_handler_module_path("tsfminference.tsfm_inference_handler")
        _validate_handler_module_path("tsfmfinetuning.tsfm_tuning_handler")
        _validate_handler_module_path("tsfm_public.toolkit.some_handler")

        # Class name validation
        _validate_handler_class_name("TinyTimeMixerForecastingInferenceHandler")
        _validate_handler_class_name("ForecastingTuningHandler")
        _validate_handler_class_name("TSFMServiceHandler")


# Made with Bob
