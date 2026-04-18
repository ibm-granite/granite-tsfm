# Copyright contributors to the TSFM project
#
"""Tests for security validation in service handler module loading"""

import os
import pytest
from unittest.mock import patch

# Import the validation function and constants
import sys
from pathlib import Path

# Add boilerplate to path for testing
boilerplate_path = Path(__file__).parent.parent.parent / "services" / "boilerplate"
sys.path.insert(0, str(boilerplate_path))

from service_handler import _validate_handler_module_path, ALLOWED_HANDLER_MODULE_PREFIXES


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
        assert "TSFM_TRUST_REMOTE_CODE=1" in error_msg

    def test_case_sensitive_validation(self):
        """Test that validation is case-sensitive"""
        # Uppercase should be blocked
        with pytest.raises(ValueError):
            _validate_handler_module_path("TSFM_PUBLIC.toolkit.handler")
        
        # Mixed case should be blocked
        with pytest.raises(ValueError):
            _validate_handler_module_path("Tsfm_Public.toolkit.handler")


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
            assert tsfminference_module.TSFM_ALLOW_LOAD_FROM_HF_HUB == False
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
            importlib.reload(sys.modules['service_handler'])
            from service_handler import TSFM_TRUST_REMOTE_CODE
            
            # Should default to False (0)
            assert TSFM_TRUST_REMOTE_CODE == False
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

# Made with Bob
