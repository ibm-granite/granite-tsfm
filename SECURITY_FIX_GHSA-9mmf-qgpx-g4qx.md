# Security Fix: GHSA-9mmf-qgpx-g4qx - Remote Code Execution via Malicious HuggingFace Repository

## Vulnerability Summary

**Severity:** Critical  
**CVE:** GHSA-9mmf-qgpx-g4qx  
**Impact:** Unauthenticated Remote Code Execution

The granite-tsfm inference service was vulnerable to arbitrary code execution through malicious `tsfm_config.json` files hosted on HuggingFace Hub. An unauthenticated attacker could publish a crafted HF repository and trigger code execution by sending a single POST request to the inference endpoint.

## Root Cause

The vulnerability existed in `services/boilerplate/service_handler.py` where attacker-controlled strings from downloaded `tsfm_config.json` files were passed directly to `importlib.import_module()` and `getattr()` without validation:

```python
# VULNERABLE CODE (before fix)
module = importlib.import_module(getattr(config, "inference_handler_module_path"))
my_class = getattr(module, getattr(config, "inference_handler_class_name"))
```

Combined with:
- `TSFM_ALLOW_LOAD_FROM_HF_HUB` defaulting to `1` (enabled)
- No module path allowlist
- No operator consent requirement for remote code

## Proof of Concept

1. Attacker publishes HF repo `attacker/evil-tsfm` with malicious `tsfm_config.json`:
```json
{
  "inference_handler_module_path": "subprocess",
  "inference_handler_class_name": "Popen"
}
```

2. Attacker sends unauthenticated request:
```bash
curl -X POST http://target/v1/inference/forecasting \
  -H "Content-Type: application/json" \
  -d '{"model_id":"attacker/evil-tsfm","schema":{},"parameters":{},"data":{}}'
```

3. Service downloads config from HF Hub and executes `subprocess.Popen` with attacker-controlled arguments.

## Security Fixes Implemented

### 1. Module Path Allowlist (Primary Defense)

**File:** `services/boilerplate/service_handler.py`

Added strict allowlist validation for handler module paths:

```python
ALLOWED_HANDLER_MODULE_PREFIXES = (
    "tsfm_public.",
    "tsfminference.",
    "tsfmfinetuning.",
)

def _validate_handler_module_path(module_path: str) -> None:
    """Validate that the handler module path is from a trusted source."""
    if not any(module_path.startswith(prefix) for prefix in ALLOWED_HANDLER_MODULE_PREFIXES):
        raise ValueError(
            f"Security: Handler module path '{module_path}' is not allowed. "
            f"Only modules with the following prefixes are permitted: {ALLOWED_HANDLER_MODULE_PREFIXES}. "
            f"Contact your security team before enabling TSFM_TRUST_REMOTE_CODE in production."
        )
```

**Impact:** Blocks all attempts to load arbitrary Python modules. Only trusted internal modules can be loaded.

### 1.5. Handler Class Name Validation (Defense in Depth)

**File:** `services/boilerplate/service_handler.py`

Added validation for handler class names to enforce naming conventions:

```python
def _validate_handler_class_name(class_name: str) -> None:
    """Validate that the handler class name follows expected naming conventions."""
    allowed_suffixes = ("Handler", "ServiceHandler")
    if not any(class_name.endswith(suffix) for suffix in allowed_suffixes):
        raise ValueError(
            f"Security: Handler class name '{class_name}' does not match allowed pattern. "
            f"Class names must end with one of: {allowed_suffixes}. "
            f"Contact your security team before enabling TSFM_TRUST_REMOTE_CODE in production."
        )
```

**Impact:** Provides a second layer of defense. Even if an allowlisted module re-exports dangerous objects (like `subprocess` or `os`) at module level, the class name validation prevents accessing them via the `class_name` field in `tsfm_config.json`.

**Example Attack Blocked:** If `tsfm_public.some_module` were to contain `from subprocess import Popen`, an attacker could not exploit it because `Popen` doesn't end with `Handler` or `ServiceHandler`.

### 2. Explicit Trust Requirement (Defense in Depth)

**File:** `services/boilerplate/service_handler.py`

Added `TSFM_TRUST_REMOTE_CODE` environment variable:

```python
TSFM_TRUST_REMOTE_CODE = int(os.getenv("TSFM_TRUST_REMOTE_CODE", "0")) == 1
```

Module validation is enforced unless operator explicitly sets `TSFM_TRUST_REMOTE_CODE=1`:

```python
if not TSFM_TRUST_REMOTE_CODE:
    _validate_handler_module_path(module_path)
else:
    LOGGER.warning(
        f"TSFM_TRUST_REMOTE_CODE is enabled. Loading handler module '{module_path}' without validation. "
        f"This may pose a security risk if loading from untrusted sources."
    )
```

**Impact:** Requires explicit operator consent before bypassing validation. Logs warning when validation is disabled.

### 3. Secure Default for HF Hub Loading

**Files:** 
- `services/inference/tsfminference/__init__.py`
- `services/finetuning/tsfmfinetuning/__init__.py`

Changed default value of `TSFM_ALLOW_LOAD_FROM_HF_HUB` from `1` to `0`:

```python
# Security: Default to disabling loading from HuggingFace Hub to prevent
# unauthenticated remote code execution. Operators must explicitly opt-in.
TSFM_ALLOW_LOAD_FROM_HF_HUB = int(os.getenv("TSFM_ALLOW_LOAD_FROM_HF_HUB", "0")) == 1
```

**Impact:** Prevents loading models from HF Hub by default. Operators must explicitly enable this feature.

## Migration Guide

### For Operators

**Breaking Change:** Models will no longer load from HuggingFace Hub by default.

To restore previous behavior (use with caution):
```bash
export TSFM_ALLOW_LOAD_FROM_HF_HUB=1
```

**Recommended:** Use local model directories instead:
```bash
export TSFM_MODEL_DIR=/path/to/local/models
```

### For Developers

**Custom Handler Modules:** If you have custom handler modules, you have two options:

1. **Recommended:** Add your module prefix to the allowlist via environment variable:
```bash
export TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.models.,custom.handlers."
```
This allows your custom modules while maintaining security validation. Each prefix should end with a dot.

2. **Built-in prefixes:** Ensure your modules use one of the built-in allowed prefixes:
- `tsfm_public.*`
- `tsfminference.*`
- `tsfmfinetuning.*`

**Handler Class Naming:** All handler classes must end with either `Handler` or `ServiceHandler` to pass validation.

**Testing with Untrusted Code:** For development/testing only:
```bash
export TSFM_TRUST_REMOTE_CODE=1  # USE WITH EXTREME CAUTION - BYPASSES ALL VALIDATION
```

## Validation

### Automated Tests

Security validation tests added in `tests/toolkit/test_security_handler_validation.py`:

- ✅ Allowed module prefixes are accepted
- ✅ Arbitrary modules (subprocess, os, sys) are blocked
- ✅ PoC attack vector is blocked
- ✅ Similar-name attacks are blocked
- ✅ Environment variables default to secure values

Run tests:
```bash
pytest tests/toolkit/test_security_handler_validation.py -v
```

### Manual Verification

1. **Verify PoC is blocked:**
```bash
# This should fail with security error
curl -X POST http://localhost:8000/v1/inference/forecasting \
  -H "Content-Type: application/json" \
  -d '{"model_id":"attacker/evil-tsfm","schema":{},"parameters":{},"data":{}}'
```

2. **Verify HF Hub is disabled by default:**
```bash
# Without TSFM_ALLOW_LOAD_FROM_HF_HUB=1, this should fail
curl -X POST http://localhost:8000/v1/inference/forecasting \
  -H "Content-Type: application/json" \
  -d '{"model_id":"some/hf-model","schema":{},"parameters":{},"data":{}}'
```

## Security Considerations

### Defense in Depth

This fix implements multiple layers of security:

1. **Module path allowlist validation** - Primary defense, blocks malicious modules
2. **Configurable additional modules** - Allows extending the allowlist securely via environment variable
3. **Class name validation** - Secondary defense, blocks dangerous objects even from allowed modules
4. **Explicit trust requirement** - Requires operator consent to bypass validation
5. **Secure defaults** - HF Hub loading disabled by default
6. **Logging** - Warnings when security features are disabled

### Residual Risks

Even with these fixes:

- **Trusted module vulnerabilities:** If vulnerabilities exist in allowed modules (`tsfm_public.*`, etc.) and the vulnerable code follows the naming convention (ends with `Handler` or `ServiceHandler`), they could still be exploited
- **Operator misconfiguration:** Setting `TSFM_TRUST_REMOTE_CODE=1` removes all protection
- **Local file attacks:** If attacker can write to `TSFM_MODEL_DIR`, they could still inject malicious configs
- **Class name bypass:** If an allowed module contains a class ending with `Handler` or `ServiceHandler` that itself provides dangerous functionality, it could be exploited

### Recommendations

1. **Never set `TSFM_TRUST_REMOTE_CODE=1` in production** - Contact your security team first
2. **Use local model directories** instead of HF Hub when possible
3. **Restrict write access** to `TSFM_MODEL_DIR`
4. **Monitor logs** for security warnings
5. **Keep dependencies updated** to patch vulnerabilities in allowed modules
6. **Code review handler classes** - Ensure all handler classes in allowed modules follow secure coding practices
7. **Principle of least privilege** - Run services with minimal required permissions
8. **Audit additional modules** - If using `TSFM_ADDITIONAL_HANDLER_MODULES`, ensure those modules are from trusted sources and regularly audited
9. **Use specific prefixes** - When adding custom modules, use specific prefixes (e.g., `mycompany.tsfm.handlers.`) rather than broad ones (e.g., `mycompany.`)

## References

- **Vulnerability Report:** GHSA-9mmf-qgpx-g4qx
- **Affected Files:**
  - `services/boilerplate/service_handler.py` (lines 140-142)
  - `services/inference/tsfminference/__init__.py` (line 34)
  - `services/finetuning/tsfmfinetuning/__init__.py` (line 33)
  - `tsfm_public/toolkit/tsfm_config.py` (line 54)

## Timeline

- **Vulnerability Discovered:** [Date from report]
- **Fix Implemented:** 2026-04-18
- **Fix Released:** [Pending]

## Credits

Security vulnerability reported by: Martin Brodeur
Fix implemented by: Bob (AI Security Engineer)
