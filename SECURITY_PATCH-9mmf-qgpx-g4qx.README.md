# Security Patch: Remote Code Execution Vulnerability Fix

## ⚠️ Critical Security Update

This patch addresses **GHSA-9mmf-qgpx-g4qx**, a critical remote code execution vulnerability in the granite-tsfm inference service.

## 🔒 What Was Fixed

The vulnerability allowed unauthenticated attackers to execute arbitrary code by publishing malicious HuggingFace repositories with crafted `tsfm_config.json` files.

**Three security measures have been implemented:**

1. **Module Path Allowlist** - Only trusted modules can be loaded (`tsfm_public.*`, `tsfminference.*`, `tsfmfinetuning.*`)
2. **Explicit Trust Requirement** - New `TSFM_TRUST_REMOTE_CODE` environment variable (defaults to disabled)
3. **Secure Defaults** - `TSFM_ALLOW_LOAD_FROM_HF_HUB` now defaults to `0` (disabled)

## 🚨 Breaking Changes

### Models No Longer Load from HuggingFace Hub by Default

**Before:** Models automatically loaded from HF Hub when not found locally  
**After:** Loading from HF Hub requires explicit opt-in

### Migration Options

**Option 1: Use Local Models (Recommended)**
```bash
export TSFM_MODEL_DIR=/path/to/local/models
# No need to set TSFM_ALLOW_LOAD_FROM_HF_HUB
```

**Option 2: Re-enable HF Hub Loading (Use with Caution)**
```bash
export TSFM_ALLOW_LOAD_FROM_HF_HUB=1
# Only load models from trusted sources!
```

## 📋 Quick Start

### For Production Deployments

```bash
# Recommended: Use local models only
export TSFM_MODEL_DIR=/path/to/trusted/models
export TSFM_ALLOW_LOAD_FROM_HF_HUB=0  # Explicit (already default)
export TSFM_TRUST_REMOTE_CODE=0       # Explicit (already default)

# Start service
python -m gunicorn -k uvicorn.workers.UvicornWorker tsfminference.main:app
```

### For Development/Testing

```bash
# If you need to load from HF Hub during development
export TSFM_ALLOW_LOAD_FROM_HF_HUB=1
export TSFM_MODEL_DIR=/path/to/local/models

# Start service
python -m gunicorn -k uvicorn.workers.UvicornWorker tsfminference.main:app
```

### ⚠️ Never in Production

```bash
# DANGEROUS: Disables all security validation
export TSFM_TRUST_REMOTE_CODE=1  # ❌ DO NOT USE IN PRODUCTION
```

## 🧪 Testing the Fix

Run the security validation tests:

```bash
pytest tests/toolkit/test_security_handler_validation.py -v
```

Expected output:
```
✅ test_allowed_tsfm_public_module PASSED
✅ test_blocked_subprocess_module PASSED
✅ test_poc_attack_blocked PASSED
...
```

## 📚 Documentation

- **Full Security Details:** See [SECURITY_FIX_GHSA-9mmf-qgpx-g4qx.md](./SECURITY_FIX_GHSA-9mmf-qgpx-g4qx.md)
- **Modified Files:**
  - `services/boilerplate/service_handler.py` - Added validation logic
  - `services/inference/tsfminference/__init__.py` - Changed default
  - `services/finetuning/tsfmfinetuning/__init__.py` - Changed default
  - `tests/toolkit/test_security_handler_validation.py` - New tests

## 🔍 Verification

### Verify the PoC Attack is Blocked

```bash
# Start the service
make -C services/inference start_service_local

# Try the PoC attack (should fail with security error)
curl -X POST http://localhost:8000/v1/inference/forecasting \
  -H "Content-Type: application/json" \
  -d '{"model_id":"attacker/evil-tsfm","schema":{},"parameters":{},"data":{}}'

# Expected: Error message about module path not being allowed
```

### Check Environment Variables

```bash
# In Python
python -c "from services.inference.tsfminference import TSFM_ALLOW_LOAD_FROM_HF_HUB; print(f'HF Hub Loading: {TSFM_ALLOW_LOAD_FROM_HF_HUB}')"
# Expected: HF Hub Loading: False
```

## ❓ FAQ

### Q: Why did my model stop loading?
**A:** Models from HuggingFace Hub no longer load by default. Set `TSFM_ALLOW_LOAD_FROM_HF_HUB=1` or use local models.

### Q: Can I use custom handler modules?
**A:** Yes, but they must be in the `tsfm_public.*`, `tsfminference.*`, or `tsfmfinetuning.*` namespaces.

### Q: What if I need to load untrusted code for testing?
**A:** Set `TSFM_TRUST_REMOTE_CODE=1` (development only, never in production).

### Q: How do I update my deployment?
**A:** 
1. Pull the latest code
2. Run `make boilerplate` in service directories
3. Update environment variables as needed
4. Restart services

### Q: Are there any performance impacts?
**A:** No. The validation adds negligible overhead (single string prefix check).

## 🆘 Support

If you encounter issues:

1. Check the [full security documentation](./SECURITY_FIX_GHSA-9mmf-qgpx-g4qx.md)
2. Review your environment variables
3. Check service logs for security warnings
4. Run the test suite to verify the fix

## 📅 Timeline

- **Vulnerability Discovered:** [Date]
- **Patch Released:** 2026-04-18
- **Recommended Action:** Update immediately

## ✅ Checklist for Operators

- [ ] Pull latest code
- [ ] Run `make boilerplate` in service directories
- [ ] Review and update environment variables
- [ ] Test with your models
- [ ] Update deployment configurations
- [ ] Restart services
- [ ] Monitor logs for security warnings
- [ ] Run security tests

---

**For detailed technical information, see [SECURITY_FIX_GHSA-9mmf-qgpx-g4qx.md](./SECURITY_FIX_GHSA-9mmf-qgpx-g4qx.md)**