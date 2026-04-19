# Configuring Additional Handler Modules

## Overview

The security fix for GHSA-9mmf-qgpx-g4qx includes a mechanism to extend the allowlist of trusted handler modules without modifying code. This allows organizations to use custom handler implementations while maintaining security validation.

## Environment Variable

### `TSFM_ADDITIONAL_HANDLER_MODULES`

**Purpose:** Extend the allowlist of trusted module prefixes for handler loading.

**Format:** Comma-separated list of module prefixes. Each prefix **MUST** end with a dot (`.`) to ensure exact module path matching.

**Default:** Empty (no additional modules)

**Security Requirement:** The dot suffix is enforced to prevent overly broad matches. For example:
- ✅ `mycompany.models.` - Only matches `mycompany.models.*`
- ❌ `mycompany.models` - Would also match `mycompany.models_evil.*` (rejected at startup)

**Example:**
```bash
export TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.models.,custom.handlers."
```

**Invalid Example (will cause startup failure):**
```bash
# Missing dots - will raise ValueError
export TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.models,custom.handlers"
```

## Use Cases

### 1. Custom Model Implementations

If your organization has developed custom time series models with their own handler implementations:

```bash
export TSFM_ADDITIONAL_HANDLER_MODULES="acme.timeseries.handlers."
```

This allows loading handlers from modules like:
- `acme.timeseries.handlers.forecasting_handler`
- `acme.timeseries.handlers.anomaly_detection_handler`

### 2. Third-Party Extensions

When using trusted third-party extensions:

```bash
export TSFM_ADDITIONAL_HANDLER_MODULES="vendor.tsfm.extensions."
```

### 3. Multiple Custom Modules

You can specify multiple module prefixes:

```bash
export TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.models.,partner.handlers.,internal.tsfm."
```

## Security Considerations

### ⚠️ Important Security Guidelines

1. **Dot suffix is mandatory** - All prefixes MUST end with a dot (`.`). This is enforced at startup and prevents overly broad module matching that could inadvertently allow malicious modules.

2. **Only add trusted modules** - Modules added via this environment variable bypass the built-in allowlist but still undergo class name validation.

3. **Use specific prefixes** - Prefer specific prefixes over broad ones:
   - ✅ Good: `mycompany.tsfm.handlers.`
   - ❌ Bad: `mycompany.` (too broad, could include unrelated code)

4. **Audit regularly** - Regularly review and audit the modules you've added to the allowlist.

5. **Document your additions** - Maintain documentation of why each module prefix was added and who approved it.

6. **Code review required** - All handler classes in additional modules should undergo security code review.

### Handler Class Requirements

Even with additional modules enabled, all handler classes must:
- End with `Handler` or `ServiceHandler`
- Follow secure coding practices
- Not expose dangerous functionality

### Example: Secure Custom Handler

```python
# mycompany/models/custom_handler.py

class CustomForecastingHandler:  # ✅ Ends with 'Handler'
    """Custom handler for proprietary forecasting model."""
    
    def __init__(self, model_id, model_path, handler_config):
        self.model_id = model_id
        self.model_path = model_path
        self.handler_config = handler_config
    
    def prepare(self, data, schema=None, parameters=None, **kwargs):
        # Safe implementation
        pass
```

## Configuration Examples

### Docker Deployment

```dockerfile
FROM granite-tsfm-inference:latest

# Add custom handler modules
ENV TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.tsfm.handlers."

# Copy custom handler code
COPY mycompany /app/mycompany
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tsfm-inference
spec:
  template:
    spec:
      containers:
      - name: inference
        image: granite-tsfm-inference:latest
        env:
        - name: TSFM_ADDITIONAL_HANDLER_MODULES
          value: "mycompany.tsfm.handlers."
```

### Local Development

```bash
# In your shell or .env file
export TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.models.,custom.handlers."

# Start the service
python -m tsfminference.main
```

## Validation

### Testing Your Configuration

1. **Verify module loading:**
```python
import os
os.environ["TSFM_ADDITIONAL_HANDLER_MODULES"] = "mycompany.models."

from services.boilerplate.service_handler import ALLOWED_HANDLER_MODULE_PREFIXES
print(ALLOWED_HANDLER_MODULE_PREFIXES)
# Should include: ('tsfm_public.', 'tsfminference.', 'tsfmfinetuning.', 'mycompany.models.')
```

2. **Check logs on startup:**
```
INFO: Additional handler module prefixes enabled: ('mycompany.models.',). Ensure these modules are from trusted sources.
```

3. **Test handler loading:**
```python
from tsfm_public.toolkit.tsfm_config import TSFMConfig

config = TSFMConfig(
    inference_handler_module_path="mycompany.models.forecasting",
    inference_handler_class_name="CustomForecastingHandler"
)
# Should load successfully if module is in allowlist
```

## Troubleshooting

### Error: "Additional handler module prefix 'X' must end with a dot"

**Cause:** One or more prefixes in `TSFM_ADDITIONAL_HANDLER_MODULES` doesn't end with a dot.

**Solution:** Add the trailing dot to each prefix:
```bash
# Wrong
export TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.models"

# Correct
export TSFM_ADDITIONAL_HANDLER_MODULES="mycompany.models."
```

### Error: "Handler module path 'X' is not allowed"

**Cause:** The module prefix is not in the allowlist.

**Solution:** Add the module prefix to `TSFM_ADDITIONAL_HANDLER_MODULES` (with trailing dot):
```bash
export TSFM_ADDITIONAL_HANDLER_MODULES="X."
```

### Error: "Handler class name 'Y' does not match allowed pattern"

**Cause:** The class name doesn't end with `Handler` or `ServiceHandler`.

**Solution:** Rename your handler class to follow the naming convention:
```python
# Before
class MyCustomClass:
    pass

# After
class MyCustomHandler:
    pass
```

### Module not found after adding to allowlist

**Cause:** The module is not installed or not in Python path.

**Solution:** Ensure the module is installed and accessible:
```bash
pip install mycompany-models
# or
export PYTHONPATH=/path/to/mycompany:$PYTHONPATH
```

## Best Practices

1. **Principle of Least Privilege:** Only add the minimum required module prefixes.

2. **Environment-Specific Configuration:**
   - Development: More permissive for testing
   - Production: Strict, audited allowlist only

3. **Version Control:** Document your `TSFM_ADDITIONAL_HANDLER_MODULES` configuration in version control.

4. **Security Review Process:**
   - Require security team approval before adding new modules
   - Regular audits of allowed modules
   - Remove unused module prefixes

5. **Monitoring:** Log and monitor which handlers are being loaded in production.

## Comparison with TSFM_TRUST_REMOTE_CODE

| Feature | TSFM_ADDITIONAL_HANDLER_MODULES | TSFM_TRUST_REMOTE_CODE |
|---------|--------------------------------|------------------------|
| Security | ✅ Maintains validation | ❌ Bypasses all validation |
| Flexibility | ✅ Extends allowlist | ✅ Allows anything |
| Production Use | ✅ Recommended | ❌ Not recommended |
| Class Name Validation | ✅ Still enforced | ❌ Bypassed |
| Audit Trail | ✅ Explicit configuration | ⚠️ Blanket trust |

**Recommendation:** Always prefer `TSFM_ADDITIONAL_HANDLER_MODULES` over `TSFM_TRUST_REMOTE_CODE` for production deployments.

## Support

For questions or security concerns regarding custom handler modules:
1. Contact your security team
2. Review the main security documentation: `SECURITY_FIX_GHSA-9mmf-qgpx-g4qx.md`
3. Open an issue in the granite-tsfm repository

---

**Last Updated:** 2026-04-19  
**Related:** GHSA-9mmf-qgpx-g4qx Security Fix