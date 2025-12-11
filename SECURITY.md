# Security Advisory

## Critical Security Updates - December 10, 2024

### Summary
Updated dependencies to patch critical security vulnerabilities in PyTorch and MONAI libraries.

## Vulnerabilities Fixed

### 1. MONAI Vulnerabilities (CVE-XXXX)
**Affected Version:** monai <= 1.5.0  
**Fixed Version:** monai >= 1.5.1  
**Severity:** HIGH

**Issues:**
- **Unsafe Pickle Deserialization**: May lead to Remote Code Execution (RCE)
- **Unsafe torch usage**: May lead to arbitrary code execution
- **Path Traversal**: Does not prevent path traversal, potentially leading to arbitrary file writes

**Impact:**
- Attackers could execute arbitrary code on the server
- Potential unauthorized file system access
- Data integrity and confidentiality at risk

**Mitigation:**
- Updated to monai==1.5.1 in requirements.txt
- All pickle-based model loading now uses safe deserialization
- Path validation implemented in storage service

### 2. PyTorch Vulnerability
**Affected Version:** torch < 2.6.0  
**Fixed Version:** torch >= 2.6.0  
**Severity:** HIGH

**Issue:**
- **Remote Code Execution via torch.load**: Even with `weights_only=True`, RCE is possible

**Impact:**
- Loading malicious model checkpoints could execute arbitrary code
- Compromise of entire system possible

**Mitigation:**
- Updated to torch==2.6.0 and torchvision==0.21.0 in requirements.txt
- All model loading code reviewed for safe practices

## Updated Dependencies

```txt
# Before (Vulnerable)
torch==2.5.1
torchvision==0.20.1
monai==1.4.0

# After (Patched)
torch==2.6.0
torchvision==0.21.0
monai==1.5.1
```

## Action Items for Users

### Immediate Actions Required:

1. **Update Dependencies:**
   ```bash
   cd backend
   source venv/bin/activate
   pip install --upgrade -r requirements.txt
   ```

2. **Verify Updates:**
   ```bash
   pip show torch monai
   # torch version should be 2.6.0
   # monai version should be 1.5.1
   ```

3. **Review Existing Checkpoints:**
   - Only load model checkpoints from trusted sources
   - Scan existing checkpoint files for potential tampering
   - Re-train models if checkpoint integrity is uncertain

### Security Best Practices

#### For Model Loading:
```python
# Good - Safe model loading
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True  # Still use this, but now with patched version
)

# Better - Add file validation
import hashlib

def verify_checkpoint(path, expected_hash):
    """Verify checkpoint integrity before loading."""
    with open(path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    if file_hash != expected_hash:
        raise SecurityError("Checkpoint hash mismatch!")
```

#### For File Operations:
```python
# Good - Path validation in storage service
def save_artifact(self, image, image_id, artifact_type):
    # Sanitize inputs to prevent path traversal
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', image_id)
    safe_type = re.sub(r'[^a-zA-Z0-9_-]', '', artifact_type)
    
    # Construct safe path
    filename = f"{safe_id}_{safe_type}.png"
    filepath = self.artifacts_dir / filename
    
    # Verify path is within allowed directory
    if not filepath.resolve().is_relative_to(self.artifacts_dir.resolve()):
        raise SecurityError("Path traversal detected!")
```

## Testing After Update

### 1. Verify Model Loading:
```bash
cd backend
python -c "
import torch
from app.models.unet.model import get_unet_model

print(f'PyTorch version: {torch.__version__}')
assert torch.__version__ >= '2.6.0', 'PyTorch not updated!'

model = get_unet_model()
print('Model loading test: PASSED')
"
```

### 2. Verify MONAI:
```bash
python -c "
import monai
print(f'MONAI version: {monai.__version__}')
assert monai.__version__ >= '1.5.1', 'MONAI not updated!'
print('MONAI version check: PASSED')
"
```

### 3. Run Security Tests:
```bash
pytest tests/test_security.py -v
```

## Additional Security Recommendations

### 1. Network Security
- Deploy behind a firewall
- Use HTTPS/TLS for all communications
- Implement rate limiting on API endpoints
- Add authentication/authorization

### 2. Input Validation
- Validate all file uploads (type, size, content)
- Sanitize all user inputs
- Implement CSRF protection

### 3. Model Security
- Store model checksums separately
- Verify model integrity before loading
- Use signed models when possible
- Implement model versioning

### 4. Monitoring
- Log all model loading events
- Monitor for suspicious file access patterns
- Set up alerts for failed authentication attempts
- Track API usage and anomalies

## Configuration Updates

### Update .env.example:
```bash
# Security Settings
ALLOWED_FILE_TYPES=image/jpeg,image/png,application/dicom
MAX_FILE_SIZE_MB=10
ENABLE_CHECKPOINT_VERIFICATION=true
CHECKPOINT_HASH_ALGORITHM=sha256
```

## References

- [PyTorch Security Advisory](https://github.com/pytorch/pytorch/security/advisories)
- [MONAI Security Advisory](https://github.com/Project-MONAI/MONAI/security/advisories)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)

## Timeline

- **2024-12-10**: Vulnerabilities identified
- **2024-12-10**: Dependencies updated to patched versions
- **2024-12-10**: Security advisory published
- **Next**: Security testing and validation

## Contact

For security concerns or questions:
- Open a security advisory on GitHub (private)
- Do not disclose vulnerabilities publicly until patched

## Status

✅ **PATCHED** - All known vulnerabilities addressed in this update.

---

**Last Updated:** 2024-12-10  
**Severity:** HIGH  
**Status:** RESOLVED
