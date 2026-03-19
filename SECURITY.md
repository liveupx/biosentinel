# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 2.1.x | ✅ Security fixes |
| 2.0.x | ✅ Security fixes (no new features) |
| < 2.0 | ❌ Not supported |

## Reporting a Vulnerability

**⚠️ PLEASE DO NOT FILE PUBLIC GITHUB ISSUES FOR SECURITY VULNERABILITIES.**

BioSentinel handles sensitive medical data. Security vulnerabilities must be reported privately to prevent exploitation before a patch is available.

### How to Report

**Email:** [security@liveupx.com](mailto:security@liveupx.com)

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- (Optional) Suggested fix

### Response Timeline

- **Acknowledgement**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Fix Timeline**: Critical issues within 7 days; high within 30 days
- **Public Disclosure**: After patch release + 30 days (coordinated disclosure)

### Scope

In-scope security concerns:
- SQL injection or NoSQL injection vulnerabilities
- Authentication / authorization bypasses
- Patient data leakage or unauthorized access
- Insecure default configurations exposing PHI
- Dependency vulnerabilities in production dependencies
- Cryptographic weaknesses in data-at-rest/in-transit encryption

Out of scope:
- Vulnerabilities in development/test environments only
- Social engineering attacks
- Denial of service (unless causing patient data exposure)

### Security Best Practices for Deployers

1. **Never expose BioSentinel API to the public internet** without authentication
2. **Always use HTTPS (TLS 1.2+)** in production
3. **Rotate API keys** regularly
4. **Enable audit logging** for all patient data access
5. **Keep all dependencies updated** — run `pip audit` regularly
6. **Use strong database passwords** and restrict DB network access
7. **Enable PostgreSQL row-level security** for multi-tenant deployments
8. **Review HIPAA/GDPR compliance** before ingesting real patient data

### Data Security Architecture

BioSentinel is designed with security by default:
- AES-256 encryption for data at rest (when enabled)
- TLS 1.3 for all API communications
- Patient IDs are UUIDs (non-sequential, non-guessable)
- Passwords hashed with bcrypt (cost factor 12+)
- JWT tokens with short expiry (15min access, 7d refresh)
- Audit log for all patient record access

### Responsible Disclosure

BioSentinel follows responsible disclosure practices. Security researchers who responsibly report valid vulnerabilities will be credited in the release notes (with their permission).
