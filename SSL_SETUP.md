# SSL Certificate Setup for Local Development

This guide helps you set up HTTPS for your local backend using mkcert.

## Quick Start

### 1. Install mkcert

**Option A: Using Chocolatey (Recommended for Windows)**
```powershell
# Install Chocolatey first (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install mkcert
choco install mkcert -y
```

**Option B: Manual Installation**
1. Download mkcert from: https://github.com/FiloSottile/mkcert/releases
2. Download `mkcert-v1.4.4-windows-amd64.exe` (or latest version)
3. Rename it to `mkcert.exe` and add it to your PATH

### 2. Install Local CA (Certificate Authority)

Run PowerShell **as Administrator**:
```powershell
mkcert -install
```

This installs mkcert's root certificate so your browser trusts locally-generated certificates.

### 3. Generate Certificates

Run the setup script:
```powershell
.\setup_ssl.ps1
```

Or manually:
```powershell
# Create certs directory
mkdir certs

# Generate certificates
cd certs
mkcert -key-file localhost-key.pem -cert-file localhost.pem localhost 127.0.0.1 ::1
cd ..
```

This will create:
- `certs/localhost.pem` - Certificate file
- `certs/localhost-key.pem` - Private key file

## Running Backend with HTTPS

### Method 1: Using the provided script
```powershell
.\run_https.ps1
```

### Method 2: Manual uvicorn command
```powershell
uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/localhost-key.pem --ssl-certfile certs/localhost.pem --reload
```

### Method 3: Using uvicorn with Python
```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/localhost-key.pem --ssl-certfile certs/localhost.pem --reload
```

## Access Your Backend

After starting the server, access it at:
- **HTTPS**: https://localhost:8443
- **Health Check**: https://localhost:8443/health
- **Search Endpoint**: https://localhost:8443/search

## Frontend Configuration

Update your frontend to use the HTTPS endpoint:

```typescript
// In your frontend code (VisualSearch.ts or similar)
const API_URL = 'https://localhost:8443/search';
```

**Important**: If your frontend is served from Adobe Express (which uses HTTPS), it can only make requests to HTTPS endpoints due to mixed content policies.

## Troubleshooting

### Certificate Errors in Browser
- Make sure you ran `mkcert -install` as Administrator
- Clear your browser cache and restart the browser
- Verify the certificate is trusted by checking certificate details in the browser

### Port Already in Use
If port 8443 is in use, change it:
```powershell
uvicorn main:app --host 0.0.0.0 --port 8444 --ssl-keyfile certs/localhost-key.pem --ssl-certfile certs/localhost.pem
```

### Permission Denied
- Run PowerShell as Administrator when installing mkcert CA
- Make sure you have write permissions in the project directory

### mkcert Not Found
- Ensure mkcert is in your PATH
- Or use full path to mkcert.exe when running commands

## Certificate Files

The generated certificates are valid for:
- `localhost`
- `127.0.0.1`
- `::1` (IPv6 localhost)

To add more domains/IPs, modify the mkcert command:
```powershell
mkcert -key-file localhost-key.pem -cert-file localhost.pem localhost 127.0.0.1 ::1 your-domain.local
```

## Security Note

⚠️ **These certificates are for local development only!** They should never be used in production. For production, use certificates from a trusted Certificate Authority (CA) like Let's Encrypt.

## Gitignore

Make sure to add certificates to `.gitignore`:
```
certs/
*.pem
*.key
*.crt
```
