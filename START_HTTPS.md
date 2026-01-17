# Quick Start - HTTPS Backend

## Uvicorn Command with HTTPS

Run your backend with HTTPS using this command:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8443 --ssl-keyfile certs\localhost-key.pem --ssl-certfile certs\localhost.pem --reload
```

**Note:** `--host 127.0.0.1` means the server listens on localhost only. Access it using `https://127.0.0.1:8443` or `https://localhost:8443`.

### Or use the provided script:
```powershell
.\run_https.ps1
```

## Access Points

⚠️ **IMPORTANT:** Use `localhost` or `127.0.0.1` to access the server, NOT `0.0.0.0`!

- **HTTPS API**: https://localhost:8443 or https://127.0.0.1:8443
- **Health Check**: https://localhost:8443/health
- **Search Endpoint**: https://localhost:8443/search

**Do NOT use:** `https://0.0.0.0:8443` (this will give ERR_ADDRESS_INVALID)

## For Frontend

Update your frontend API URL to:
```typescript
const API_URL = 'https://localhost:8443/search';
```
