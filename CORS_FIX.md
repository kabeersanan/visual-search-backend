# CORS Fix Explanation

## The Error

```
Access to fetch at 'https://127.0.0.1:8443/search' from origin 'https://localhost:5241' 
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present 
on the requested resource.
```

## Why This Happens

**CORS (Cross-Origin Resource Sharing)** is a browser security feature that blocks requests between different origins. An origin is defined by:
- Protocol (http vs https)
- Domain (localhost vs 127.0.0.1 vs example.com)
- Port (5241 vs 8443)

Your frontend at `https://localhost:5241` trying to access backend at `https://127.0.0.1:8443` is a **cross-origin request** because:
- Different domains: `localhost` ≠ `127.0.0.1` (even though they point to the same place)
- Different ports: `5241` ≠ `8443`

## The Fix

The backend now includes CORS middleware that:
1. ✅ Allows requests from `https://localhost:5241` (your frontend)
2. ✅ Sends proper CORS headers (`Access-Control-Allow-Origin`, etc.)
3. ✅ Handles preflight OPTIONS requests automatically
4. ✅ Allows credentials and all HTTP methods/headers

## What Changed

**Before:**
```python
app = FastAPI()  # No CORS configuration
```

**After:**
```python
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://localhost:5241",
    "https://127.0.0.1:5241",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Testing

1. **Restart your backend server** (the CORS middleware needs to load)
2. **Try the request again** from your frontend
3. The CORS error should be gone!

## If Frontend Port Changes

If your Adobe Express Add-on runs on a different port (not 5241), update the `origins` list in `main.py`:

```python
origins = [
    "https://localhost:5241",  # Update port if needed
    "https://localhost:NEW_PORT",  # Add new port
    "https://127.0.0.1:5241",
    # etc.
]
```

## Alternative: More Permissive (Development Only)

For development, you could allow all localhost origins using a regex pattern:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

⚠️ **Warning:** This is less secure and should only be used for local development!

## Production Considerations

For production:
- ✅ Specify exact allowed origins (no wildcards)
- ✅ Restrict `allow_methods` to only what's needed (e.g., `["GET", "POST"]`)
- ✅ Restrict `allow_headers` to only what's needed
- ✅ Consider using environment variables for origins

## Additional Notes

- The `allow_credentials=True` option allows cookies and authorization headers
- FastAPI's CORSMiddleware automatically handles OPTIONS preflight requests
- Make sure there are **no trailing slashes** in origin strings
- The origin must match **exactly** (case-sensitive, includes protocol and port)
