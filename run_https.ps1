# Run FastAPI backend with HTTPS using uvicorn
# Make sure you've run setup_ssl.ps1 first to generate certificates

$certsDir = "certs"
$certFile = Join-Path $certsDir "localhost.pem"
$keyFile = Join-Path $certsDir "localhost-key.pem"

# Check if certificates exist
if (-not (Test-Path $certFile) -or -not (Test-Path $keyFile)) {
    Write-Host "❌ SSL certificates not found!" -ForegroundColor Red
    Write-Host "Please run setup_ssl.ps1 first to generate certificates." -ForegroundColor Yellow
    exit 1
}

Write-Host "=== Starting FastAPI Backend with HTTPS ===" -ForegroundColor Green
Write-Host "Server will be available at:" -ForegroundColor Cyan
Write-Host "  - https://127.0.0.1:8443" -ForegroundColor Green
Write-Host "  - https://localhost:8443" -ForegroundColor Green
Write-Host ""
Write-Host "Server is bound to 127.0.0.1 (localhost only)" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & "venv\Scripts\Activate.ps1"
}

# Run uvicorn with SSL - bind to 127.0.0.1
uvicorn main:app `
    --host 127.0.0.1 `
    --port 8443 `
    --ssl-keyfile $keyFile `
    --ssl-certfile $certFile `
    --reload
