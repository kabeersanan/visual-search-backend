# SSL Certificate Setup Script for Visual Search Backend
# This script sets up mkcert and generates SSL certificates for local development

Write-Host "=== SSL Certificate Setup ===" -ForegroundColor Green

# Check if mkcert is installed
$mkcertInstalled = Get-Command mkcert -ErrorAction SilentlyContinue

if (-not $mkcertInstalled) {
    Write-Host "`n mkcert is not installed. Installing..." -ForegroundColor Yellow
    
    # Check if Chocolatey is installed
    $chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue
    
    if ($chocoInstalled) {
        Write-Host "Installing mkcert via Chocolatey..." -ForegroundColor Cyan
        choco install mkcert -y
    } else {
        Write-Host "`n Chocolatey not found. Please install mkcert manually:" -ForegroundColor Yellow
        Write-Host "1. Download from: https://github.com/FiloSottile/mkcert/releases" -ForegroundColor Cyan
        Write-Host "2. Or install Chocolatey first: https://chocolatey.org/install" -ForegroundColor Cyan
        Write-Host "3. Then run: choco install mkcert -y" -ForegroundColor Cyan
        Write-Host "`n Press any key to continue after installing mkcert..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
    
    # Verify installation
    $mkcertInstalled = Get-Command mkcert -ErrorAction SilentlyContinue
    if (-not $mkcertInstalled) {
        Write-Host "`n ERROR: mkcert installation failed. Please install it manually." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n mkcert is installed!" -ForegroundColor Green

# Install the local CA (if not already installed)
Write-Host "`n Installing local CA (Certificate Authority)..." -ForegroundColor Cyan
mkcert -install

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to install local CA. You may need to run this script as Administrator." -ForegroundColor Yellow
}

# Create certs directory if it doesn't exist
$certsDir = "certs"
if (-not (Test-Path $certsDir)) {
    New-Item -ItemType Directory -Path $certsDir | Out-Null
    Write-Host "`n Created certs directory" -ForegroundColor Green
}

# Generate certificate for localhost and 127.0.0.1
Write-Host "`n Generating SSL certificates..." -ForegroundColor Cyan
Write-Host "Certificate will be valid for: localhost, 127.0.0.1, ::1" -ForegroundColor Cyan

Set-Location $certsDir
mkcert -key-file localhost-key.pem -cert-file localhost.pem localhost 127.0.0.1 ::1

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ SSL certificates generated successfully!" -ForegroundColor Green
    Write-Host "   - Certificate: certs\localhost.pem" -ForegroundColor Cyan
    Write-Host "   - Private Key: certs\localhost-key.pem" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Failed to generate certificates" -ForegroundColor Red
    exit 1
}

Set-Location ..

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "`nYou can now run your backend with HTTPS using:" -ForegroundColor Yellow
Write-Host "uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs\localhost-key.pem --ssl-certfile certs\localhost.pem" -ForegroundColor Cyan
Write-Host "`nOr use the provided run_https.ps1 script." -ForegroundColor Yellow
