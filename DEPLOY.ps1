# ============================================
# COMPLETE DEPLOYMENT - RUN THIS ONE SCRIPT
# ============================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host "RESUME SYSTEM - COMPLETE DEPLOYMENT" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green
Write-Host ""

# STEP 1: Stop MySQL
Write-Host "[1] Stopping MySQL..." -ForegroundColor Cyan
docker-compose down -v
Write-Host "  ? MySQL stopped" -ForegroundColor Green

# STEP 2: Start MySQL
Write-Host ""
Write-Host "[2] Starting MySQL..." -ForegroundColor Cyan
docker-compose up -d
Write-Host "  Waiting 15 seconds..."
Start-Sleep -Seconds 15
docker ps

$mysql = docker ps --filter "name=resume_mysql" --format "{{.Names}}"
if ($mysql -ne "resume_mysql") {
    Write-Host "  ? MySQL failed to start!" -ForegroundColor Red
    exit 1
}
Write-Host "  ? MySQL running" -ForegroundColor Green

# STEP 3: Apply schema
Write-Host ""
Write-Host "[3] Creating database schema..." -ForegroundColor Cyan

# Use root account to apply schema (uses values from docker-compose)
Get-Content "schema.sql" | docker exec -i resume_mysql mysql -u root -prootpassword resume_processing

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ? Schema created" -ForegroundColor Green
} else {
    Write-Host "  ? Schema creation failed!" -ForegroundColor Red
    exit 1
}

# STEP 4: Verify resume table
Write-Host ""
Write-Host "[4] Verifying schema..." -ForegroundColor Cyan

$cols = docker exec -i resume_mysql mysql -u root -prootpassword resume_processing -e "DESCRIBE resume;" 2>$null

if ($cols -match "employee_jobdiva_id") {
    Write-Host "  ? resume table verified (employee_jobdiva_id column exists)" -ForegroundColor Green
} else {
    Write-Host "  ? resume table or employee_jobdiva_id column MISSING!" -ForegroundColor Red
    exit 1
}

# STEP 5: Test Python connection
Write-Host ""
Write-Host "[5] Testing Python?MySQL connection..." -ForegroundColor Cyan

$testCode = @"
import mysql.connector
import os
from dotenv import load_dotenv
load_dotenv()
conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST'),
    port=int(os.getenv('MYSQL_PORT')),
    database=os.getenv('MYSQL_DATABASE'),
    user=os.getenv('MYSQL_USER'),
    password=os.getenv('MYSQL_PASSWORD')
)
conn.close()
print('OK')
"@

$result = $testCode | python 2>$null
if ($result -eq "OK") {
    Write-Host "  ? Python can connect to MySQL" -ForegroundColor Green
} else {
    Write-Host "  ? Python cannot connect to MySQL!" -ForegroundColor Red
    Write-Host "  Check .env file" -ForegroundColor Yellow
    exit 1
}

# DONE
Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Edit test_single_resume.py with your resume IDs" -ForegroundColor White
Write-Host "  2. Run: python test_single_resume.py" -ForegroundColor White
Write-Host "  3. Run: python scripts\verify_db.py" -ForegroundColor White
Write-Host ""
