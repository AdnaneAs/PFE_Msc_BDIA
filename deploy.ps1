# Docker Deployment Script for Windows
param(
    [string]$Action = "start",
    [switch]$Build = $false,
    [switch]$Logs = $false
)

Write-Host "🐳 PFE Audit System Docker Manager" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

function Show-Status {
    Write-Host "`n📊 Service Status:" -ForegroundColor Cyan
    docker-compose ps
    
    Write-Host "`n🏥 Health Status:" -ForegroundColor Cyan
    try {
        $backendHealth = docker-compose exec backend curl -s http://localhost:8000/api/hello 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Backend: Healthy" -ForegroundColor Green
        } else {
            Write-Host "❌ Backend: Unhealthy" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Backend: Cannot check health" -ForegroundColor Red
    }
}

function Show-Logs {
    Write-Host "`n📋 Recent Logs:" -ForegroundColor Cyan
    docker-compose logs --tail=50 --follow
}

switch ($Action.ToLower()) {
    "start" {
        Write-Host "🚀 Starting PFE Audit System..." -ForegroundColor Yellow
        
        # Create necessary directories
        New-Item -ItemType Directory -Force -Path "backend/data", "backend/logs", "data" | Out-Null
        
        if ($Build) {
            Write-Host "📦 Building Docker images..." -ForegroundColor Yellow
            docker-compose build --no-cache
        }
        
        Write-Host "� Starting services..." -ForegroundColor Yellow
        docker-compose up -d
        
        Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
        
        Show-Status
        
        Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
        Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor White
        Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor White
        Write-Host "📊 Redis: localhost:6379" -ForegroundColor White
        Write-Host "📋 View logs: .\deploy.ps1 -Action logs" -ForegroundColor Gray
    }
    
    "stop" {
        Write-Host "🛑 Stopping PFE Audit System..." -ForegroundColor Yellow
        docker-compose down
        Write-Host "✅ Services stopped!" -ForegroundColor Green
    }
    
    "restart" {
        Write-Host "🔄 Restarting PFE Audit System..." -ForegroundColor Yellow
        docker-compose restart
        Start-Sleep -Seconds 10
        Show-Status
    }
    
    "status" {
        Show-Status
    }
    
    "logs" {
        Show-Logs
    }
    
    "clean" {
        Write-Host "🧹 Cleaning up Docker resources..." -ForegroundColor Yellow
        docker-compose down -v --remove-orphans
        docker system prune -f
        Write-Host "✅ Cleanup complete!" -ForegroundColor Green
    }
    
    default {
        Write-Host "Usage: .\deploy.ps1 -Action <start|stop|restart|status|logs|clean> [-Build] [-Logs]" -ForegroundColor White
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Gray
        Write-Host "  .\deploy.ps1 -Action start -Build    # Build and start" -ForegroundColor Gray
        Write-Host "  .\deploy.ps1 -Action logs             # View logs" -ForegroundColor Gray
        Write-Host "  .\deploy.ps1 -Action status           # Check status" -ForegroundColor Gray
    }
}

if ($Logs -and $Action -eq "start") {
    Show-Logs
}
