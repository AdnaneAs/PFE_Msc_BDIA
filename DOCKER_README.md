# 🐳 Docker Deployment Guide

## PFE Audit System - Dockerized Multi-Worker Setup

This guide provides a complete Docker setup with Redis for multi-worker support, solving Windows multiprocessing issues.

## 🏗️ **Architecture**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frontend  │    │   Backend   │    │    Redis    │
│ (React App) │◄──►│(FastAPI +4W)│◄──►│ (State Mgr) │
│   Port 3000 │    │  Port 8000  │    │  Port 6379  │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🚀 **Quick Start**

### **Prerequisites**
- Docker Desktop for Windows
- Docker Compose v3.8+
- PowerShell 5.1+ (for deployment script)

### **1. Deploy the System**
```powershell
# Method 1: Using deployment script (Recommended)
.\deploy.ps1 -Action start -Build

# Method 2: Manual Docker commands
docker-compose build
docker-compose up -d
```

### **2. Access Services**
- 🌐 **Frontend**: http://localhost:3000
- 🔧 **Backend API**: http://localhost:8000
- 📖 **API Documentation**: http://localhost:8000/docs
- 📊 **Redis**: localhost:6379

### **3. Monitor Services**
```powershell
# Check service status
.\deploy.ps1 -Action status

# View logs
.\deploy.ps1 -Action logs

# Restart services
.\deploy.ps1 -Action restart
```

## 📦 **Service Details**

### **Backend Container**
- **Image**: `python:3.11-slim` (Debian-based)
- **Workers**: 4 (configurable via WORKERS env var)
- **State Management**: Redis-based shared state
- **Health Check**: `/api/hello` endpoint
- **Volumes**: 
  - `./backend/data:/app/data` (persistent data)
  - `./backend/logs:/app/logs` (application logs)

### **Frontend Container**
- **Image**: `node:18-alpine` (Alpine Linux)
- **Build**: Production optimized React build
- **Server**: `serve` static file server
- **Security**: Non-root user execution

### **Redis Container**
- **Image**: `redis:7-alpine` (Alpine Linux)
- **Persistence**: Append-only file (AOF) enabled
- **Volume**: `redis_data` (persistent storage)
- **Health Check**: `redis-cli ping`

## 🛠️ **Configuration**

### **Environment Variables**

#### Backend
```env
REDIS_URL=redis://redis:6379
ENVIRONMENT=production
WORKERS=4
HOST=0.0.0.0
PORT=8000
```

#### Frontend
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=production
```

### **Scaling Workers**
```yaml
# In docker-compose.yml
environment:
  - WORKERS=8  # Increase worker count
```

## 🔧 **Management Commands**

### **Deployment Script** (`deploy.ps1`)
```powershell
# Start services
.\deploy.ps1 -Action start

# Start with fresh build
.\deploy.ps1 -Action start -Build

# Stop services
.\deploy.ps1 -Action stop

# Restart services
.\deploy.ps1 -Action restart

# View real-time logs
.\deploy.ps1 -Action logs

# Check service status
.\deploy.ps1 -Action status

# Clean up everything
.\deploy.ps1 -Action clean
```

### **Docker Compose Commands**
```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Scale backend workers
docker-compose up -d --scale backend=2

# Stop services
docker-compose down

# Remove volumes (reset data)
docker-compose down -v
```

## 📊 **Monitoring & Health**

### **Health Endpoints**
```bash
# Backend health
curl http://localhost:8000/api/hello

# Frontend health
curl http://localhost:3000/

# Redis health
docker-compose exec redis redis-cli ping
```

### **Log Locations**
- **Backend**: `./backend/logs/`
- **Docker Logs**: `docker-compose logs <service>`
- **Redis**: `docker-compose exec redis redis-cli monitor`

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. Port Already in Use**
```powershell
# Check what's using the port
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

#### **2. Redis Connection Issues**
```bash
# Check Redis status
docker-compose exec redis redis-cli ping

# View Redis logs
docker-compose logs redis
```

#### **3. Frontend Build Fails**
```bash
# Clean npm cache
docker-compose exec frontend npm cache clean --force

# Rebuild frontend
docker-compose build --no-cache frontend
```

#### **4. Backend State Issues**
```bash
# Clear Redis state
docker-compose exec redis redis-cli FLUSHALL

# Restart backend
docker-compose restart backend
```

### **Debug Mode**
```powershell
# Start with detailed logging
docker-compose up --no-daemon

# View specific service logs
docker-compose logs -f backend
```

## 🧹 **Maintenance**

### **Update Dependencies**
```bash
# Scan for new dependencies
python scan_dependencies.py

# Rebuild with new requirements
docker-compose build --no-cache backend
```

### **Backup Data**
```bash
# Backup database
docker cp pfe_backend:/app/data ./backup/

# Backup Redis
docker-compose exec redis redis-cli --rdb /data/backup.rdb
```

### **Clean Up**
```powershell
# Remove unused containers/images
.\deploy.ps1 -Action clean

# Full system cleanup
docker system prune -a --volumes
```

## ⚡ **Performance Tips**

1. **Adjust Worker Count**: Start with 4 workers, scale based on CPU cores
2. **Memory Limits**: Add memory limits to containers if needed
3. **Volume Optimization**: Use bind mounts for development, volumes for production
4. **Redis Tuning**: Configure Redis memory limits for production

## 🎯 **Production Considerations**

1. **Security**: Use secrets management for API keys
2. **SSL/TLS**: Add reverse proxy (nginx) for HTTPS
3. **Monitoring**: Add Prometheus/Grafana for metrics
4. **Backups**: Implement automated backup strategy
5. **Scaling**: Consider container orchestration (Kubernetes)

## 📚 **Additional Resources**

- [Docker Documentation](https://docs.docker.com/)
- [Redis Documentation](https://redis.io/documentation)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Deployment Guide](https://create-react-app.dev/docs/deployment/)

---

✅ **Your multi-worker, Redis-backed PFE Audit System is now ready for production!** 🎉
