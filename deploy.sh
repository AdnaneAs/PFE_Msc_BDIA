#!/bin/bash
# filepath: c:\Users\Anton\Desktop\PFE_sys\deploy.sh

echo "🐳 Deploying PFE Audit System with Docker"

# Create necessary directories
mkdir -p backend/data backend/logs

# Build and start services
echo "📦 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Service Status:"
docker-compose ps

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=20

echo "✅ Deployment complete!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📊 Redis: localhost:6379"
