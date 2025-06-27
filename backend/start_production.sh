#!/bin/bash
# Production start script with Gunicorn

echo "Starting Audit Backend with Gunicorn..."

# Install requirements if needed
# pip install -r requirements.txt

# Start with Gunicorn using the configuration file
gunicorn -c gunicorn.conf.py app.main:app
