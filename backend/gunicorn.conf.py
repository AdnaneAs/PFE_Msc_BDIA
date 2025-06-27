# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = min(4, multiprocessing.cpu_count())
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'audit_backend'

# Preload app for better performance
preload_app = True
