@echo off
REM Production start script with multiple workers for Windows

echo Starting Audit Backend with multiple workers...

REM Install requirements if needed
pip install -r requirements.txt

REM Start using the production server script
python server.py
