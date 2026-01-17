@echo off
echo ================================================
echo   Sync2 TTS Service - Starting
echo ================================================

REM Use the existing kokoro-tts venv which has kokoro installed
cd /d "D:\Live Projects\kokoro-tts"
call venv\Scripts\activate

REM Go back to our service directory
cd /d "D:\Live Projects\Sync-AI\sync2-tts-service"

REM Install additional dependencies if needed
pip install soundfile fastapi "uvicorn[standard]" requests websockets -q

REM Run the server
python -m src.server
