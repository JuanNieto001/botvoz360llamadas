@echo off
cd /d "%~dp0"
if not exist .env (
  echo No existe el archivo .env. Copia .env.example como .env y pon tus claves.
  pause
  exit /b 1
)
.venv\Scripts\python.exe agent.py
pause
