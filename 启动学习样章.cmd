@echo off
cd /d "%~dp0"
start "" /b pythonw "%~dp0webapp\background_start.py"
