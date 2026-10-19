@echo off
:: Navigate to project root directory
cd /d "%~dp0..\.." 2>nul || cd /d "%~dp0.." 2>nul || cd /d "%~dp0"
echo ====================================================
echo Starting Chemical Reaction Yield Predictor...
echo Project Folder: %CD%
echo ====================================================
echo Step 1: Installing required packages...
pip install -r requirements.txt
echo.
echo Step 2: Launching Streamlit frontend...
streamlit run frontend/app.py
pause
