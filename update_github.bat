@echo off
echo ========================================
echo   Updating GitHub Repository
echo ========================================
echo.

echo Adding all changes...
git add .

echo.
echo Enter commit message (or press Enter for default):
set /p commit_msg="Commit message: "

if "%commit_msg%"=="" (
    set commit_msg=Update project files
)

echo.
echo Committing changes...
git commit -m "%commit_msg%"

echo.
echo Pushing to GitHub...
git push origin main

echo.
echo ========================================
echo   Update Complete!
echo ========================================
echo Repository: https://github.com/Cambaki/HydroElectric
echo Status: All changes uploaded successfully
echo.
pause
