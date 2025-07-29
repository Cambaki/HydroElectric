@echo off
echo Initializing Git repository for Hydro Generator Project...
echo.

REM Configure Git (replace with your info)
git config --global user.name "Cambaki"
git config --global user.email "Clambak874@gmail.com"

REM Initialize repository
git init

REM Add all files
git add .

REM Create initial commit
git commit -m "Initial commit: Hydro Generator Dashboard with ML and Engineering Modules"

echo.
echo Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a repository on GitHub.com
echo 2. Copy the repository URL
echo 3. Run: git remote add origin [YOUR_REPOSITORY_URL]
echo 4. Run: git push -u origin main
echo.
pause
