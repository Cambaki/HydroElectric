@echo off
echo Setting up Git configuration...
git config --global user.name "Cambaki"
git config --global user.email "Clambak874@gmail.com"
echo.
echo Git configuration completed!
echo Username: Cambaki
echo Email: Clambak874@gmail.com
echo.
echo Verifying configuration:
git config --global --list | findstr "user"
pause
