@echo off
echo Initializing Git repository...
git init

echo Adding files...
git add .

echo Committing...
git commit -m "Initial commit"

echo Renaming branch to main...
git branch -M main

echo Adding remote origin...
git remote add origin https://github.com/tkaringa/Thiruka.git

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause