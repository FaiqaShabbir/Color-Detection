@echo off
echo Advanced Color Detection - Test Runner
echo =====================================

echo.
echo 1. Running test suite...
python test_color_detection.py

echo.
echo 2. Testing the improved application with sample image...
echo    (Press ESC to exit the application)
python improved_color_detection.py -i colorpic.jpg

echo.
echo Tests completed!
pause 