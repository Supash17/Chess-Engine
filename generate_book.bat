@echo off
echo ====================================
echo Chess Engine - Opening Book Generator
echo ====================================
echo.
echo This will generate a 50,000 position opening book with depth=5 search
echo Estimated time: 30-60 minutes
echo.
pause

python generate_opening_book.py

echo.
echo ====================================
echo Opening book generation complete!
echo You can now run: python app.py
echo ====================================
pause
