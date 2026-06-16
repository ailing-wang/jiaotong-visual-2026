@echo off
set PYTHONPATH=D:\lib\site-packages
python scripts/run_retrieval.py
python scripts/run_detection.py
python scripts/run_evaluation.py
echo "All tasks completed!"
pause