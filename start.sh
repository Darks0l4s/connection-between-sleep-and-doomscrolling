#!/bin/bash
if [ "$1" == "--reset" ]; then
    rm -rf models
fi
if ["$1" == "--test"]; then
    pytest tests/ --cov=src --cov=app --cov-report=term-missing
fi
source SleepVenv/bin/activate
streamlit run main.py --server.headless true