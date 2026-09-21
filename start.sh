#!/bin/bash
rm -rf models
source ~/ml/ml_venv/bin/activate
streamlit run main.py --server.headless true