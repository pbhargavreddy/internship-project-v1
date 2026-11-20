#!/bin/bash
cd "$(dirname "$0")"
../internship_venv/bin/python -m uvicorn backend.main:app &
sleep 5
../internship_venv/bin/python -m streamlit run frontend/app.py