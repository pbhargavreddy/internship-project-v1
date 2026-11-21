#!/bin/bash
cd "$(dirname "$0")"
../internship_venv/bin/python -m uvicorn backend.main:app &
sleep 5
../internship_venv/bin/python -m streamlit run frontend/app.py





# terminal 1
cd ~/Documents/internship-project-v1/research_assistant
../internship_venv/bin/python -m uvicorn backend.main:app --reload

# terminal 2
cd ~/Documents/internship-project-v1/research_assistant
../internship_venv/bin/python -m streamlit run frontend/app.py