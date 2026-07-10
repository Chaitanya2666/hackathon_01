# Streamlit HR Demo

This repository contains a lightweight Streamlit demo app that uses a small sample HR dataset. It is designed for easy deployment and live demoing without shipping the full production dataset.

## Files
- `streamlit_app.py`: Streamlit application code
- `requirements.txt`: dependencies for the demo
- `sample_hr.csv`: sample HR dataset used by the app

## Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## What it demonstrates
- Natural-language-like query input
- SQL preview generation for common HR questions
- Data results and summary output

## Notes
- This repo uses a sample dataset so it can be hosted on Streamlit easily.
- For the full enterprise version, replace `sample_hr.csv` with the production dataset or connect to a managed database.
