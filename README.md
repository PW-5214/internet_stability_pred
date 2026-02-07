# Internet Stability Predictor (Streamlit)

This repository contains a Streamlit app `app.py` converted from the notebook `UI of Project.ipynb`.

Quick start (local):

1. Create a virtual environment (optional):

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run "app.py"
```

Deployment:
- Push this repo to GitHub and deploy on Streamlit Community Cloud (https://share.streamlit.io/) by connecting the repo. The service will use `requirements.txt` and the `app.py` entrypoint.

Notes:
- The app trains several sklearn models on `Internet Speed.csv` at startup (cached). If a model fails to train it will be omitted from the selector.
- If you want to avoid retraining every time, pre-train and persist model objects and load them instead.
