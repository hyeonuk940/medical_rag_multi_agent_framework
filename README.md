pip install uv
uv sync
source .venv/bin/activate
PYTHONPATH=. uv run streamlit run ui/app.py
