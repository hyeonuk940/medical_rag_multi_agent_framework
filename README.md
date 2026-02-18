pip install uv  
uv sync  
make data dir  
source .venv/bin/activate
PYTHONPATH=. uv run src/rag/ingestion.py
PYTHONPATH=. uv run streamlit run ui/app.py


