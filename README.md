pip install uv  
uv sync  
data/datasets 생성  
.env 생성 OPENAI_API_KEY=  
source .venv/bin/activate  
PYTHONPATH=. uv run src/rag/ingestion.py  
PYTHONPATH=. uv run streamlit run ui/app.py


