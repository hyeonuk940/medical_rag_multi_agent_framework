pip install uv  
uv sync  
data/datasets 생성  
.env 생성 OPENAI_API_KEY=  
source .venv/bin/activate  
PYTHONPATH=. uv run src/rag/ingestion.py  
PYTHONPATH=. uv run streamlit run ui/app.py  

[notion](https://www.notion.so/2ffc5a54e3c08085846ef3670b4c5b2d)


