import os
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class JsonMedicalDataIngestor: # Class to handle ingestion of medical JSON data
    def __init__(self, data_path: str, db_path: str):
        self.data_path = data_path
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def run_ingestion(self): # Main method to run the ingestion process
        documents = self._load_and_json_files()
        print(f"Loaded {len(documents)} documents from JSON files.")
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("finished creating vector database.")
        vector_db.persist() # Persist the database to disk

    def _load_and_json_files(self): # Load and parse JSON files from the data directory
        all_docs = []
        for root, dirs, files in os.walk(self.data_path):
            for file_name in files:
                if file_name.endswith('.json'):
                    file_path = os.path.join(root, file_name)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            content = f"질문: {data.get('question')}\n답변: {data.get('answer')}"
                            metadata = {
                                "qa_id": data.get("qa_id"),
                                "domain": data.get("domain"),
                                "q_type": data.get("q_type"),
                                "source": file_path
                            }

                            all_docs.append(Document(page_content=content, metadata=metadata))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from file {file_path}: {e}")
        return all_docs

if __name__ == "__main__":
    data_path = 'data/datasets'  # Path to the directory containing JSON files
    db_path = 'data/chroma_db'  # Path to store the vector database

    ingestor = JsonMedicalDataIngestor(data_path, db_path)
    ingestor.run_ingestion()
