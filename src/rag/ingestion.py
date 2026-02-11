import os
import json
import time
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm
from openai import RateLimitError

load_dotenv()

class JsonMedicalDataIngestor: # Class to handle ingestion of medical JSON data
    def __init__(self, data_path: str, db_path: str):
        self.data_path = data_path
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def run_ingestion(self): # Main method to run the ingestion process
        documents = self._load_json_files()

        if not documents:
            print("No documents found to ingest.")
            return
        
        print(f"Loaded {len(documents)} documents from JSON files.")
        print("Creating embeddings and storing in vector database...")

        batch_size = 50  
        vector_db = None
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Creating vector database"):
            batch = documents[i:i + batch_size]
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if vector_db is None:
                        vector_db = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=self.db_path
                        )
                    else:
                        vector_db.add_documents(batch)
                    break  
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  
                        print(f"\nRate limit reached. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"\nFailed after {max_retries} attempts. Error: {e}")
                        raise
            
            time.sleep(0.5)
        
        print("Finished creating vector database.")
        
    def _load_json_files(self): # Load and parse JSON files from the data directory
        all_docs = []
        file_list = []
        for root, dirs, files in os.walk(self.data_path):
            for file_name in files:
                if file_name.endswith('.json'):
                    file_list.append(os.path.join(root, file_name))
        print(f"Found {len(file_list)} JSON files to process.")

        for file_path in tqdm(file_list, desc="Processing JSON files"):
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
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
                print(f"\nError decoding JSON from file {file_path}: {e}")
        return all_docs

if __name__ == "__main__":
    data_path = 'data/datasets'  # Path to the directory containing JSON files
    db_path = 'data/chroma_db'  # Path to store the vector database

    ingestor = JsonMedicalDataIngestor(data_path, db_path)
    ingestor.run_ingestion()
