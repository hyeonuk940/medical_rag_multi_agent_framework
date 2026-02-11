import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class MedicalRetriever:
    def __init__ (self, db_path: str):
        self.db_path = db_path
        self. embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        if os.path.exists(self.db_path):
            self.vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            print("Vector database loaded successfully.")
        else:
            raise FileNotFoundError(f"Vector database not found at {self.db_path}. Please run the ingestion process first.")
            self.vector_db = None
    
    def retrieve(self, query: str, k: int = 5) -> str:
        if not self.vector_db:
            raise ValueError("Vector database is not initialized.")
        
        docs = self.vector_db.similarity_search(query, k=k)
        combined_content = "\n\n".join([doc.page_content for doc in docs])
        return combined_content
    
    def retrieve_with_scores(self, query: str, k: int = 5):
        return self.vector_db.similarity_search_with_score(query, k=k)
    
if __name__ == "__main__":
    db_path = 'data/chroma_db'  # Path to the vector database
    retriever = MedicalRetriever(db_path)
    
    sample_query = "고혈압의 일반적인 치료 방법은 무엇인가요?"
    results = retriever.retrieve(sample_query, k=3)
    results_with_scores = retriever.retrieve_with_scores(sample_query, k=3)
    print("Retrieved Documents:\n", results)
    print("Retrieved Documents with Scores:\n", results_with_scores)