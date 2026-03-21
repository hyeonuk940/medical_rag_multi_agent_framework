import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

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
        
        initial_docs = self.vector_db.similarity_search(query, k=k)

        if not initial_docs:
            print("No relevant documents found for the query.")
            return ""

        seen_chapters = set()
        final_contents = []

        for doc in initial_docs:
            target_book = doc.metadata.get("book_title")
            target_chapter = doc.metadata.get("chapter")

            if not target_book or not target_chapter:
                if doc.page_content not in final_contents:
                    final_contents.append(doc.page_content)
                continue

            if (target_book, target_chapter) in seen_chapters:
                continue
            print(f"Searching full context for: [{target_book}] > [{target_chapter}]")
            seen_chapters.add((target_book, target_chapter))

            full_context_data = self.vector_db.get(
                where={
                    "$and": [
                        {"book_title": {"$eq": target_book}},
                        {"chapter": {"$eq": target_chapter}}
                    ]
                }
            )
            
            chapter_docs = full_context_data.get('documents', [])
            if chapter_docs:
                final_contents.extend(chapter_docs)

        if not final_contents:
            return ""

        unique_contents = list(dict.fromkeys(final_contents))

        combined_content = "\n\n".join(unique_contents)
        return combined_content
    
    def retrieve_with_scores(self, query: str, k: int = 5):
        return self.vector_db.similarity_search_with_score(query, k=k)
    
if __name__ == "__main__":
    db_path = 'data/chroma_db'  # Path to the vector database
    retriever = MedicalRetriever(db_path)
    
    sample_query = "환자에 대한 임종 판단"
    results = retriever.retrieve(sample_query, k=3)
    # results_with_scores = retriever.retrieve_with_scores(sample_query, k=3)
    print("Retrieved Documents:\n", results)
    # print("Retrieved Documents with Scores:\n", results_with_scores)