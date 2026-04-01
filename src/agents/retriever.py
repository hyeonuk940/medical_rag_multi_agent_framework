import os
from src.rag.retriever import MedicalRetriever
from src.graph.state import AgentState
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_openai import ChatOpenAI
import torch
from deep_translator import GoogleTranslator
from langchain_core.tools import tool

@tool
def retrieve_medical_info(query: str, k: int):
    """
    완화의료 전문 지식 및 가이드라인이 필요할 때 사용합니다.
    호스피스 상황에서의 환자 증상 관리, 의사소통 기법, 윤리적 가이드라인 등 다양한 의료 정보를 검색할 수 있습니다.
    """
    retriever = MedicalRetriever("data/chroma_db")
    
    
    translator = GoogleTranslator(source='ko', target='en')
    query_en = translator.translate(query)

    combined_query = f"{query} {query_en}"

    docs = retriever.retrieve_with_scores(combined_query, k=k)
    return docs

        


