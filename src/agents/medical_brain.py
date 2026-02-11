import os
from src.rag.retriever import MedicalRetriever
from src.graph.state import AgentState

class MedicalBrain:  # Medical Information Retrieval Agent
    def __init__ (self, db_path: str):
        self.retriever = MedicalRetriever(db_path)

    def __call__ (self, state: AgentState):
        print("MedicalBrain: Retrieving relevant medical information...")
        last_message = state["messages"][-1].content
        retrieved_knowledge = self.retriever.retrieve_context(last_message, k=3)

        return {
            "medical_info": retrieved_knowledge,
            "next_step": "patient"
        }
    
if __name__ == "__main__":    # Example usage
    from langchain_core.messages import HumanMessage

    node_state = {
        "messages": [HumanMessage(content="고혈압의 일반적인 치료 방법은 무엇인가요?")],
        "medical_info": "",
        "checklist": {},
        "next_step": ""
    }

    brain = MedicalBrain(db_path='data/chroma_db')
    result = brain(node_state)
    print("Retrieved Medical Information:\n", result["medical_info"])