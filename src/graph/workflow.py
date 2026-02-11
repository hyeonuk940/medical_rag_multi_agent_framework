from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.medical_brain import MedicalBrain
from src.agents.patient import PatientAgent

brain = MedicalBrain(db_path='data/chroma_db')
patient_agent = PatientAgent(model_name="gpt-4")

def buildworkflow():  # Build the multi-agent workflow 
    workflow = StateGraph(AgentState)
    workflow.add_node("medical_brain", brain)
    workflow.add_node("patient_agent", patient_agent)

    workflow.set_entry_point("medical_brain")
    workflow.add_edge("medical_brain", "patient_agent")
    workflow.add_edge("patient_agent", END)

    return workflow.compile()

app = buildworkflow()

if __name__ == "__main__":   # Example usage
    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content="안녕하세요, 오늘 어디가 불편해서 오셨나요?")],
        "medical_info": "",
        "checklist": {},
        "next_step": ""
    }

    print("Starting Medical RAG Multi-Agent Workflow...")
    final_state = app.invoke(initial_state)

    print(f"의사: {initial_state['messages'][0].content}")
    print(f"환자: {final_state['messages'][-1].content}")