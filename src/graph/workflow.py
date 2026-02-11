from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.medical_brain import MedicalBrain
from src.agents.patient import PatientAgent
from src.agents.evaluator import EvaluatorAgent

brain = MedicalBrain(db_path='data/chroma_db')
patient_agent = PatientAgent(model_name="gpt-4")
evaluator_agent = EvaluatorAgent(model_name="gpt-4")

def buildworkflow():  # Build the multi-agent workflow 
    workflow = StateGraph(AgentState)
    workflow.add_node("medical_brain", brain)
    workflow.add_node("patient_agent", patient_agent)
    workflow.add_node("evaluator_agent", evaluator_agent)

    workflow.set_entry_point("medical_brain")
    workflow.add_edge("medical_brain", "patient_agent")
    workflow.add_edge("patient_agent", "evaluator_agent")
    workflow.add_edge("evaluator_agent", END)

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

    final_state = app.invoke(initial_state)