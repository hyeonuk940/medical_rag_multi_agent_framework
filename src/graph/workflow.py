from src.agents.retriever import retrieve_medical_info
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.medical_brain import MedicalBrain
from src.agents.patient import PatientAgent
from src.agents.evaluator import EvaluatorAgent
from langgraph.prebuilt import ToolNode
    

tool_node = ToolNode(tools=[retrieve_medical_info], name="medical_retriever")

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "medical_retriever"
    return "patient_agent"

def buildworkflow(patient_model_name: str = "gpt-40-mini", medical_brain_model_name: str = "gpt-40-mini", evaluator_model_name: str = "gpt-40-mini"):  # Build the multi-agent workflow
    brain = MedicalBrain(model_name=medical_brain_model_name, db_path='data/chroma_db')
    patient_agent = PatientAgent(model_name=patient_model_name)
    evaluator_agent = EvaluatorAgent(model_name=evaluator_model_name)

    workflow = StateGraph(AgentState)
    workflow.add_node("medical_brain", brain)
    workflow.add_node("medical_retriever", tool_node)
    workflow.add_node("patient_agent", patient_agent)
    workflow.add_node("evaluator_agent", evaluator_agent)

    workflow.set_entry_point("medical_brain")
    workflow.add_conditional_edges(
        "medical_brain",
        should_continue,
        {
            "medical_retriever": "medical_retriever",
            "patient_agent": "patient_agent"

        }
    )
    # workflow.add_edge("medical_brain", "patient_agent")
    # workflow.add_edge("patient_agent", "evaluator_agent")
    # workflow.add_edge("evaluator_agent", END)
    workflow.add_edge("medical_retriever", "medical_brain")
    workflow.add_edge("patient_agent", END)

    return workflow.compile()

if __name__ == "__main__":   # Example usage
    from langchain_core.messages import HumanMessage

    app = buildworkflow()

    initial_state = {
        "messages": [HumanMessage(content="안녕하세요, 오늘 어디가 불편해서 오셨나요?")],
        "medical_info": "",
        "checklist": {},
        "next_step": ""
    }

    final_state = app.invoke(initial_state)
    print("Final State:\n", final_state)