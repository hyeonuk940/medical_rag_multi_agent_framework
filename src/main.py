import os
from langchain_core.messages import HumanMessage
from src.graph.workflow import buildworkflow
from dotenv import load_dotenv

load_dotenv()

class Controller:
    def __init__(self, patient_model_name: str, medical_brain_model_name: str, evaluator_model_name: str):
        self.app = buildworkflow(patient_model_name=patient_model_name, medical_brain_model_name=medical_brain_model_name, evaluator_model_name=evaluator_model_name)

    @staticmethod
    def get_initial_state():
        return {
            "messages": [],
            "medical_info": "",
            "checklist": {},
            "next_step": "",
            "current_scenario": "자궁내막암 환자 시나리오"
        }

    def process_turn(self, user_input: str, current_state: dict):
        current_state["messages"].append(HumanMessage(content=user_input))
        # updated_state = self.app.invoke(current_state)

        # return updated_state
        final_state = current_state
        for chunk in self.app.stream(current_state):
            for node_name, output in chunk.items():
                print(f"\n [Node: {node_name}] is finished processing.")
                
                if node_name == "medical_brain" and "medical_info" in output:
                    print(f"     medical analysis:\n   {output['medical_info'][:150]}...") 

                if node_name == "patient_agent" and "messages" in output:
                    print(f"    Patient's response: {output['messages'][-1].content}")

                final_state.update(output)
        
        print("="*60 + "\n")
        return final_state
    
if __name__ == "__main__":
    patient_model = "gpt-4o-mini"
    medical_brain_model = "snuh/hari-q2.5-thinking"
    evaluator_model = "gpt-4o-mini"
    controller = Controller(patient_model_name=patient_model, medical_brain_model_name=medical_brain_model, evaluator_model_name=evaluator_model)
    state = controller.get_initial_state()

    print("medical RAG Multi-Agent Framework CLI 실행")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit", "q", "Q"]:
            break
        
        state = controller.process_turn(user_input, state)
        print(f":{state['messages'][-1].content}\n")