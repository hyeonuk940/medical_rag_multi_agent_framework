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
            "retrieved_docs": [],
            "checklist": {},
            "next_step": "",
            "scenario_theme": "",
            "scenario_details": {},
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
                    docs = output.get('retrieved_docs', [])
                    doc_preview = docs[0][:300] if docs else "No docs"
                    print(f"     Retrieved medical information:\n   {doc_preview}...")
                    print(f"     medical analysis:\n   {output['medical_info'][:150]}...") 

                if node_name == "patient_agent" and "messages" in output:
                    print(f"    Patient's response: {output['messages'][-1].content}")

                final_state.update(output)
        
        print("="*60 + "\n")
        return final_state
    
if __name__ == "__main__":
    patient_model = "gpt-4o"
    medical_brain_model = "gpt-4o"
    evaluator_model = "gpt-4o"
    controller = Controller(patient_model_name=patient_model, medical_brain_model_name=medical_brain_model, evaluator_model_name=evaluator_model)
    state = controller.get_initial_state()

    print("medical RAG Multi-Agent Framework CLI 실행")
    while True:
        if not state.get("scenario_theme"):
            print("시나리오 테마를 선택해 주세요(1: 죽음에 대해 알리는 상황, 2: 존엄한 선택을 돕는 상황)")
            scenario_input = input(">>")
            if scenario_input == "1":
                state["scenario_theme"] = "breaking_bad_news"
                print("선택된 시나리오 테마: 죽음에 대해 알리는 상황")
            elif scenario_input == "2":
                state["scenario_theme"] = "dignified_choice"
                print("선택된 시나리오 테마: 존엄한 선택을 돕는 상황")
            else:
                print("잘못된 입력입니다. 1 또는 2를 입력해 주세요.")
                continue

        if not state.get("scenario_details"):
            print("시나리오 세부 정보를 각 차례에 맞게 입력해 주세요")
            print("환자 나이")
            age = input(">>")
            print("환자 성별")
            gender = input(">>")
            print("환자의 기본적인 인적 사항")
            identity_details = input(">>")
            print("환자의 성격")
            personality = input(">>")
            print("환자의 상태및 주요 증상")
            condition = input(">>")
            print("환자의 가족 관계")
            family = input(">>")
            print("추가 세부 정보")
            additional_details = input(">>")

            state["scenario_details"] = {
                "patient_identity": age + gender + identity_details,
                "patient_personality": personality,
                "current_condition": condition,
                "family_context": family,
                "additional_notes": additional_details 
            }
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit", "q", "Q"]:
            break

        elif user_input.lower() in ["reset", "r", "R"]:
            state = controller.get_initial_state()
            print("대화가 초기화되었습니다.")
            continue
        
        state = controller.process_turn(user_input, state)
        print(f":{state['messages'][-1].content}\n")