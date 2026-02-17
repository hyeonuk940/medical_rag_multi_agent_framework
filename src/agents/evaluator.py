import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import AgentState

class EvaluatorAgent:   # Response Evaluation Agent
    def __init__(self, model_name : str = "gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        self.system_prompt = """
        당신은 의사와 환자 간의 대화를 평가하는 전문가입니다. 다음 [체크리스트]에 따라 [응답형식]을 출력하세요:
        [체크리스트]
        [응답 형식]
        json 형식으로 다음과 같이 응답하세요:
        {"checklist": {}}
           
        """

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "현재까지의 대화 기록입니다:\n\n{messages}\n\n현재 체크리스트 상태:\n{current_checklist}"),
        ])

    def __call__(self, state: AgentState):
        print("EvaluatorAgent: Evaluating patient response...")

        current_checklist = state.get("checklist", {
            "greeting": False,
            "symptom_description": False,
            "medical_history": False,
            "treatment_understanding": False ###등등 설정
        })

        chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "messages": chat_history,
            "current_checklist": json.dumps(current_checklist)
        })

        try:
            raw_content = response.content.strip()
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            updated_checklist = json.loads(raw_content)
            for key in current_checklist:
                if updated_checklist.get(key) is True:
                    current_checklist[key] = True

        except Exception as e:
            print(f"Error parsing evaluator response: {e}")
        
        return {
            "checklist": current_checklist,
            "next_step": "end"
        }
    
if __name__ == "__main__":    # Example usage
    
    from langchain_core.messages import HumanMessage, AIMessage
    
    mock_state = {
        "messages": [
            HumanMessage(content="안녕하세요, 학생의사 홍길동입니다. 오늘 어디가 불편하신가요?"),
            AIMessage(content="네 선생님, 가슴이 너무 답답해서 왔어요.")
        ],
        "medical_info": "",
        "checklist": {},
        "next_step": ""
    }
    
    evaluator = EvaluatorAgent()
    result = evaluator(mock_state)