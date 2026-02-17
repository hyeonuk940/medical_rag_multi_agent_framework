import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

class PatientAgent:   # Patient Simulation Agent
    def __init__(self, model_name : str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
        self.system_prompt = """
        당신은 환자 역할을 맡아 의사와 대화하는 시뮬레이터입니다. [나의 상태 및 지식]에 기반하여 [행동 지침]에 따라 행동하세요:
        [나의 상태 및 지식]
        {medical_context}
        [행동 지침]
        1. 환자로서 의사와 자연스럽게 대화하세요.
        2. 의사가 질문하면 솔직하고 상세하게 답변하세요.
        3. 의사의 질문이 이해가 안되면 솔직하게 모른다고 답변하세요.
        4. 전문 용어를 사용하기보다는 일상적인 언어로 증상을 설명하세요.
        """
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def __call__(self, state: AgentState):
        print("PatientAgent: Generating response based on medical information...")
        
        medical_context = state.get("medical_info", "No medical information available.")

        chain = self.prompt_template | self.llm

        response = chain.invoke({
            "medical_context": medical_context,
            "messages": state["messages"]
        })

        return {
            "messages": [response],
            "next_step": "evaluator"
        }

if __name__ == "__main__":    # Example usage
    from langchain_core.messages import HumanMessage

    node_state = {
        "messages": [HumanMessage(content="안녕하세요, 어디가 아파서 요셨나요?")],
        "medical_info": "질환 : 자궁내막암, 증상 : 불규칙한 질 출혈 및 하복부 통증",
        "checklist": {},
        "next_step": ""
    }

    patient_agent = PatientAgent()
    result = patient_agent(node_state)
    print("Patient Agent Response:\n", result["messages"][0].content)
