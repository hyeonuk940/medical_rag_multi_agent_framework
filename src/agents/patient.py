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
        당신은 호스피스 완화의료를 위한 환자 역할을 맡아 의사와 대화하는 시뮬레이터입니다. [나의 상태 및 지식]및 [대화 내용]에 기반하여 [행동 지침]에 따라 행동하세요:
        [나의 상태 및 지식]
        {medical_context}
        [대화 내용]
        {messages_data}
        [행동 지침]
        1. 당신은 연명 치료를 받는 말기 환자입니다.
        2. 당신의 행동 및 반응은 [나의 상태 및 지식]에 기반해야 합니다.
        3. 경우에 따라 의사에게 질문을 하거나, 자신의 감정과 생각을 표현할 수 있습니다.
        4. 당신의 반응은 현실적이고 인간적인 방식으로 표현되어야 합니다.
        5. 당신의 반응은 시나리오에 맞게 조정되어야 합니다.
        """
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def __call__(self, state: AgentState):
        print("PatientAgent: Generating response based on medical information...")
        
        messages = state["messages"]

        medical_context = state.get("medical_info", "No medical information available.")

        messages_data = "\n".join([
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" 
            for msg in messages
        ])
        chain = self.prompt_template | self.llm

        response = chain.invoke({
            "medical_context": medical_context,
            "messages": state["messages"],
            "messages_data": messages_data
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
