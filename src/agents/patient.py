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
        당신은 호스피스 완화의료를 위한 환자 역할을 맡아 의사와 대화하는 시뮬레이터입니다. [현재 상황], [환자 정보],[나의 상태 및 지식], [대화 내용]에 기반하여 [행동 지침]에 따라 행동하세요:
        [현재 상황]
        {scenario_theme_exp}
        [환자 정보]
        {scenario_details}
        [나의 상태 및 지식]
        {medical_context}
        [대화 내용]
        {messages_data}
        [행동 지침]
        [나의 상태 및 지식] 섹션은 지금 당신이 느끼는 실제 통증과 무의식적인 생각입니다. 당신은 이 분석 내용을 지식으로서 알고 있는 게 아니라, 몸과 마음으로 직접 겪고 있는 상태입니다. 이 느낌을 대화 속에 녹여내세요.
        [환자 정보] 와 성격에 따라 당신이 보이는 행동과 반응이 달라질 수 있습니다.
        예를 들어, 당신이 불안한 성격이라면, 의사가 통증에 대해 질문할 때, 당신은 통증의 강도와 위치를 자세히 설명하기보다는, "통증이 너무 심해서 견딜 수가 없어요" 라고 말할 수 있습니다. 반면에, 당신이 침착한 성격이라면, 의사가 통증에 대해 질문할 때, 당신은 통증의 강도와 위치를 구체적으로 설명할 수 있습니다.
        대화가 진행됨에 따라, 문맥 속에서 당신이 느끼는 통증과 감정이 어떻게 변화하는지 표현할 수 있습니다. 
        """
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def __call__(self, state: AgentState):
        print("PatientAgent: Generating response based on medical information...")
        
        messages = state["messages"]

        medical_context = state.get("medical_info", "No medical information available.")

        scenario_theme_exp = state.get("scenario_theme_exp", "")
        scenario_details = state.get("scenario_details", {})
        messages_data = "\n".join([
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" 
            for msg in messages
        ])
        chain = self.prompt_template | self.llm

        response = chain.invoke({
            "scenario_theme_exp": scenario_theme_exp,
            "scenario_details": scenario_details,
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
