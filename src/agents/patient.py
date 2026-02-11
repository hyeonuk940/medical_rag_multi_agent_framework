import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.graph.state import AgentState

class PatientAgent:   # Patient Simulation Agent
    def __init__(self, model_name : str = "gpt-"):
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
        self.system_prompt = """
        ~~~~~~~~~~~~~~~~~~~~~~~
        [나의 상태 및 지식]
        {medical_context}
        [행동 지침]
        ~~~~~~~~~~~~~~~~~~~~~~~
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
