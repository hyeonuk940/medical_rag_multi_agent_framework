"""
Medical Brain Agent using SNUH HARI Model
Citation:
Healthcare AI Research Institute(HARI) of Seoul National University Hospital(SNUH). (2025). 
hari-q2.5-thinking. https://huggingface.co/snuh/hari-q2.5-thinking
"""

import os
from src.rag.retriever import MedicalRetriever
from src.graph.state import AgentState
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_openai import ChatOpenAI
import torch
from deep_translator import GoogleTranslator
from src.agents.retriever import retrieve_medical_info
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class MedicalBrain:  # Medical Information Retrieval Agent
    def __init__ (self, model_name = None, db_path="data/chroma_db"):
        self.model_name = model_name
        # self.retriever = MedicalRetriever(db_path)
        self.model = None
        self.tokenizer = None

        local_model_path = "models/hari-q2.5-thinking"

        if model_name:
            if model_name == "snuh/hari-q2.5-thinking":
                target_path = local_model_path if os.path.exists(local_model_path) else model_name
                print(f"Initializing local SNUH HARI model from: {target_path}")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)

                max_memory = {0: "16GiB", "cpu": "30GiB"}
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    target_path, 
                    quantization_config=bnb_config, 
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory=max_memory,
                    trust_remote_code=True
                )
            else:
                print(f"Initializing OpenAI model: {model_name}")
                self.model = ChatOpenAI(model_name=model_name, temperature=0).bind_tools([retrieve_medical_info], parallel_tool_calls=False)
        
    def _generate_reasoning(self, prompt):
        if not self.model:
            return None
        if self.tokenizer:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512)
            
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        else:
            response = self.model.invoke(prompt)
            return response.content

    def __call__ (self, state: AgentState):
        print("MedicalBrain: Retrieving relevant medical information...")
        scenario_theme = state.get("scenario_theme", "")
        if scenario_theme == "breaking_bad_news":
            scenario_theme_exp = "환자에게 죽음에 대해 알리는 상황"
        elif scenario_theme == "dignified_choice":
            scenario_theme_exp = "존엄한 선택을 돕는 상황"
        elif scenario_theme == "final_moments":
            scenario_theme_exp = "환자의 임종 직전 환자의 가족과의 대화 상황"
        scenario_details = state.get("scenario_details", {})
        messages = state["messages"]
        last_human_msg = state["messages"][-1].content
        if len(messages) >= 2:
            last_ai_msg = messages[-2].content
            search_query = f"{scenario_theme_exp} {last_ai_msg} {last_human_msg} {scenario_details}"
        else:
            search_query = f"{scenario_theme_exp} {last_human_msg} {scenario_details}"
        messages_data = "\n".join([
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" 
            for msg in messages
        ])
        last_msg = messages[-1]
        retrieved_knowledge = ""
        if last_msg.type == "tool":
            print("MedicalBrain: 에이전트가 스스로 검색을 수행하도록 지시했습니다. 검색 도구를 사용하여 의료 정보를 검색합니다...")

            pass
        else :
            print("MedicalBrain: 초기 검색을 수행합니다...")
            
            retrieved_knowledge = retrieve_medical_info.invoke({"query": search_query, "k": 5})

            if isinstance(retrieved_knowledge, str):
                retrieved_knowledge = [retrieved_knowledge]
        
        if not self.model_name or not retrieved_knowledge:
            print("No model or retrieved knowledge available, skipping reasoning step.")
            return{
                "medical_info": retrieved_knowledge,
                "retrieved_knowledge": retrieved_knowledge,
                "next_step": "patient"
            }
        print("MedicalBrain: Generating reasoning based on retrieved information...")

        prompt = """
        ### Instruction:
        당신은 완화의료 전문가입니다 {scenario_theme_exp}에서 환자가 어떤 증상을 보일 가능성이 높은지, 그리고 환자가 어떤 감정과 반응을 보일 가능성이 높은지를 분석하는 것이 당신의 역할입니다. 검색된 [의학 정보]의 내용 중 [대화 내용]과 [환자 정보]에 알맞은 정보를 참고하여 분석 질문에 대답하세요.
        [의학 정보]
        {retrieved_knowledge}
        [환자 정보]
        {scenario_details}
        [대화 내용]
        {messages_data}
        ### 분석 질문 :
        1. 검색된 [의학 정보]를 바탕으로 현재 상황과 대화 내용에 관련성이 높은 정보들을 요약하고 환자 입장에서의 핵심을 분석하세요.
        2. 현재 상황과 [환자 정보]및 [대화 내용]을 바탕으로 환자가 보일 가능성이 높은 핵심 증상, 특징 그리고 감정 및 반응을 분석하세요.
        3. 대화의 맥락속에서 이번 의사에 질문에 대한 이 환자가 보일 가능성이 높은 반응은 무엇인가요?
        3-1. 심리적 방어 기제 분석 : 환자가 현재 '부정', '분노', '타협', '우울', '수용' 중 어느 단계에 있는지 분석하고, 그 단계에서 흔히 나타나는 '말투의 특징'을 정의하세요.
        3-2. RAG 데이터의 투영 : 환자의 성격과 [의학 정보]에 의해 어떠한 행동과 반응이 유발될 수 있는지 분석하세요. 예를 들어, 환자가 완화의료 상황에서 보이는 일반적인 행동 패턴과 감정적 반응을 설명할 수 있습니다.(대화 예제를 찾아볼 수도 있습니다.)
        #####
        [행동 지침]
        'retrieve_medical_info' 도구를 사용하여 필요한 완화의료 지식을 검색하고 분석에 활용하세요.
        이미 검색된 [의학 정보]가 충분하다면, 추가 검색 없이 분석을 진행할 수 있습니다.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder(variable_name="messages")
        ])

        chain = prompt | self.model
        response = chain.invoke({
            "messages": state["messages"],
            "scenario_theme_exp": scenario_theme_exp,
            "retrieved_knowledge" : retrieved_knowledge,
            "scenario_details": scenario_details,
            "messages_data": messages_data
        })

        medical_info_text = ""
        if not response.tool_calls:
            medical_info_text = response.content
            
        return {
            "messages": [response],
            "scenario_theme_exp": scenario_theme_exp,
            "medical_info": medical_info_text,
            "retrieved_docs": retrieved_knowledge,
            "next_step": "patient"
        }
    
if __name__ == "__main__":    # Example usage
    from langchain_core.messages import HumanMessage

    node_state = {
        "messages": [HumanMessage(content="")],
        "medical_info": "",
        "checklist": {},
        "next_step": ""
    }

    brain = MedicalBrain(db_path='data/chroma_db')
    result = brain(node_state)
    print("Retrieved Medical Information:\n", result["medical_info"])