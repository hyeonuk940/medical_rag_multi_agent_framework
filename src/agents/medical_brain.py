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


class MedicalBrain:  # Medical Information Retrieval Agent
    def __init__ (self, model_name = None, db_path="data/chroma_db"):
        self.model_name = model_name
        self.retriever = MedicalRetriever(db_path)
        self.model = None
        self.tokenizer = None

        if model_name:
           if model_name:
            if model_name == "snuh/hari-q2.5-thinking":
                print(f"Initializing local model: {model_name}")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config=bnb_config, 
                dtype=torch.bfloat16,        # dtype으로 이름을 바꿔줍니다.
                device_map="auto",
                trust_remote_code=True
            )
            else:
                print(f"Initializing OpenAI model: {model_name}")
                self.model = ChatOpenAI(model_name=model_name, temperature=0)
        
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
        scenario = state.get("current_scenario", "")
        messages = state["messages"]
        last_human_msg = state["messages"][-1].content
        if len(messages) >= 2:
            last_ai_msg = messages[-2].content
            search_query = f"{scenario} {last_ai_msg} {last_human_msg}"
        else:
            search_query = f"{scenario} {last_human_msg}"
        retrieved_knowledge = self.retriever.retrieve(search_query, k=3)

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

        prompt = f"""### Instruction:
        당신은 임상 지식을 갖춘 유능하고 신뢰할 수 있는 한국어 기반 의료 어시스턴트 입니다. [의학 정보]는 환자의 상태에 대하여 검색한 관련 정보들 입니다 이중 [현재 시나리오]에 적합한 정보를 바탕으로 환자의 현재 상태를 분석하세요.
        분석 결과는 '환자 에이전트'가 연기할 수 있도록 핵심 증상과 의학적 배경 위주로 요약하세요.
        [현재 시나리오]
        {scenario}
        [의학 정보]
        {retrieved_knowledge}
        ### Question:
        이 환자가 지금 보여야 할 핵심 증상과 통증의 특징은 무엇인가요?
        """
        reasoned_info = self._generate_reasoning(prompt)
        return {
            "medical_info": reasoned_info,
            "retrieved_docs": retrieved_knowledge,
            "next_step": "patient"
        }
    
if __name__ == "__main__":    # Example usage
    from langchain_core.messages import HumanMessage

    node_state = {
        "messages": [HumanMessage(content="고혈압의 일반적인 치료 방법은 무엇인가요?")],
        "medical_info": "",
        "checklist": {},
        "next_step": ""
    }

    brain = MedicalBrain(db_path='data/chroma_db')
    result = brain(node_state)
    print("Retrieved Medical Information:\n", result["medical_info"])