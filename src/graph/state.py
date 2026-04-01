from typing import Annotated, Sequence, TypedDict, List, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ScenarioDetails(TypedDict):
    patient_identity: str  # 환자의 나이, 성별, 직업 등 기본적인 인적 사항
    personality: str     # 환자의 성격적 특성 (예: 낙천적, 걱정이 많은, 독립적인 등)
    current_condition: str # 환자의 현재 건강 상태 및 주요 증상
    family_context: str # 환자의 가족 상황 (예: 가족 구성원, 가족과의 관계, 가족의 건강 상태 등)
    additional_notes: str # 시나리오에 추가적으로 필요한 세부 사항이나 배경 정보

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    scenario_theme: Literal[
        "breaking_bad_news",  # 환자에게 죽음에 대해 알리는 상황
        "dignified_choice",  # 존엄한 선택을 돕는 상황
        "final_moments"  # 환자의 임종 직전 환자의 가족과의 대화 상황
    ]

    scenario_details: ScenarioDetails

    medical_info: str  # 검색된 의료 정보를 현재 시나리오와 대화 내용에 맞게 정제 및 분석한 결과
    retrieved_docs: List[str]  # 검색된 의료 정보의 원문 또는 요약된 형태로 저장된 리스트 
    next_step: str
    checklist: dict