from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    medical_info: str
    retrieved_docs: List[str]
    checklist: dict
    next_step: str
    current_scenario: str