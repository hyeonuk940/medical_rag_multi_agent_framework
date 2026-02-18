import streamlit as st
from src.main import Controller
import time

st.set_page_config(page_title="title", layout="wide")
@st.cache_resource
def get_controller():
    return Controller(
        patient_model_name="gpt-4o-mini", 
        # medical_brain_model_name="snuh/hari-q2.5-thinking", 
        medical_brain_model_name="gpt-4o-mini",
        evaluator_model_name="gpt-4o-mini"
    )

controller = get_controller()

if "state" not in st.session_state:
    st.session_state.state = controller.get_initial_state()
if "logs" not in st.session_state:
    st.session_state.logs = [] 

with st.sidebar:
    st.header("Workflow Trace")
    st.info("에이전트 간의 데이터 이동과 추론 과정을 실시간으로 확인하세요.")
    
    if not st.session_state.logs:
        st.write("대화가 시작되면 로그가 이곳에 표시됩니다.")
    
    for i, log in enumerate(reversed(st.session_state.logs)):
        with st.expander(f"Turn {len(st.session_state.logs)-i}: {log['node']}", expanded=(i==0)):
            st.caption(f"Time: {log['time']}")
            st.write(log["content"])
            
            if log.get("metadata"):
                tab1, tab2 = st.tabs([" AI Reasoning", " RAG Docs"])
                
                with tab1:
                    st.write(log["metadata"].get("Reasoning", "분석 데이터 없음"))
                
                with tab2:
                    docs = log["metadata"].get("RAG_Documents", [])
                    if docs:
                        for idx, doc in enumerate(docs):
                            st.markdown(f"**[Source {idx+1}]**")
                            st.caption(doc)
                            st.divider()
                    else:
                        st.write("검색된 원문 데이터가 없습니다.")

st.title("title")
st.subheader(f"Scenario: {st.session_state.state.get('current_scenario', 'Unknown')}")

for msg in st.session_state.state["messages"]:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input(">>"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("에이전트들이 협업 중입니다..."):
        new_state = st.session_state.state
        new_state["messages"].append({"role": "human", "content": prompt})
        
        for chunk in controller.app.stream(st.session_state.state):
            for node_name, output in chunk.items():
                log_entry = {
                    "node": node_name,
                    "time": time.strftime("%H:%M:%S"),
                    "content": "",
                    "metadata": {}
                }
                
                if node_name == "medical_brain":
                    log_entry["content"] = "의학 정보 분석 및 지식 검색 완료"
                    
                    if "medical_info" in output:
                        log_entry["metadata"]["Reasoning"] = output["medical_info"]
                
                    if "retrieved_docs" in output:
                        log_entry["metadata"]["RAG_Documents"] = output["retrieved_docs"]
                
                elif node_name == "patient_agent":
                    log_entry["content"] = " 환자 에이전트 답변 생성"
                
                st.session_state.logs.append(log_entry)
                new_state.update(output)
        
        st.session_state.state = new_state

    with st.chat_message("assistant"):
        st.markdown(new_state['messages'][-1].content)

    st.rerun()