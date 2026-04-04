import streamlit as st
from src.main import Controller
from langchain_core.messages import HumanMessage
import time

st.set_page_config(page_title="Medical RAG Multi-Agent Framework", layout="wide")

@st.cache_resource
def get_controller():
    return Controller(
        patient_model_name="gpt-4o",
        medical_brain_model_name="gpt-4o",
        evaluator_model_name="gpt-4o-mini"
    )

controller = get_controller()

if "state" not in st.session_state:
    st.session_state.state = controller.get_initial_state()
if "logs" not in st.session_state:
    st.session_state.logs = []

with st.sidebar:

    if st.button("전체 대화 초기화", use_container_width=True):
        st.session_state.state = controller.get_initial_state()
        st.session_state.logs = []
        st.rerun()
    st.divider()
    st.header("Workflow Trace")
    st.info("각 턴에서 에이전트들이 수행한 작업과 분석 결과를 확인할 수 있습니다.")

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

st.title("Medical RAG Multi-Agent Framework")

scenario_theme = st.session_state.state.get("scenario_theme", "")
scenario_details = st.session_state.state.get("scenario_details", {})

if not scenario_theme:
    st.subheader("시나리오 테마를 선택해 주세요")
    with st.form("scenario_theme_form"):
        theme_option = st.radio(
            "시나리오 테마",
            options=["1: 죽음에 대해 알리는 상황", "2: 존엄한 선택을 돕는 상황"]
        )
        submit_theme = st.form_submit_button("다음")
        if submit_theme:
            if theme_option.startswith("1"):
                st.session_state.state["scenario_theme"] = "breaking_bad_news"
            else:
                st.session_state.state["scenario_theme"] = "dignified_choice"
            st.rerun()

elif not scenario_details:
    st.subheader("시나리오 세부 정보를 입력해 주세요")
    with st.form("scenario_details_form"):
        age = str(st.number_input("환자 나이", min_value=1, max_value=120, value=50, step=1))
        gender = st.selectbox("환자 성별", options=["남성", "여성"])
        identity_details = st.text_input("환자의 기본적인 인적 사항")
        personality = st.text_input("환자의 성격")
        condition = st.text_area("환자의 상태 및 주요 증상")
        family = st.text_input("환자의 가족 관계")
        additional_details = st.text_area("추가 세부 정보")
        submit_details = st.form_submit_button("시나리오 설정 완료")
        if submit_details:
            st.session_state.state["scenario_details"] = {
                "patient_identity": age + gender + identity_details,
                "patient_personality": personality,
                "current_condition": condition,
                "family_context": family,
                "additional_notes": additional_details
            }
            st.rerun()

else:
    theme_label = "죽음에 대해 알리는 상황" if scenario_theme == "breaking_bad_news" else "존엄한 선택을 돕는 상황"
    st.subheader(f"현재 시나리오: {theme_label}")
    st.divider()

    for msg in st.session_state.state["messages"]:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input(">>"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("에이전트들이 협업 중입니다..."):
            new_state = st.session_state.state
            new_state["messages"].append(HumanMessage(content=prompt))

            for chunk in controller.app.stream(new_state):
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