# app/streamlit_app.py
import os
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import sys
APP_DIR = Path(__file__).resolve().parent  # .../app
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
# --------------------------------------------------------------------------------------------

from agent import graph
from loader import list_supported_files, load_and_split_one
from retriever import build_faiss
from tools.rag_tools import set_scope, corpus_info  
from tools.web_tools import corroborate_answer
import time

# -----------------------------------------------------------------------------
# 초기 세팅
# -----------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="AI Fitness Assistant", page_icon="🏋️", layout="wide")
st.title("🏋️ AI Fitness Assistant")

# -----------------------------------------------------------------------------
# 사이드바: 프로필 + RAG 스코프 + 업로드/인덱스 빌드
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Profile")
    sex = st.selectbox("성별", ["M", "F"], index=0)
    age = st.number_input("나이", min_value=12, max_value=99, value=28)
    height = st.number_input("키 (cm)", min_value=120, max_value=220, value=175)
    weight = st.number_input("체중 (kg)", min_value=35, max_value=200, value=72)
    activity = st.selectbox("활동수준", ["sedentary", "light", "moderate", "high"], index=2)
    goal = st.selectbox("목표", ["cut", "recomp", "bulk"], index=1)
    conditions = st.text_input("질환/부상 (쉼표로 구분)", value="")
    use_web = st.checkbox("웹 교차검증 사용", value=False)

    st.markdown("---")
    st.subheader("검색 범위 (RAG Scope)")
    scope = st.radio("RAG 스코프", ["전체 코퍼스", "선택 PDF만"], index=0)

    # 선택 PDF만일 때 파일 선택
    selected_file = None
    if scope == "선택 PDF만":
        files = list_supported_files(APP_DIR / "resources")
        names = [f.name for f in files]
        pick = st.selectbox("파일 선택", names) if names else None
        if pick:
            selected_file = next(fp for fp in files if fp.name == pick)

    st.markdown("---")
    st.subheader("리소스 업로드 & 인덱스 빌드")
    upload = st.file_uploader("PDF/CSV 업로드 (app/resources에 저장)", type=["pdf", "csv"])
    if upload:
        res_dir = APP_DIR / "resources"
        res_dir.mkdir(parents=True, exist_ok=True)
        save_path = res_dir / upload.name
        with open(save_path, "wb") as f:
            f.write(upload.getbuffer())
        st.success(f"저장됨: {save_path}")

    # 코퍼스 재빌드 버튼 
    build_clicked = st.button("코퍼스 인덱스 Build/Update (resources 전체)")

# Build 버튼 처리: resources 전체를 코퍼스 인덱스로 갱신 
if build_clicked:
    files = list_supported_files(APP_DIR / "resources")
    if not files:
        st.error("app/resources 폴더에 PDF/CSV가 없습니다.")
    else:
        all_docs = []
        for fp in files:
            try:
                all_docs.extend(load_and_split_one(fp))
            except Exception as e:
                st.warning(f"로딩 실패: {fp.name} ({e})")
        # corpus는 항상 동일 경로 사용
        corpus_vs = os.getenv("RAG_INDEX_PATH", "app/vectorstore/corpus__small")
        Path(corpus_vs).parent.mkdir(parents=True, exist_ok=True)
        model_size = "small" if "small" in str(corpus_vs) else "large"
        try:
            build_faiss(all_docs, persist_dir=corpus_vs, model_size=model_size)
            st.success(f"인덱스 빌드 완료: {corpus_vs}")
        except Exception as e:
            st.error(f"인덱스 빌드 실패: {e}")

# -----------------------------------------------------------------------------
# 채팅 영역
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

user_msg = st.chat_input("질문을 입력하세요 (예: '하체 위주 4일 루틴 + 감량 매크로')")
if user_msg:
    # 🔧 스코프 설정 (통합 코퍼스 + 메타데이터 필터링)
    if scope == "전체 코퍼스":
        set_scope(mode="corpus")
    else:
        if not selected_file:
            st.error("파일을 선택하세요.")
            st.stop()
        set_scope(mode="file", file_path=str(selected_file))

    # 히스토리 업데이트
    st.session_state.history.append({"role": "user", "content": user_msg})

    # 프로필 구성
    profile = {
        "sex": sex,
        "age": age,
        "height_cm": height,
        "weight_kg": weight,
        "activity": activity,
        "goal": goal,
        "conditions": [c.strip() for c in conditions.split(",") if c.strip()],
    }

    # 현재 코퍼스/스코프 상태 디버그 출력
    # try:
    #     info = json.loads(corpus_info("{}"))
    #     st.caption(f"Index: {info.get('index')} | Scope: {info.get('scope')}")
    # except Exception:
    #     pass

    # 그래프 실행
    state = {
        "messages": st.session_state.history,
        "profile": profile,
        "next": "",
        "use_web": use_web,
    }
    try:
        out = graph.invoke(state, config={"recursion_limit": 50})
        assistant_msg = out["messages"][-1].content
        st.session_state.history.append({"role": "assistant", "content": assistant_msg})

        # 🔎 웹 교차 검증: UI에서 use_web 켜졌을 때만 실행
        if use_web:
            payload = {"question": user_msg, "draft": assistant_msg, "max_results": 3}
            t0 = time.perf_counter()
            try:
                evidence_json = corroborate_answer(json.dumps(payload))
                elapsed_total = int((time.perf_counter() - t0) * 1000)
                data = json.loads(evidence_json)
                meta = data.get("meta", {}) or {}
                used = bool(meta.get("used"))
                provider = meta.get("provider") or "-"
                count = int(meta.get("count") or 0)
                t_api = int(meta.get("elapsed_ms") or 0)
                t_wrap = int(meta.get("wrapper_elapsed_ms") or elapsed_total)
                err = meta.get("error")

                # ✅ 디버그 배지
                status = "✅ 사용" if used and not err else ("⚠️ 실패" if err else "❌ 미사용")
                st.caption(f"**웹 교차 검증:** {status} · provider={provider} · results={count} · api={t_api}ms · total={t_wrap}ms")

                # 🔗 근거 목록
                ev_list = data.get("evidence", [])
                if ev_list:
                    with st.expander("🔎 웹 근거 보기"):
                        for i, ev in enumerate(ev_list, 1):
                            title = ev.get("title") or "(제목 없음)"
                            url = ev.get("url") or ""
                            snippet = ev.get("snippet") or ev.get("content") or ""
                            st.markdown(f"**[{i}]** [{title}]({url})")
                            if snippet:
                                st.caption(snippet[:300] + ("..." if len(snippet) > 300 else ""))
                if err:
                    st.warning(f"웹 검색 오류: {err}")
            except Exception as e:
                st.warning(f"웹 교차 검증 중 예외: {e}")

    except Exception as e:
        st.session_state.history.append({"role": "assistant", "content": f"오류가 발생했습니다: {e}"})

# 대화 렌더
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
