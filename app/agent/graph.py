import os
import json
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

from tools.fitness_tools import estimate_tdee, macro_plan, exercise_picker, contraindication_check
from tools.rag_tools import search_papers
from tools.web_tools import web_search, corroborate_answer

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")

llm = AzureChatOpenAI(
    azure_endpoint = AOAI_ENDPOINT,
    azure_deployment = AOAI_DEPLOY_GPT4O_MINI,
    api_version = "2024-10-21",
    api_key = AOAI_API_KEY
)

members = ["workout", "nutrition", "supplement", "qa"]

class State(TypedDict, total=False):  # total=False로 변경
    messages: list
    next: str
    profile: dict
    use_web: bool


class Router(TypedDict):
    next: Literal["workout","nutrition","supplement","qa","FINISH"]

system_prompt = (
    "당신은 에이전트 팀의 관리자입니다: " + ", ".join(members) + ". "
    "사용자 요청을 보고 다음 에이전트를 하나 선택하세요. "
    "운동 계획/주간세트/대체운동→ workout, TDEE/매크로/식단→ nutrition, 보조제→ supplement, 일반 지식문답→ qa. "
    "사용자가 '고마워/그만/끝/thanks' 등으로 끝내면 FINISH."
)

def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    response = llm.with_structured_output(Router).invoke(
        [{"role":"system","content": system_prompt}] + state["messages"]
    )
    goto = response["next"]
    if goto == "FINISH":
        followup = llm.invoke(state["messages"])
        return Command(goto=END, update={"messages":[HumanMessage(content=followup.content)], "next": goto})
    return Command(goto=goto, update={"next": goto})

# Agents
workout_agent = create_react_agent(
    llm,
    tools=[exercise_picker, contraindication_check, search_papers],
    prompt="당신은 스트렝스 코치입니다. 사용자 프로필은 system 메시지로 별도 제공됩니다."
)

nutrition_agent = create_react_agent(
    llm,
    tools=[estimate_tdee, macro_plan, search_papers],
    prompt="당신은 영양 코치입니다. TDEE/매크로/식단 예시를 제시하고 필요 시 search_papers로 근거 인용."
)
supplement_agent = create_react_agent(
    llm,
    tools=[search_papers],
    prompt="당신은 보조제 코치입니다. 용량/타이밍/주의점을 설명하고 search_papers로 근거 인용."
)
qa_agent = create_react_agent(
    llm,
    tools=[search_papers, web_search, corroborate_answer],
    prompt="당신은 운동과학 Q&A를 담당합니다. RAG 인용과(선택) 웹 교차검증을 곁들여 간결히 답하세요."
)

qa_prompt = """...
규칙:
- use_web=True면 최종 답변 직전 반드시 corroborate_answer 도구를 호출해 근거 링크/스니펫을 수집하고,
  '웹 교차 검증 사용: 예/아니오, provider, 결과 N건, 소요 t ms'를 답변 마지막 줄에 요약 배지로 표시하라.
..."""

import json

def _agent_step(agent):
    def _node(state: State):
        # 프로필을 system message로 합성
        profile_msg = {
            "role": "system",
            "content": (
                "사용자 프로필:\n"
                f"- 성별: {state['profile']['sex']}\n"
                f"- 나이: {state['profile']['age']}\n"
                f"- 키: {state['profile']['height_cm']}cm\n"
                f"- 체중: {state['profile']['weight_kg']}kg\n"
                f"- 활동수준: {state['profile']['activity']}\n"
                f"- 목표: {state['profile']['goal']}\n"
                f"- 질환/부상: {', '.join(state['profile']['conditions']) or '없음'}\n"
            )
        }

        augmented_state = {
            **state,
            "messages": [profile_msg] + state["messages"]
        }

        result = agent.invoke(augmented_state)
        msg = result["messages"][-1].content
        msg += "\n\n⚠️ 본 정보는 일반적 피트니스 조언이며, 질환/약물/부상은 전문가와 상의하세요."
        return Command(update={"messages":[HumanMessage(content=msg)]}, goto=END)
    return _node


workout_node    = _agent_step(workout_agent)
nutrition_node  = _agent_step(nutrition_agent)
supplement_node = _agent_step(supplement_agent)
qa_node         = _agent_step(qa_agent)

builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("workout", workout_node)
builder.add_node("nutrition", nutrition_node)
builder.add_node("supplement", supplement_node)
builder.add_node("qa", qa_node)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", END)   # FINISH 시 종료

graph = builder.compile()
