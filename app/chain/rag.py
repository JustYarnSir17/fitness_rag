import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")

llm = AzureChatOpenAI(
    azure_endpoint = AOAI_ENDPOINT,
    azure_deployment = AOAI_DEPLOY_GPT4O_MINI,
    api_version = "2024-10-21",
    api_key = AOAI_API_KEY
)

DEBUG_RETRIEVE = True

PROMPT = ChatPromptTemplate.from_template(
    """당신은 근거 기반 어시스턴트입니다.
다음 컨텍스트만 사용해 한국어로 간결하고 정확하게 답하세요.
컨텍스트가 불충분하면 '자료에서 답을 찾기 어렵습니다.'라고 말하세요.
가능하면 [source p.page] 형식의 간단한 인용을 포함하세요.

# 컨텍스트
{context}

# 질문
{question}
"""
)

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] meta={{'source':{d.metadata.get('source')}, 'page':{d.metadata.get('page')}}}\n{d.page_content}"
        for i, d in enumerate(docs)
    )

def build_rag_chain(retriever):
    parser = StrOutputParser()

    def chain(question: str) -> str:
        docs = retriever.invoke(question)
        if DEBUG_RETRIEVE:
            print("\n[DEBUG] Top-k retrieved chunks:")
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source"); page = d.metadata.get("page")
                print(f"  {i:>2}. source={src}, page={page}, len={len(d.page_content)}")
        ctx = _format_docs(docs)
        msg = PROMPT.format(context=ctx, question=question)
        return parser.parse(llm.invoke(msg).content)

    return chain
