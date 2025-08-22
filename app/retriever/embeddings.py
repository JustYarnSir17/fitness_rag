import os
from langchain_openai import AzureOpenAIEmbeddings

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
small_model = os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
large_model = os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")

def get_embeddings(model_size: str = "small") -> AzureOpenAIEmbeddings:
    model = small_model if model_size != "large" else large_model
    if not model:
        raise ValueError(f"[embeddings] 환경변수에 {model_size} 임베딩 배포명이 없습니다.")
    return AzureOpenAIEmbeddings(
        model=model,
        api_key=AOAI_API_KEY,
        azure_endpoint=AOAI_ENDPOINT,
        api_version="2024-10-21"
    )
