from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings

PERSIST_DIR = "vectorstore"

def build_faiss(chunks: List[Document], persist_dir: str = PERSIST_DIR, model_size: str = "small") -> FAISS:
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not chunks:
        raise ValueError("[build_faiss] 저장할 청크가 없습니다.")
    p = Path(persist_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]
    emb = get_embeddings(model_size)
    db = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metas)
    db.save_local(str(p))
    return db

def load_faiss(persist_dir: str = PERSIST_DIR, model_size: str = "small") -> FAISS:
    p = Path(persist_dir).resolve()
    emb = get_embeddings(model_size)
    return FAISS.load_local(str(p), emb, allow_dangerous_deserialization=True)
