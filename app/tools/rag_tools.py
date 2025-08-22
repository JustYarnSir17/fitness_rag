# app/tools/rag_tools.py
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain_core.tools import tool
from loader.loader import list_supported_files, load_and_split_one
from retriever.index import load_faiss, build_faiss

"""
단일 '코퍼스' 벡터스토어를 사용하고, 질의 시 메타데이터로 파일 필터링합니다.
- 인덱스 위치: RAG_INDEX_PATH (기본: app/vectorstore/corpus__small)
- 리소스 위치: RAG_RESOURCES_DIR (기본: app/resources)
- 스코프: set_scope(mode="corpus" | "file", file_path=...)
"""

DEFAULT_INDEX = os.getenv("RAG_INDEX_PATH", "app/vectorstore/corpus__small")
RES_DIR = Path(os.getenv("RAG_RESOURCES_DIR", "app/resources")).resolve()

# 내부 상태
_DB = None                   # FAISS 인스턴스 (로드 후 캐시)
_SCOPE: Dict[str, Any] = {   # {"mode":"corpus"} or {"mode":"file", "file":"..."}
    "mode": "corpus"
}

def _index_exists(vs_dir: str | Path) -> bool:
    p = Path(vs_dir)
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()

def _ensure_corpus_index() -> None:
    """코퍼스 인덱스가 없으면 전체 resources를 스캔해 생성."""
    if _index_exists(DEFAULT_INDEX):
        return
    files = list_supported_files(RES_DIR)
    if not files:
        raise RuntimeError(f"[RAG] resources 비어 있음: {RES_DIR}")
    all_docs = []
    for fp in files:
        # 문서 로드 + 청크 (메타데이터에 'source'가 들어감)
        chunks = load_and_split_one(fp)
        # 필요하면 여기서 추가 메타데이터 가공 가능
        all_docs.extend(chunks)

    Path(DEFAULT_INDEX).parent.mkdir(parents=True, exist_ok=True)
    model_size = "small" if "small" in str(DEFAULT_INDEX) else "large"
    build_faiss(all_docs, persist_dir=str(DEFAULT_INDEX), model_size=model_size)

def _load_db():
    global _DB
    if _DB is None:
        model_size = "small" if "small" in str(DEFAULT_INDEX) else "large"
        _DB = load_faiss(persist_dir=str(DEFAULT_INDEX), model_size=model_size)
    return _DB

def _filter_for_scope() -> Optional[Dict[str, Any]]:
    """현재 스코프에 맞는 메타데이터 필터 생성. FAISS는 dict의 '정확 매칭' 필터를 지원."""
    if _SCOPE.get("mode") == "file" and _SCOPE.get("file"):
        # loader가 넣은 metadata["source"]는 '절대경로 문자열'임
        return {"source": str(Path(_SCOPE["file"]).resolve())}
    return None  # 전체 코퍼스

def set_scope(mode: str = "corpus", file_path: Optional[str] = None) -> None:
    """질의 스코프 설정. 'corpus' or 'file' (file일 때 file_path 필요)"""
    mode = (mode or "corpus").lower()
    if mode not in {"corpus", "file"}:
        raise ValueError("set_scope: mode must be 'corpus' or 'file'")
    if mode == "file":
        if not file_path:
            raise ValueError("set_scope(mode='file') requires file_path")
        _SCOPE["mode"] = "file"
        _SCOPE["file"] = str(Path(file_path).resolve())
    else:
        _SCOPE.clear()
        _SCOPE["mode"] = "corpus"

@tool("search_papers", return_direct=False)
def search_papers(query_json: str) -> str:
    """운동/영양/보조제 질문에 대해 코퍼스에서 상위 근거를 검색합니다.
    입력(JSON): {"query":"...", "k":6, "method":"mmr|similarity"}
    - 단일 코퍼스 인덱스를 사용합니다.
    - 'set_scope'로 스코프가 'file'이면 해당 파일(source 메타데이터)로 필터링합니다.
    반환: JSON 문자열 [{"text":..., "source":..., "page":..., "score":...}, ...]
    """
    _ensure_corpus_index()
    db = _load_db()

    args = json.loads(query_json)
    q = args.get("query", "")
    k = int(args.get("k", 6))
    method = (args.get("method") or "mmr").lower()
    flt = _filter_for_scope()

    try:
        if method == "mmr":
            # 다양성 고려 검색
            docs = db.max_marginal_relevance_search(q, k=k, fetch_k=max(k*4, 20), filter=flt)
            scored = [(d, None) for d in docs]
        else:
            # 순수 유사도
            docs_scored = db.similarity_search_with_score(q, k=k, filter=flt)
            scored = docs_scored
    except Exception:
        # 일부 백엔드/버전에서 filter가 붙은 API가 없을 경우 fallback
        docs = db.similarity_search(q, k=k)
        scored = [(d, None) for d in docs]

    out = []
    for d, score in scored:
        out.append({
            "text": d.page_content[:1200],
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "score": float(score) if score is not None else None
        })
    return json.dumps(out, ensure_ascii=False)

@tool("corpus_info", return_direct=False)
def corpus_info(_: str = "") -> str:
    """현재 코퍼스/스코프 상태를 반환합니다(디버그용)."""
    info = {
        "index": str(Path(DEFAULT_INDEX).resolve()),
        "resources": str(RES_DIR),
        "scope": dict(_SCOPE),
        "index_exists": _index_exists(DEFAULT_INDEX),
    }
    return json.dumps(info, ensure_ascii=False)
