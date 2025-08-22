from typing import Any

def make_retriever(db: Any, method: str = "similarity", *, k: int = 6, lambda_mult: float = 0.5, score_threshold: float = 0.75):
    method = (method or "similarity").strip().lower()
    if method not in {"similarity", "mmr", "similarity_score_threshold"}:
        raise ValueError(f"[retriever] 지원하지 않는 method: {method}")
    if method == "mmr":
        kwargs = {"k": k, "lambda_mult": float(lambda_mult)}
    elif method == "similarity_score_threshold":
        kwargs = {"k": k, "score_threshold": float(score_threshold)}
    else:
        kwargs = {"k": k}
    return db.as_retriever(search_type=method, search_kwargs=kwargs)
