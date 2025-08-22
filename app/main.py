from pathlib import Path
import re
from loader import list_supported_files, pick_one, load_and_split_one
from retriever import build_faiss, load_faiss, make_retriever
from chain import build_rag_chain

RES_DIR = Path("./resources").resolve()
BASE_VS_DIR = Path("vectorstore")
EXIT_COMMANDS = {"", "exit", "quit", "종료", "끝", "stop", "bye", "q"}

def _slugify(name: str) -> str:
    import re as _re
    return _re.sub(r"[^\w\-]+", "_", name).strip("_").lower()

def _vs_dir_for(file_path: Path, model_size: str) -> Path:
    return BASE_VS_DIR / f"{_slugify(file_path.stem)}__{model_size}"

def _index_exists(vs_dir: Path) -> bool:
    return (vs_dir / "index.faiss").exists() and (vs_dir / "index.pkl").exists()

def run_index_for(file_path: Path, model_size: str = "small") -> Path:
    vs_dir = _vs_dir_for(file_path, model_size)
    vs_dir.mkdir(parents=True, exist_ok=True)
    docs = load_and_split_one(file_path)
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        print(f"[경고] 유효 청크 없음: {file_path.name}")
        return vs_dir
    print(f"{file_path.name} ▶ {len(docs)}개 청크 → 인덱싱 (model={model_size})")
    build_faiss(docs, persist_dir=str(vs_dir), model_size=model_size)
    print(f"✅ 인덱싱 완료: {vs_dir.resolve()}")
    return vs_dir

def run_query_for(file_path: Path, model_size: str = "small") -> None:
    vs_dir = _vs_dir_for(file_path, model_size)
    if not _index_exists(vs_dir):
        print("[알림] 인덱스 없음 → 인덱싱부터 진행")
        run_index_for(file_path, model_size=model_size)
    db = load_faiss(persist_dir=str(vs_dir), model_size=model_size)
    retriever = make_retriever(db, method="mmr", k=6, lambda_mult=0.5)
    rag_chain = build_rag_chain(retriever)
    print(f"\n질의 모드 시작 (file={file_path.name}, model={model_size})")
    print("종료하려면 Enter(빈 입력) 또는 'exit' 입력.")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[종료] 인터럽트/EOF")
            break
        if q.lower() in EXIT_COMMANDS:
            print("\n[종료] 질의 모드를 마칩니다.")
            break
        try:
            a = rag_chain(q)
            print("\n=== 답변 ===\n" + a)
        except Exception as e:
            print(f"\n[오류] {e}")

if __name__ == "__main__":
    model_size = (input("임베딩 크기 (small/large) [기본: small]: ").strip().lower() or "small")
    mode = (input("모드 선택 (index/query/auto) [기본: auto]: ").strip().lower() or "auto")
    if mode == "index":
        files = list_supported_files(RES_DIR); target = pick_one(files) if files else None
        if target: run_index_for(target, model_size=model_size)
    elif mode == "query":
        files = list_supported_files(RES_DIR); target = pick_one(files) if files else None
        if target: run_query_for(target, model_size=model_size)
    else:
        files = list_supported_files(RES_DIR)
        if not files:
            print("PDF/CSV 파일이 없습니다."); raise SystemExit(1)
        target = pick_one(files)
        run_query_for(target, model_size=model_size)
