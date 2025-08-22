from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rapidocr_onnxruntime import RapidOCR
import fitz  # PyMuPDF

def list_supported_files(directory: Path) -> List[Path]:
    pdfs = list(directory.rglob("*.pdf"))
    csvs = list(directory.rglob("*.csv"))
    return sorted(pdfs + csvs)

def pick_one(files: List[Path]) -> Path:
    print(f"\n총 파일: {len(files)}")
    for i, p in enumerate(files, 1):
        print(f"{i:>3}. {p.name} ({p.suffix.lower().lstrip('.')})")
    while True:
        idx = input("\n하나를 선택하세요 (번호): ").strip()
        try:
            i = int(idx)
            if 1 <= i <= len(files):
                return files[i - 1]
        except ValueError:
            pass
        print("잘못된 입력입니다. 다시 입력해주세요.")

def detect_pdf_type(pdf_path: Path, text_ratio_threshold=0.1) -> str:
    with fitz.open(str(pdf_path)) as doc:
        n = len(doc)
        if n == 0:
            return "text"
        text_pages = sum(1 for page in doc if len(page.get_text("text").strip()) > 10)
        ratio = text_pages / n
        if ratio >= 1.0:
            return "text"
        elif ratio <= text_ratio_threshold:
            return "image"
        return "mixed"

def _ocr_pdf_to_documents(pdf_path: Path) -> List[Document]:
    ocr = RapidOCR()
    docs: List[Document] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            result, _ = ocr(img_bytes)
            text = "\n".join([r[1] for r in result]) if result else ""
            docs.append(Document(
                page_content=text,
                metadata={"source": str(pdf_path), "page": i, "extracted_via": "rapidocr", "dpi_hint": 144},
            ))
    return docs

def load_csv(csv_path: Path, encoding: str = "utf-8") -> List[Document]:
    loader = CSVLoader(file_path=str(csv_path), encoding=encoding)
    return loader.load()

def load_and_split_one(path: Path, chunk_size=1000, chunk_overlap=200) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        kind = detect_pdf_type(path)
        if kind == "text":
            pages = PyPDFLoader(str(path)).load()
        elif kind == "image":
            pages = _ocr_pdf_to_documents(path)
        else:
            pages = PyPDFLoader(str(path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(pages)
    elif ext == ".csv":
        rows = load_csv(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(rows)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")
