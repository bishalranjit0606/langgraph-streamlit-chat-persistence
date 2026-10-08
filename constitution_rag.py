import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(
    APP_DIR,
    "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf",
)
INDEX_DIR = os.path.join(APP_DIR, "faiss_constitution")

# MiniLM embeds about 256 tokens. Keep chunks under that limit.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 6

_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _clean_page(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "www.lawcommission.gov.np":
            continue
        if stripped.isdigit():
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _load_chunks():
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(
            "Put the Constitution of Nepal PDF next to constitution_rag.py."
        )

    pages = PyPDFLoader(PDF_PATH).load()
    for page in pages:
        page.page_content = _clean_page(page.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    return [chunk for chunk in chunks if len(chunk.page_content) > 40]


def get_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()
    index_file = os.path.join(INDEX_DIR, "index.faiss")
    if os.path.isfile(index_file):
        _vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _vectorstore

    chunks = _load_chunks()
    if not chunks:
        raise RuntimeError("No text found in the constitution PDF.")

    _vectorstore = FAISS.from_documents(chunks, embeddings)
    _vectorstore.save_local(INDEX_DIR)
    return _vectorstore


def retrieve(question, k=TOP_K):
    return get_vectorstore().similarity_search(question, k=k)


def format_passages(docs):
    parts = []
    for number, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        where = f"page {page + 1}" if isinstance(page, int) else "constitution"
        parts.append(f"[{number}] ({where})\n{doc.page_content}")
    return "\n\n".join(parts)
