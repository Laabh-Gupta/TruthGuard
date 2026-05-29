# ingestion/pdf_loader.py

import fitz  # pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> list[dict]:
    """
    Load a PDF and return list of pages with metadata.
    Each item: {"text": "...", "page": 1, "source": "filename.pdf"}
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip empty pages
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": Path(file_path).name
            })

    doc.close()
    print(f"✅ Loaded {len(pages)} pages from {Path(file_path).name}")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split pages into smaller chunks for better retrieval.
    Each chunk keeps metadata: source filename + page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])

        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "page": page["page"],
                "source": page["source"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{i}"
            })

    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks