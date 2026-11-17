"""Index the supplied 50-page Chaosium Quick-Start PDF, without modifying it."""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from pypdf import PdfReader

from .rag import EMBED_MODEL, data_dir, database_path, embed


def section_for_page(page: int) -> str | None:
    # Physical PDF page numbers, not the book's printed folios.
    if 6 <= page <= 16:
        return "rules"
    if 18 <= page <= 32:
        return "scenario"
    if page == 33:
        return "handout"
    if 35 <= page <= 44:
        return "character"
    return None


def split_text(text: str, size: int = 1400, overlap: int = 200):
    text = re.sub(r"[ \t]+", " ", text.replace("\x00", ""))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def build(pdf: Path) -> int:
    reader = PdfReader(pdf)
    if len(reader.pages) != 50:
        raise ValueError("This importer expects the supplied 50-page Quick-Start edition. Other editions need a page map adjustment.")
    # Confirm content instead of silently indexing an unrelated 50-page file.
    if "haunting" not in (reader.pages[17].extract_text() or "").lower():
        raise ValueError("Expected scenario heading not found; check PDF edition.")
    records = []
    empty_character_pages = 0
    for number, page in enumerate(reader.pages, 1):
        section = section_for_page(number)
        if not section:
            continue
        text = page.extract_text() or ""
        if len(text.strip()) < 40:
            if section == "character":
                empty_character_pages += 1
            continue
        for piece in split_text(text):
            records.append((pdf.name, number, section, piece))
    if not records:
        raise ValueError("No extractable text found")
    if empty_character_pages:
        print("Note: character sheets contain no usable text; they were not indexed. Use a separately entered character sheet.")
    fd, temporary = tempfile.mkstemp(prefix="index-", suffix=".sqlite3", dir=data_dir())
    os.close(fd)
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY,source TEXT,page INTEGER,section TEXT,text TEXT,embedding TEXT)")
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
            db.execute("INSERT INTO metadata VALUES('embedding_model',?)", (EMBED_MODEL,))
            for offset in range(0, len(records), 8):
                batch = records[offset:offset+8]
                vectors = embed([row[3] for row in batch])
                db.executemany("INSERT INTO chunks(source,page,section,text,embedding) VALUES(?,?,?,?,?)",
                               [(*row, json.dumps(vector)) for row, vector in zip(batch, vectors)])
                print(f"Indexed {min(offset+8, len(records))}/{len(records)} chunks", flush=True)
        os.replace(temporary, database_path())
    finally:
        Path(temporary).unlink(missing_ok=True)
    if empty_character_pages:
        print("Note: image-only character sheets were not indexed. The app uses its documented simplified character preset.")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error("PDF file does not exist")
    print(f"Complete: {build(args.pdf)} chunks. Original PDF was not copied or changed.")


if __name__ == "__main__":
    main()
