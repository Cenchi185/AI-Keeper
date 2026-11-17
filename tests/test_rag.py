"""Offline retrieval and atomic-ingestion checks. No commercial PDF is required."""
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from keeper import ingest, rag


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, _):
        self.pages = [FakePage("Sample document content " * 4) for _ in range(50)]
        self.pages[17] = FakePage("THE HAUNTING\nSynthetic test fixture only.")


class RagTests(unittest.TestCase):
    def setUp(self):
        # Explicit parent keeps temporary files inside the checked-out project.
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.env = patch.dict(os.environ, {"KEEPER_DATA_DIR": self.temp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    @patch.object(rag, "embed", return_value=[[1.0, 0.0]])
    def test_search_never_crosses_section_boundary(self, _):
        with closing(sqlite3.connect(rag.database_path())) as db, db:
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
            db.execute("INSERT INTO metadata VALUES('embedding_model','bge-m3')")
            db.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY,source TEXT,page INTEGER,section TEXT,text TEXT,embedding TEXT)")
            db.executemany("INSERT INTO chunks(source,page,section,text,embedding) VALUES(?,?,?,?,?)", [
                ("fixture", 1, "rules", "rule result", "[0,1]"),
                ("fixture", 2, "scenario", "secret scenario result", "[1,0]"),
            ])
        result = rag.search("test", section="rules")
        self.assertEqual([item["text"] for item in result], ["rule result"])
        with self.assertRaises(ValueError):
            rag.search("test", section="all")

    @patch.object(ingest, "PdfReader", FakeReader)
    def test_embedding_failure_preserves_previous_index(self):
        original = b"previous index must survive"
        rag.database_path().write_bytes(original)
        with patch.object(ingest, "embed", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                ingest.build(Path("not-read.pdf"))
        self.assertEqual(rag.database_path().read_bytes(), original)
        self.assertEqual(list(Path(self.temp.name).glob("index-*.sqlite3")), [])

    @patch.object(ingest, "PdfReader", FakeReader)
    def test_successful_build_replaces_existing_file(self):
        rag.database_path().write_bytes(b"old index")
        with patch.object(ingest, "embed", side_effect=lambda texts: [[1.0, 0.0] for _ in texts]):
            count = ingest.build(Path("not-read.pdf"))
        self.assertGreater(count, 0)
        self.assertEqual(rag.document_count(), count)
        with closing(sqlite3.connect(rag.database_path())) as db:
            sections = {row[0] for row in db.execute("SELECT DISTINCT section FROM chunks")}
        self.assertEqual(sections, {"rules", "scenario", "handout", "character"})


if __name__ == "__main__":
    unittest.main()
