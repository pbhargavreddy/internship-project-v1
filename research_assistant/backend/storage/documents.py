"""
documents.py
------------
Per-chat documents. A PDF is extracted, chunked and embedded once, then written
to MongoDB (``documents`` + ``chunks`` with embeddings, see ``doc_store.py``) and
to the local ChromaDB; the PDF file itself is deleted. ChromaDB is a cache that
:func:`rebuild_from_mongo` restores from MongoDB without re-embedding. When
MongoDB is offline, ChromaDB alone is used and :func:`sync_chroma_to_mongo`
uploads the missing documents once it is back.

  • :func:`add_document`             → index a PDF for a chat (skipped if the chat has it)
  • :func:`remove_document`          → drop it from one chat (Mongo + Chroma + names.json)
  • :func:`remove_chat_documents`    → everything of a deleted chat
  • :func:`startup`                  → migrations + rebuild, returns a summary

Run the startup migration by hand (prints the summary):
    python -m backend.storage.documents
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gridfs
from pydantic import BaseModel, Field
from pymongo.database import Database

from backend.ingestion import Chunk, chunk_blocks, load_pdf
from backend.ingestion.names import (
    _pdf_title,
    attach_doc,
    detach_doc,
    doc_chats,
    doc_name,
    legacy_doc_ids,
    load_doc_names,
)
from backend.retrieval.vector_store import VectorStore
from backend.storage import doc_store
from backend.storage.doc_store import DocumentRecord
from backend.storage.mongo import get_db

log = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
CHUNKS_DIR = Path("data/chunks")  # legacy JSONL chunk files (no longer written)
PAPERS_DIR = Path("data/pdfs")  # paper_search downloads (deleted after ingest)
LEGACY_CHAT_NAME = "My papers"
GRIDFS_BUCKET = "pdfs"


class AddOutcome(BaseModel):
    doc_id: str
    chat_id: str
    status: str  # indexed | skipped (already in this chat)
    chunk_count: int
    in_mongo: bool


class DocRemoval(BaseModel):
    doc_id: str
    chat_id: str
    found: bool
    chunks_removed: int  # from ChromaDB
    mongo_chunks_removed: int


class MigrationResult(BaseModel):
    chat_id: str
    doc_ids: list[str]


class StartupSummary(BaseModel):
    """What :func:`startup` did; logged at backend startup."""

    mongo_online: bool
    legacy_assigned: int = 0
    synced_to_mongo: int = 0
    gridfs_reingested: int = 0
    gridfs_dropped: bool = False
    gridfs_kept_reason: str | None = None
    local_pdfs_deleted: int = 0
    rebuilt_in_chroma: int = 0
    errors: list[str] = Field(default_factory=list)

    def text(self) -> str:
        if not self.mongo_online:
            return "MongoDB offline: using the local ChromaDB as is."
        gridfs = ("GridFS 'pdfs' bucket dropped" if self.gridfs_dropped
                  else f"GridFS kept ({self.gridfs_kept_reason})" if self.gridfs_kept_reason else "no GridFS bucket")
        return (f"Documents startup: {self.legacy_assigned} assigned to '{LEGACY_CHAT_NAME}', "
                f"{self.synced_to_mongo} copied Chroma→Mongo, {self.gridfs_reingested} re-ingested from GridFS, "
                f"{gridfs}, {self.local_pdfs_deleted} local PDFs deleted, {self.rebuilt_in_chroma} rebuilt in Chroma"
                + (f", {len(self.errors)} error(s)" if self.errors else "") + ".")


def extract_chunks(path: str | Path, doc_id: str, max_pages: int | None = None) -> list[Chunk]:
    """Load and chunk a PDF (no files written)."""
    return chunk_blocks(load_pdf(path, max_pages=max_pages), doc_id=doc_id)


def add_document(
    path: str | Path, doc_id: str, filename: str, chat_id: str, *,
    store: VectorStore, title: str | None = None, max_pages: int | None = None,
    db: Database | None = None, delete_source: bool = True,
) -> AddOutcome:
    """Extract, chunk and embed a PDF for ``chat_id``; write it to MongoDB and ChromaDB.

    Args:
        path: The PDF (a temporary upload or a paper download).
        doc_id: Its content-hash ID.
        filename: Name shown in this chat (upload filename or paper title).
        chat_id: Chat that gets the document.
        store: Vector store.
        title: Document title (defaults to the PDF's metadata title, else ``filename``).
        max_pages: Only ingest the first N pages (paper_search ingests).
        db: MongoDB database. Defaults to the live connection; ``None`` offline → Chroma only.
        delete_source: Delete the PDF file afterwards (PDFs are not kept).

    Returns:
        ``status="skipped"`` if the chat already had this document.

    Raises:
        ValueError: If the PDF is unreadable or has no text.
    """
    db = db if db is not None else get_db()
    path = Path(path)
    try:
        if store.has_doc(doc_id, chat_id):
            attach_doc(doc_id, doc_name(doc_id, chat_id=chat_id) or filename, chat_id)
            return AddOutcome(doc_id=doc_id, chat_id=chat_id, status="skipped",
                              chunk_count=store.doc_chunk_count(doc_id, chat_id),
                              in_mongo=db is not None and doc_store.has_document(db, doc_id, chat_id))
        title = title or _pdf_title(path) or filename
        chunks = extract_chunks(path, doc_id, max_pages)
        if not chunks:
            raise ValueError(f"No text could be extracted from {filename}")
        embeddings = store.embed_chunks(chunks)
        in_mongo = False
        if db is not None:
            try:
                record = DocumentRecord(doc_id=doc_id, chat_id=chat_id, filename=filename, title=title,
                                        n_chunks=len(chunks))
                doc_store.save_document(db, record, chunks, embeddings)
                in_mongo = True
            except Exception as exc:  # noqa: BLE001 — Chroma still gets it; synced to Mongo on next start
                log.warning("MongoDB write of %s failed (%s); kept in ChromaDB only", doc_id, type(exc).__name__)
        store.add_embedded(chunks, embeddings, chat_id)
        attach_doc(doc_id, filename, chat_id)
        return AddOutcome(doc_id=doc_id, chat_id=chat_id, status="indexed", chunk_count=len(chunks), in_mongo=in_mongo)
    finally:
        if delete_source:
            path.unlink(missing_ok=True)


def _cleanup_local(doc_id: str, uploads_dir: Path, chunks_dir: Path) -> None:
    """Leftover files from before PDFs stopped being stored."""
    for leftover in (uploads_dir / f"{doc_id}.pdf", chunks_dir / f"{doc_id}.jsonl"):
        leftover.unlink(missing_ok=True)


def remove_document(
    doc_id: str, chat_id: str, *, store: VectorStore, db: Database | None = None,
    uploads_dir: str | Path = UPLOAD_DIR, chunks_dir: str | Path = CHUNKS_DIR,
) -> DocRemoval:
    """Remove ``doc_id`` from ``chat_id``: MongoDB, ChromaDB and names.json.

    Other chats holding the same file keep their copies. Earlier answers keep their citation text.
    """
    db = db if db is not None else get_db()
    registered = doc_id in load_doc_names(chat_id=chat_id)
    removed = store.delete_doc(doc_id, chat_id)
    mongo_removed = 0
    in_mongo = False
    if db is not None:
        in_mongo = doc_store.has_document(db, doc_id, chat_id)
        mongo_removed = doc_store.delete_document(db, doc_id, chat_id)
    in_chat = registered or in_mongo
    remaining = detach_doc(doc_id, chat_id) if registered else doc_chats(doc_id)
    if not remaining and not store.has_doc(doc_id):
        _cleanup_local(doc_id, Path(uploads_dir), Path(chunks_dir))
    return DocRemoval(doc_id=doc_id, chat_id=chat_id, found=in_chat or removed > 0 or mongo_removed > 0,
                      chunks_removed=removed, mongo_chunks_removed=mongo_removed)


def chat_doc_ids(chat_id: str, store: VectorStore, db: Database | None = None) -> list[str]:
    """Documents of a chat: registered ones, ones in ChromaDB and (if online) ones in MongoDB."""
    db = db if db is not None else get_db()
    mongo = [r.doc_id for r in doc_store.list_documents(db, chat_id)] if db is not None else []
    return list(dict.fromkeys([*load_doc_names(chat_id=chat_id), *store.chat_doc_ids(chat_id), *mongo]))


def remove_chat_documents(chat_id: str, *, store: VectorStore, db: Database | None = None, **dirs: Path) -> list[DocRemoval]:
    """Remove every document of ``chat_id`` (see :func:`remove_document`)."""
    return [remove_document(d, chat_id, store=store, db=db, **dirs) for d in chat_doc_ids(chat_id, store, db)]


# ── Migrations / rebuild ─────────────────────────────────────────────────────


def migrate_legacy_documents(*, store: VectorStore) -> MigrationResult | None:
    """One-time: assign every document without a chat (registry or ChromaDB) to "My papers".

    Their chunks are re-keyed for the chat (no re-embedding). Returns ``None`` if nothing needed it.
    """
    from backend.storage.chats import create_chat, list_chats  # chats imports the agent package

    legacy = list(dict.fromkeys([*legacy_doc_ids(), *store.legacy_doc_ids()]))
    if not legacy:
        return None
    chat = next((c for c in list_chats() if c.name == LEGACY_CHAT_NAME), None) or create_chat(LEGACY_CHAT_NAME)
    for doc_id in legacy:
        moved = store.assign_chat(doc_id, chat.chat_id)
        attach_doc(doc_id, doc_name(doc_id) or f"{doc_id}.pdf", chat.chat_id)
        log.info("Assigned %s to chat %r (%d chunks re-keyed)", doc_id, LEGACY_CHAT_NAME, moved)
    return MigrationResult(chat_id=chat.chat_id, doc_ids=legacy)


def sync_chroma_to_mongo(store: VectorStore, db: Database) -> int:
    """Copy every chat document ChromaDB has but MongoDB lacks (chunks + embeddings). Returns how many."""
    copied = 0
    for doc_id, chat_id in store.doc_chat_pairs():
        if doc_store.has_document(db, doc_id, chat_id):
            continue
        chunks, embeddings = store.export_doc(doc_id, chat_id)
        name = doc_name(doc_id, chat_id=chat_id) or f"{doc_id}.pdf"
        record = DocumentRecord(doc_id=doc_id, chat_id=chat_id, filename=name, title=doc_name(doc_id) or name,
                                n_chunks=len(chunks))
        doc_store.save_document(db, record, chunks, embeddings)
        copied += 1
    return copied


def migrate_gridfs(store: VectorStore, db: Database, summary: StartupSummary) -> None:
    """Re-ingest GridFS PDFs found nowhere else, then drop the ``pdfs`` bucket.

    The bucket is dropped only when every backed-up (doc_id, chat) is in MongoDB's
    ``documents``, so no document is lost.
    """
    files = db[f"{GRIDFS_BUCKET}.files"]
    if GRIDFS_BUCKET + ".files" not in db.list_collection_names():
        return
    fs = gridfs.GridFS(db, collection=GRIDFS_BUCKET)
    missing: list[tuple[str, str]] = []
    for f in list(files.find({})):
        meta = f.get("metadata") or {}
        doc_id = meta.get("doc_id")
        for chat_id in meta.get("chat_ids") or []:
            if not doc_id or doc_store.has_document(db, doc_id, chat_id):
                continue
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(fs.get(f["_id"]).read())
                add_document(tmp.name, doc_id, meta.get("filename") or f"{doc_id}.pdf", chat_id, store=store, db=db)
                summary.gridfs_reingested += 1
            except Exception as exc:  # noqa: BLE001
                summary.errors.append(f"GridFS {doc_id}: {type(exc).__name__}")
            if not doc_store.has_document(db, doc_id, chat_id):
                missing.append((doc_id, chat_id))
    if missing:
        summary.gridfs_kept_reason = f"{len(missing)} document(s) not yet in MongoDB"
        return
    db.drop_collection(f"{GRIDFS_BUCKET}.files")
    db.drop_collection(f"{GRIDFS_BUCKET}.chunks")
    summary.gridfs_dropped = True


def delete_local_pdfs(db: Database, uploads_dir: Path = UPLOAD_DIR, papers_dir: Path = PAPERS_DIR) -> int:
    """Delete stored PDF files whose documents are safely in MongoDB. Returns how many."""
    from backend.ingestion import compute_doc_id

    in_mongo = {r.doc_id for r in doc_store.list_documents(db)}
    deleted = 0
    for pdf in [*Path(uploads_dir).glob("*.pdf"), *Path(papers_dir).glob("*.pdf")]:
        try:
            doc_id = pdf.stem if pdf.parent == Path(uploads_dir) else compute_doc_id(pdf)
        except OSError:
            continue
        if doc_id in in_mongo:
            pdf.unlink(missing_ok=True)
            (CHUNKS_DIR / f"{doc_id}.jsonl").unlink(missing_ok=True)
            deleted += 1
    return deleted


def rebuild_from_mongo(store: VectorStore, db: Database) -> int:
    """Restore chat documents missing from ChromaDB using MongoDB's stored embeddings.

    Never calls the embedder. Also re-registers names that names.json lacks.

    Returns:
        Number of (doc_id, chat) copies rebuilt.
    """
    rebuilt = 0
    for record in doc_store.list_documents(db):
        if record.doc_id not in load_doc_names(chat_id=record.chat_id):
            attach_doc(record.doc_id, record.filename, record.chat_id)
        if store.has_doc(record.doc_id, record.chat_id):
            continue
        stored = doc_store.load_chunks(db, record.doc_id, record.chat_id)
        if stored:
            store.add_embedded([c.to_chunk() for c in stored], [c.embedding for c in stored], record.chat_id)
            rebuilt += 1
    return rebuilt


def startup(store: VectorStore, db: Database | None = None) -> StartupSummary:
    """Backend startup: per-chat migration, Chroma→Mongo sync, GridFS migration, rebuild.

    Each step is idempotent; with MongoDB offline only the per-chat migration runs.
    """
    db = db if db is not None else get_db()
    summary = StartupSummary(mongo_online=db is not None)
    try:
        migrated = migrate_legacy_documents(store=store)
        summary.legacy_assigned = len(migrated.doc_ids) if migrated else 0
    except Exception as exc:  # noqa: BLE001
        summary.errors.append(f"legacy migration: {type(exc).__name__}")
    if db is None:
        return summary
    for step in ("sync", "gridfs", "cleanup", "rebuild"):
        try:
            if step == "sync":
                summary.synced_to_mongo = sync_chroma_to_mongo(store, db)
            elif step == "gridfs":
                migrate_gridfs(store, db, summary)
            elif step == "cleanup":
                summary.local_pdfs_deleted = delete_local_pdfs(db)
            else:
                summary.rebuilt_in_chroma = rebuild_from_mongo(store, db)
        except Exception as exc:  # noqa: BLE001 — one failed step must not stop the others
            log.exception("Documents startup step %s failed", step)
            summary.errors.append(f"{step}: {type(exc).__name__}")
    return summary


if __name__ == "__main__":
    from backend.retrieval import get_vector_store

    logging.basicConfig(level=logging.INFO)
    print(startup(get_vector_store()).text())
