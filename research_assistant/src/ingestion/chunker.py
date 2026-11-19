"""
chunker.py
----------
Section-aware / token-based sliding-window text chunker with configurable
chunk size and overlap.
"""

import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """A single chunk of text with its associated metadata."""

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""


class TextChunker:
    """
    Splits a list of page-level text dicts into overlapping token-based chunks.

    Token approximation: 1 token ≈ 4 characters (GPT-style rough estimate).
    For exact token counting, substitute a tokeniser such as ``tiktoken``.

    Args:
        chunk_size:    Target chunk length in *tokens* (default 500).
        chunk_overlap: Number of overlapping tokens between consecutive chunks
                       (default 50).
    """

    CHARS_PER_TOKEN: int = 4  # rough approximation

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._char_size = chunk_size * self.CHARS_PER_TOKEN
        self._char_overlap = chunk_overlap * self.CHARS_PER_TOKEN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_pages(self, pages: list[dict]) -> list[TextChunk]:
        """
        Chunk a list of page dicts (as returned by ``PDFLoader.load``).

        Args:
            pages: List of ``{"text": str, "source": str, "page": int}`` dicts.

        Returns:
            A flat list of :class:`TextChunk` objects with metadata.
        """
        all_chunks: list[TextChunk] = []
        for page_dict in pages:
            text: str = self._normalise(page_dict["text"])
            metadata = {
                "source": page_dict.get("source", "unknown"),
                "page": page_dict.get("page", 0),
            }
            page_chunks = self._sliding_window(text, metadata, start_index=len(all_chunks))
            all_chunks.extend(page_chunks)
        return all_chunks

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[TextChunk]:
        """
        Convenience method: chunk a single raw string.

        Args:
            text:     Raw text to split.
            metadata: Optional metadata dict to attach to every chunk.

        Returns:
            A list of :class:`TextChunk` objects.
        """
        normalised = self._normalise(text)
        return self._sliding_window(normalised, metadata or {}, start_index=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Strip excessive whitespace and page-break artifacts."""
        text = text.replace("\f", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _sliding_window(
        self, text: str, metadata: dict, start_index: int
    ) -> list[TextChunk]:
        """
        Apply a sliding-window split over *characters* (token-approximated).

        Args:
            text:        Normalised text to split.
            metadata:    Metadata dict to attach to each chunk.
            start_index: Global chunk index offset for unique IDs.

        Returns:
            List of :class:`TextChunk` objects.
        """
        chunks: list[TextChunk] = []
        step = self._char_size - self._char_overlap
        pos = 0
        idx = start_index

        while pos < len(text):
            end = pos + self._char_size
            chunk_text = text[pos:end].strip()
            if chunk_text:
                # Try to end at a sentence boundary to avoid mid-sentence splits
                chunk_text = self._snap_to_sentence(chunk_text, end >= len(text))
                source = metadata.get("source", "doc")
                page = metadata.get("page", 0)
                chunk_id = f"{source}_p{page}_c{idx}"
                chunks.append(
                    TextChunk(text=chunk_text, metadata=dict(metadata), chunk_id=chunk_id)
                )
                idx += 1
            pos += step

        return chunks

    @staticmethod
    def _snap_to_sentence(text: str, is_last: bool) -> str:
        """
        If not the final chunk, trim trailing content after the last sentence-
        ending punctuation to avoid cutting mid-sentence.
        """
        if is_last:
            return text
        # Find the last sentence boundary
        match = re.search(r"[.!?]['\"]?\s", text[::-1])
        if match:
            cut = len(text) - match.start()
            return text[:cut].strip()
        return text

