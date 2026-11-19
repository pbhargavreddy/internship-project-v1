"""
pdf_loader.py
-------------
Extracts clean text and metadata from PDF files using pypdf.
"""

from pathlib import Path
from typing import Generator

import pypdf


class PDFLoader:
    """
    Loads a PDF file and yields (page_text, metadata) tuples page-by-page.

    Attributes:
        pdf_path: Absolute path to the PDF file.
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    def load(self) -> list[dict]:
        """
        Read every page of the PDF and return a list of page dicts.

        Returns:
            A list of dicts, each with:
                - ``text``   (str): Extracted text from the page.
                - ``source`` (str): Filename of the source PDF.
                - ``page``   (int): 1-based page number.
        """
        pages: list[dict] = []
        with open(self.pdf_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            for page_number, page in enumerate(reader.pages, start=1):
                raw_text: str = page.extract_text() or ""
                cleaned = self._clean(raw_text)
                if cleaned:
                    pages.append(
                        {
                            "text": cleaned,
                            "source": self.pdf_path.name,
                            "page": page_number,
                        }
                    )
        return pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Strip excessive whitespace and common PDF page-break artifacts."""
        import re

        # Collapse runs of whitespace (but keep single newlines for paragraph detection)
        text = re.sub(r"[ \t]+", " ", text)
        # Remove form-feed characters
        text = text.replace("\f", "\n")
        # Collapse 3+ consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

