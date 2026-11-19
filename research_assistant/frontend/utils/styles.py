"""
utils/styles.py
---------------
Loads the shared stylesheet and provides tiny helpers for CSS-classed HTML,
so no component needs inline style strings.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

STYLES_DIR = Path(__file__).resolve().parents[1] / "styles"


def load_css(filename: str = "main.css") -> None:
    """Inject a stylesheet from styles/ (style-only HTML takes no layout space)."""
    st.html(STYLES_DIR / filename)


def spacer(css_class: str) -> None:
    """Render an empty div whose height is defined by a CSS class."""
    st.markdown(f"<div class='{css_class}'></div>", unsafe_allow_html=True)
