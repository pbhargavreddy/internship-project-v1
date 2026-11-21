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


def current_theme() -> str | None:
    """``"light"`` / ``"dark"`` as the browser reports it (``None`` before the first report)."""
    return getattr(getattr(st.context, "theme", None), "type", None)


def load_css(filename: str = "main.css") -> None:
    """Inject a stylesheet from styles/ plus a hidden theme marker for its light/dark palettes."""
    st.html(STYLES_DIR / filename)
    theme = current_theme()
    if theme in ("light", "dark"):
        st.html(f"<span class='sp-theme-{theme}'></span>")


# Streamlit keeps the user's theme choice ("System" / "Light" / "Dark") in this
# localStorage key and applies it when the page loads; it has no Python API for it.
_SET_THEME_JS = """
<script>
  const key = `stActiveTheme-${window.location.pathname}-v2`;
  window.localStorage.setItem(key, JSON.stringify("%s"));
  window.location.reload();
</script>
"""


def theme_toggle() -> None:
    """Button that switches between light and dark mode (the default follows the system)."""
    dark = current_theme() == "dark"
    if st.button("Light" if dark else "Dark", key="theme-toggle", type="tertiary",
                 icon=":material/light_mode:" if dark else ":material/dark_mode:",
                 help="Switch the colour theme (default: your system setting)"):
        with st.spinner("Switching theme..."):
            st.html(_SET_THEME_JS % ("Light" if dark else "Dark"), unsafe_allow_javascript=True)


def spacer(css_class: str) -> None:
    """Render an empty div whose height is defined by a CSS class."""
    st.markdown(f"<div class='{css_class}'></div>", unsafe_allow_html=True)
