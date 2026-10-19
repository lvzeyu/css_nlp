"""Add notebook-specific Colab badges to the built MyST website."""

from base64 import b64encode
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT / "_build" / "html"
BADGE = ROOT / "src" / "colab-badge.svg"
BADGE_DATA_URL = "data:image/svg+xml;base64," + b64encode(BADGE.read_bytes()).decode("ascii")
SCRIPT = (ROOT / "src" / "colab_toolbar.js").read_text().replace(
    "__COLAB_BADGE_DATA_URL__", BADGE_DATA_URL
)
MARKER = "data-css-nlp-colab-toolbar"
TAG = f"<script {MARKER}>\n{SCRIPT}\n</script>"


def main() -> None:
    pages = list(HTML_DIR.rglob("index.html"))
    if not pages:
        raise SystemExit(f"No built HTML pages found in {HTML_DIR}")

    updated = 0
    for page in pages:
        html = page.read_text()
        if '"kind":"Notebook"' in html and "myst-fm-edit-link" not in html:
            raise SystemExit(f"Cannot locate the toolbar edit link in {page}")
        if MARKER in html:
            continue
        if "</body>" not in html:
            raise SystemExit(f"Missing </body> in {page}")
        page.write_text(html.replace("</body>", f"{TAG}</body>", 1))
        updated += 1
    print(f"Added Colab toolbar script to {updated} HTML pages")


if __name__ == "__main__":
    main()
