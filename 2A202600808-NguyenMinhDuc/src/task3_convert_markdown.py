"""
Task 3 - Convert files in data/landing/ to Markdown.

Legal PDF/DOCX files are converted with MarkItDown when available. News JSON
files are normalized into Markdown with metadata headers for later citation.
"""

import json
import re
import unicodedata
from pathlib import Path

try:
    from markitdown import MarkItDown
except Exception:  # pragma: no cover - handled at runtime
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def safe_console(text: str) -> str:
    """Return text that can be printed in Windows cp1252 terminals."""
    return text.encode("ascii", "ignore").decode("ascii")


def slugify(text: str, fallback: str) -> str:
    """Create a stable ASCII filename stem."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:100] or fallback


def read_pdf_with_pypdf(filepath: Path) -> str:
    """Fallback PDF text extraction if MarkItDown cannot read a PDF."""
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception:
            return ""

    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def convert_with_markitdown(filepath: Path) -> str:
    """Convert supported legal documents into Markdown text."""
    if MarkItDown is not None:
        try:
            result = MarkItDown().convert(str(filepath))
            content = getattr(result, "text_content", "") or ""
            if content.strip():
                return content.strip()
        except Exception as exc:
            print(f"  MarkItDown failed, trying fallback: {safe_console(str(exc))}")

    if filepath.suffix.lower() == ".pdf":
        return read_pdf_with_pypdf(filepath).strip()

    return ""


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX files in data/landing/legal/ to markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_files = []
    if not legal_dir.exists():
        print(f"Legal directory not found: {legal_dir}")
        return converted_files

    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        print(f"Converting legal: {safe_console(filepath.name)}")
        content = convert_with_markitdown(filepath)
        if not content:
            content = (
                f"# {filepath.stem}\n\n"
                f"Source file: {filepath.name}\n\n"
                "Conversion did not extract readable text. Please review this source "
                "file manually or install a PDF/DOCX extractor before final demo."
            )

        header = (
            f"# {filepath.stem}\n\n"
            f"**Source:** {filepath.name}\n"
            f"**Type:** legal\n\n---\n\n"
        )
        output_path = output_dir / f"{slugify(filepath.stem, filepath.stem)}.md"
        output_path.write_text(header + content, encoding="utf-8")
        converted_files.append(output_path)
        print(f"  Saved: {safe_console(output_path.name)} ({len(content)} chars)")

    return converted_files


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles in data/landing/news/ to markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_files = []
    if not news_dir.exists():
        print(f"News directory not found: {news_dir}")
        return converted_files

    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue

        print(f"Converting news: {safe_console(filepath.name)}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        title = data.get("title") or filepath.stem
        url = data.get("url", "N/A")
        crawled = data.get("date_crawled", "N/A")
        content_markdown = data.get("content_markdown") or data.get("content") or ""

        markdown = (
            f"# {title}\n\n"
            f"**Source:** {url}\n"
            f"**Crawled:** {crawled}\n"
            f"**Type:** news\n\n---\n\n"
            f"{content_markdown.strip()}\n"
        )
        output_path = output_dir / f"{slugify(filepath.stem, filepath.stem)}.md"
        output_path.write_text(markdown, encoding="utf-8")
        converted_files.append(output_path)
        print(f"  Saved: {safe_console(output_path.name)} ({len(content_markdown)} chars)")

    return converted_files


def convert_all() -> list[Path]:
    """Convert all landing files into standardized Markdown files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_outputs = convert_legal_docs()

    print("\n--- News Articles ---")
    news_outputs = convert_news_articles()

    outputs = legal_outputs + news_outputs
    print(f"\nDone. Converted {len(outputs)} files into: {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    convert_all()
