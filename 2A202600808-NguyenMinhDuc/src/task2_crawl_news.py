"""
Task 2 - Crawl news articles about Vietnamese artists related to drugs.

The crawler writes one JSON file per article into data/landing/news/.
Each JSON includes url, title, date_crawled, and content_markdown.
"""

import asyncio
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
USE_CRAWL4AI = False

ARTICLE_URLS = [
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://ngoisao.vnexpress.net/nam-than-lai-nga-nhikolai-dinh-bi-bat-4762594.html",
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
    "https://vietnamnet.vn/van-hoa-giai-tri/toan-canh-vu-bat-tam-giam-long-nhat-va-son-ngoc-sk0008VN.html",
]


def setup_directory() -> None:
    """Create data/landing/news/ if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str, fallback: str) -> str:
    """Create a filesystem-safe ASCII-ish slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:80] or fallback


def extract_title_and_text_with_bs4(url: str) -> tuple[str, str]:
    """
    Fallback crawler using requests + BeautifulSoup.

    This keeps Task 2 runnable even when Crawl4AI is not installed or browser
    automation is unavailable on the current machine.
    """
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    if not title:
        title = "Unknown"

    article = (
        soup.find("article")
        or soup.find(class_=re.compile(r"(article|content|detail|main)", re.I))
        or soup.body
    )
    paragraphs = article.find_all(["p", "h2", "h3"]) if article else soup.find_all("p")
    lines = [p.get_text(" ", strip=True) for p in paragraphs]
    lines = [line for line in lines if len(line) > 30]
    content = "\n\n".join(lines)

    if len(content) < 500:
        all_text = soup.get_text("\n", strip=True)
        content = "\n".join(line for line in all_text.splitlines() if len(line) > 30)

    return title, content


async def crawl_with_crawl4ai(url: str) -> dict[str, Any] | None:
    """Try Crawl4AI first because it usually extracts article markdown better."""
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception:
        return None

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

        title = "Unknown"
        metadata = getattr(result, "metadata", None) or {}
        if isinstance(metadata, dict):
            title = metadata.get("title") or title

        content = getattr(result, "markdown", "") or ""
        if len(content.strip()) < 500:
            return None

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content.strip(),
            "crawler": "crawl4ai",
        }
    except Exception:
        return None


async def crawl_article(url: str) -> dict[str, Any]:
    """
    Crawl one article and return metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,
            "content_markdown": str,
            "crawler": str
        }
    """
    if USE_CRAWL4AI:
        crawl4ai_result = await crawl_with_crawl4ai(url)
        if crawl4ai_result:
            return crawl4ai_result

    title, content = extract_title_and_text_with_bs4(url)
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content.strip(),
        "crawler": "requests_bs4",
    }


async def crawl_all() -> None:
    """Crawl all configured articles and save them as JSON files."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}_{slugify(article['title'], f'article-{i:02d}')}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {filepath.name} ({len(article['content_markdown'])} chars)")


if __name__ == "__main__":
    asyncio.run(crawl_all())
