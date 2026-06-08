"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://tuoitre.vn/ca-si-chi-dan-nguoi-mau-an-tay-co-tien-truc-phuong-to-chuc-su-dung-ma-tuy-ra-sao-2026040214370414.htm",
    "https://tuoitre.vn/chuyen-an-vn10-truy-to-227-bi-can-trong-do-co-ca-si-chi-dan-an-tay-2026040308051239.htm",
    "https://tuoitre.vn/dien-vien-hai-huu-tin-bi-khoi-to-bat-tam-giam-vi-ma-tuy-20220617185327576.htm",
    "https://tuoitre.vn/dien-vien-huu-tin-bi-truy-to-vi-to-chuc-su-dung-ma-tuy-20221117104908287.htm",
    "https://thanhnien.vn/khoi-to-bat-khan-cap-ca-si-chau-viet-cuong-185738463.htm"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        title = "Unknown"
        if result and hasattr(result, 'metadata') and result.metadata:
            title = result.metadata.get("title", "Unknown")
        
        if not title or title == "Unknown":
            if result and hasattr(result, 'html') and result.html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(result.html, 'html.parser')
                title = soup.title.string.strip() if soup.title else "Unknown"

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown if result and hasattr(result, 'markdown') else "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  v Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
