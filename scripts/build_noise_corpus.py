"""Download random Wikipedia articles as noise documents for the evaluation.

The first run picks random titles and saves them to eval/noise_titles.txt (committed); later runs
download exactly those titles, so every evaluation uses the same noise. Articles are saved as
plain text to data/noise/ (gitignored). Run from the repository root:

    uv run python scripts/build_noise_corpus.py
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TITLES_FILE = ROOT / "eval" / "noise_titles.txt"
NOISE_DIR = ROOT / "data" / "noise"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "RAG-Engine/0.1 (evaluation noise corpus; personal open-source project)"
MIN_CHARS = 1500
MIN_SOURCE_BYTES = 6000  # wiki source size; shorter pages are usually stubs


def api_get(params: dict) -> dict:
    """One API request, paced at one per second. On HTTP 429 wait as long as the server asks, once."""
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json', 'formatversion': 2})}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(1.0)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code != 429:
            raise
        time.sleep(int(error.headers.get("Retry-After", "30")))
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)


def article_text(title: str) -> str:
    """Plain-text content of one article (the API returns one full extract per request)."""
    data = api_get({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title})
    return data["query"]["pages"][0].get("extract", "")


def long_random_titles() -> list[str]:
    """20 random articles in one request; keep those whose source is long enough to be useful."""
    data = api_get({"action": "query", "generator": "random", "grnnamespace": 0, "grnlimit": 20, "prop": "info"})
    return [page["title"] for page in data["query"]["pages"] if page.get("length", 0) >= MIN_SOURCE_BYTES]


def save(title: str, text: str) -> None:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:120]
    (NOISE_DIR / f"{safe_name}.txt").write_text(f"{title}\n\n{text}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()
    NOISE_DIR.mkdir(parents=True, exist_ok=True)

    if TITLES_FILE.exists():
        titles = [t for t in TITLES_FILE.read_text(encoding="utf-8").splitlines() if t.strip()]
        for number, title in enumerate(titles, start=1):
            save(title, article_text(title))
            print(f"[{number}/{len(titles)}] {title}")
        return

    for old_file in NOISE_DIR.glob("*.txt"):  # a new random selection replaces any earlier one
        old_file.unlink()
    chosen: list[str] = []
    while len(chosen) < args.count:
        for title in long_random_titles():
            text = article_text(title)
            if len(text) >= MIN_CHARS and len(chosen) < args.count:
                save(title, text)
                chosen.append(title)
                print(f"[{len(chosen)}/{args.count}] {title} ({len(text)} chars)")
    TITLES_FILE.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"Saved {len(chosen)} titles to {TITLES_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
