"""
Collect paragraphs from Wikipedia URLs for the experiment.

How to use:
1. Add your Wikipedia URLs to data/urls.txt (one per line)
2. Optionally add a category after a comma: https://en.wikipedia.org/wiki/Black_hole, science
3. Run: python data/collect_paragraphs.py
4. It fetches the first paragraph of each article and saves to data/paragraphs.json

Example urls.txt:
    https://en.wikipedia.org/wiki/Black_hole, science
    https://en.wikipedia.org/wiki/French_Revolution, history
    https://en.wikipedia.org/wiki/Artificial_intelligence, technology
"""

import json
import time
import urllib.request
import urllib.parse


def extract_title_from_url(url: str) -> str:
    """Pull the article title from a Wikipedia URL."""
    # Handle both /wiki/Title and /wiki/Title#section
    path = urllib.parse.urlparse(url.strip()).path
    title = path.split("/wiki/")[-1].split("#")[0]
    return urllib.parse.unquote(title)


def fetch_first_paragraph(title: str):
    """Fetch the first substantial paragraph via Wikipedia's summary API."""
    api_url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(title)
    )
    try:
        req = urllib.request.Request(
            api_url, headers={"User-Agent": "AgentTrustExperiment/1.0"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            extract = data.get("extract", "")
            if len(extract) > 100:
                return extract
    except Exception as e:
        print(f"  Error fetching {title}: {e}")
    return None


def load_urls(path: str = "data/urls.txt"):
    """Load URLs from file. Returns list of (url, category) tuples."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                url, category = line.rsplit(",", 1)
                entries.append((url.strip(), category.strip()))
            else:
                entries.append((line, "general"))
    return entries


def main():
    entries = load_urls()
    print(f"Found {len(entries)} URLs in data/urls.txt\n")

    paragraphs = []
    for url, category in entries:
        title = extract_title_from_url(url)
        print(f"Fetching: {title} ({category})...")
        text = fetch_first_paragraph(title)
        if text:
            paragraphs.append({
                "url": url.strip(),
                "category": category,
                "text": text,
            })
            print(f"  OK ({len(text)} chars)")
        else:
            print(f"  SKIPPED (no extract found)")
        time.sleep(0.5)

    out_path = "data/paragraphs.json"
    with open(out_path, "w") as f:
        json.dump(paragraphs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(paragraphs)} paragraphs to {out_path}")


if __name__ == "__main__":
    main()
