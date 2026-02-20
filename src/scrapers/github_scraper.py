"""
GitHub Scraper for Spanish Educational Chatbot System Prompts.

Searches GitHub for Spanish-language educational chatbot system prompts using
the GitHub Search API. Supports optional authentication via a personal access
token for higher rate limits (60 req/hr unauthenticated vs 5000 req/hr authenticated).
"""

import os
import re
import json
import time
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GITHUB_API_BASE = "https://api.github.com"

# Search queries targeting Spanish educational chatbot prompts in GitHub
SEARCH_QUERIES = [
    '"eres un tutor" language:markdown',
    '"system prompt" "educación" language:markdown',
    '"instrucciones" "tutor" español',
    '"eres un asistente" "estudiante"',
]

# Regex patterns that indicate the start of a system prompt in Spanish
PROMPT_START_PATTERNS = [
    r"(Eres un[\s\S]{20,2000})",
    r"(Tu rol es[\s\S]{20,2000})",
    r"(Actúa como[\s\S]{20,2000})",
    r"(Actua como[\s\S]{20,2000})",
    r"(Eres una[\s\S]{20,2000})",
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_headers() -> dict:
    """
    Build HTTP headers for the GitHub API.

    Uses a personal access token from GITHUB_TOKEN env var if available,
    which raises the rate limit from 60 to 5000 requests per hour.
    """
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        logger.info("GitHub token found — using authenticated requests (5000 req/hr)")
    else:
        logger.warning("No GITHUB_TOKEN set — using unauthenticated requests (60 req/hr)")
    return headers


def _search_code(query: str, headers: dict, per_page: int = 30) -> list[dict]:
    """
    Execute a GitHub code search and return raw API items.

    Args:
        query:    GitHub search query string.
        headers: Auth headers for the request.
        per_page: Max results per page (GitHub caps at 100).

    Returns:
        List of raw search result items from the GitHub API.
    """
    url = f"{GITHUB_API_BASE}/search/code"
    params = {"q": query, "per_page": per_page}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        logger.info(f"Query '{query[:50]}...' → {len(items)} results")
        return items
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for query '{query}': {e}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for query '{query}': {e}")
        return []


def _fetch_file_content(url: str, headers: dict) -> Optional[str]:
    """
    Fetch and decode the raw content of a GitHub file via the Contents API.

    Args:
        url:     GitHub API URL for the file (e.g. repos/{owner}/{repo}/contents/{path}).
        headers: Auth headers for the request.

    Returns:
        Decoded file content as a string, or None on error.
    """
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # GitHub returns base64-encoded content
        encoded = data.get("content", "")
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to fetch file content from {url}: {e}")
        return None


def _fetch_repo_stars(repo_full_name: str, headers: dict) -> int:
    """
    Fetch the star count for a GitHub repository.

    Args:
        repo_full_name: Repository identifier in '{owner}/{repo}' format.
        headers:        Auth headers for the request.

    Returns:
        Star count as an integer (0 on error).
    """
    url = f"{GITHUB_API_BASE}/repos/{repo_full_name}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("stargazers_count", 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def _extract_prompt_text(content: str) -> str:
    """
    Extract the system prompt section from a markdown file's content.

    Looks for blocks that begin with common Spanish system-prompt openers
    (e.g. 'Eres un', 'Tu rol es'). Falls back to returning the full content
    if no pattern matches, so we never silently discard a result.

    Args:
        content: Full raw file content as a string.

    Returns:
        Extracted prompt text (possibly the full content if no match found).
    """
    for pattern in PROMPT_START_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            # Limit to first 2000 chars to avoid ingesting entire READMEs
            return extracted[:2000]
    # No pattern matched — return a truncated version of the raw content
    return content[:2000].strip()


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape_github(target: int = 30) -> list[dict]:
    """
    Scrape GitHub for Spanish educational chatbot system prompts.

    Runs each search query, fetches file contents, extracts prompt text,
    and deduplicates by URL. Respects rate limits with sleep intervals.

    Args:
        target: Approximate number of results to collect (default 30).

    Returns:
        List of structured prompt records in the project's JSON schema.
    """
    headers = _get_headers()
    results: list[dict] = []
    seen_urls: set[str] = set()
    prompt_counter = 1

    for query in SEARCH_QUERIES:
        if len(results) >= target:
            break

        items = _search_code(query, headers)
        # Respect secondary rate limits — GitHub penalises bursts
        time.sleep(2)

        for item in items:
            if len(results) >= target:
                break

            file_url = item.get("url", "")
            html_url = item.get("html_url", "")
            repo_info = item.get("repository", {})
            repo_name = repo_info.get("full_name", "unknown/unknown")
            repo_html = repo_info.get("html_url", "")

            # Skip duplicates (same file may surface across multiple queries)
            if file_url in seen_urls:
                continue
            seen_urls.add(file_url)

            # Fetch full file content
            raw_content = _fetch_file_content(file_url, headers)
            if not raw_content:
                continue

            # Only keep files that have at least one Spanish prompt opener
            lower_content = raw_content.lower()
            has_spanish_prompt = any(
                kw in lower_content
                for kw in ["eres un", "tu rol es", "actúa como", "actua como", "eres una"]
            )
            if not has_spanish_prompt:
                logger.debug(f"Skipping — no Spanish prompt markers in {html_url}")
                continue

            prompt_text = _extract_prompt_text(raw_content)
            stars = _fetch_repo_stars(repo_name, headers)

            record = {
                "id": f"github_{prompt_counter:03d}",
                "source": "github",
                "raw_text": prompt_text,
                "language": "es",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "repo": repo_name,
                    "url": html_url,
                    "repo_url": repo_html,
                    "stars": stars,
                    "search_query": query,
                },
            }
            results.append(record)
            prompt_counter += 1
            logger.info(f"Collected github_{prompt_counter - 1:03d} from {repo_name}")

            # Polite delay between individual file fetches
            time.sleep(1)

    logger.info(f"GitHub scraping complete. Total collected: {len(results)}")
    return results


def save_results(results: list[dict], output_path: str) -> None:
    """
    Save scraped results to a JSON file, creating parent directories if needed.

    Args:
        results:     List of prompt records to save.
        output_path: Absolute or relative path for the output JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} records to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "github_prompts.json"
    )
    prompts = scrape_github(target=30)
    save_results(prompts, os.path.normpath(output))
    print(f"\n✅ GitHub scraper done. {len(prompts)} prompts saved to {os.path.normpath(output)}")
