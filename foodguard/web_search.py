"""Provider-neutral web search and safe page fetching for FoodGuard.

The MCP server is the only caller exposed to the application.  This module
keeps provider details out of the router so a deployment can switch between a
no-key development fallback and hosted providers without changing QA logic.
"""

from __future__ import annotations

import html
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests


DEFAULT_TIMEOUT = 12.0
DEFAULT_USER_AGENT = "FoodGuard/1.0 (+https://github.com/Xhgh0405/FoodGuard)"


class WebSearchError(RuntimeError):
    """Raised when a configured provider cannot return a usable result."""


def _setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _enabled() -> bool:
    value = _setting("WEB_SEARCH_ENABLED", "")
    if value:
        return value.casefold() in {"1", "true", "yes", "on"}
    # Existing offline deployments remain safe unless they explicitly opt in.
    return _setting("FOODGUARD_OFFLINE", "0").casefold() not in {"1", "true", "yes", "on"}


def _timeout() -> float:
    try:
        return max(2.0, float(_setting("WEB_SEARCH_TIMEOUT", str(DEFAULT_TIMEOUT))))
    except ValueError:
        return DEFAULT_TIMEOUT


def _domain_name(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _domain_allowed(url: str, domains: Iterable[str] | None) -> bool:
    allowed = [str(item).lower().removeprefix("www.").lstrip(".") for item in (domains or [])]
    if not allowed:
        return True
    host = _domain_name(url)
    return any(host == item or host.endswith("." + item) for item in allowed)


def _publisher(url: str, fallback: str = "") -> str:
    return fallback or _domain_name(url)


def _record(
    *, title: str, url: str, snippet: str = "", publisher: str = "", published_date: str = "", rank: int = 1
) -> dict[str, Any]:
    return {
        "title": " ".join(str(title or "").split()),
        "url": str(url or "").strip(),
        "snippet": " ".join(str(snippet or "").split()),
        "publisher": _publisher(str(url or ""), publisher),
        "published_date": str(published_date or "").strip(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
    }


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            if self._current is not None and self._current.get("url") and self._current.get("title"):
                self.results.append(self._current)
            href = attributes.get("href") or ""
            self._current = {"url": href, "title": "", "snippet": ""}
            self._capture = "title"
        elif self._current is not None and tag in {"a", "div", "td"} and (
            "result__snippet" in classes or "result__snippet" in (attributes.get("class") or "")
        ):
            self._capture = "snippet"

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture:
            self._current[self._capture] += data

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag == "a" and self._capture == "title":
            self._capture = None
        elif tag == "a" and self._capture == "snippet":
            self._capture = None
            if self._current.get("url") and self._current.get("title"):
                self.results.append(self._current)
                self._current = None

    def close(self) -> None:
        super().close()
        if self._current is not None and self._current.get("url") and self._current.get("title"):
            self.results.append(self._current)
            self._current = None
        if tag in {"div", "td"} and self._capture == "snippet":
            self._capture = None
            if self._current.get("url") and self._current.get("title"):
                self.results.append(self._current)
                self._current = None


def _unwrap_ddg_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _search_duckduckgo(query: str, domains: list[str], limit: int) -> list[dict[str, Any]]:
    search_query = query
    if domains:
        search_query += " " + " ".join(f"site:{domain}" for domain in domains)
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": search_query},
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=_timeout(),
    )
    response.raise_for_status()
    parser = _DuckDuckGoParser()
    parser.feed(response.text)
    results: list[dict[str, Any]] = []
    for item in parser.results:
        url = _unwrap_ddg_url(item["url"])
        if not url.startswith(("http://", "https://")) or not _domain_allowed(url, domains):
            continue
        results.append(_record(title=item["title"], url=url, snippet=item.get("snippet", ""), rank=len(results) + 1))
        if len(results) >= limit:
            break
    return results


def _search_brave(query: str, domains: list[str], limit: int) -> list[dict[str, Any]]:
    key = _setting("WEB_SEARCH_API_KEY")
    if not key:
        raise WebSearchError("WEB_SEARCH_API_KEY is required for the Brave provider")
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": limit, **({"domains": domains} if domains else {})},
        headers={"Accept": "application/json", "X-Subscription-Token": key},
        timeout=_timeout(),
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("web", {}).get("results", []):
        url = str(item.get("url", ""))
        if not _domain_allowed(url, domains):
            continue
        results.append(_record(title=item.get("title", ""), url=url, snippet=item.get("description", ""), rank=len(results) + 1))
    return results[:limit]


def _search_tavily(query: str, domains: list[str], limit: int) -> list[dict[str, Any]]:
    key = _setting("WEB_SEARCH_API_KEY")
    if not key:
        raise WebSearchError("WEB_SEARCH_API_KEY is required for the Tavily provider")
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": limit, **({"include_domains": domains} if domains else {})},
        timeout=_timeout(),
    )
    response.raise_for_status()
    results = []
    for item in response.json().get("results", []):
        url = str(item.get("url", ""))
        if _domain_allowed(url, domains):
            results.append(_record(title=item.get("title", ""), url=url, snippet=item.get("content", ""), published_date=item.get("published_date", ""), rank=len(results) + 1))
    return results[:limit]


def _search_serper(query: str, domains: list[str], limit: int) -> list[dict[str, Any]]:
    key = _setting("WEB_SEARCH_API_KEY")
    if not key:
        raise WebSearchError("WEB_SEARCH_API_KEY is required for the Serper provider")
    scoped_query = query + (" " + " ".join(f"site:{domain}" for domain in domains) if domains else "")
    response = requests.post(
        "https://google.serper.dev/search",
        json={"q": scoped_query, "num": limit},
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        timeout=_timeout(),
    )
    response.raise_for_status()
    results = []
    for item in response.json().get("organic", []):
        url = str(item.get("link", ""))
        if _domain_allowed(url, domains):
            results.append(_record(title=item.get("title", ""), url=url, snippet=item.get("snippet", ""), rank=len(results) + 1))
    return results[:limit]


def search_web(
    query: str,
    domains: list[str] | None = None,
    recency_days: int | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Return a stable, cleaned search-result schema without exposing HTML."""

    query = " ".join(str(query or "").split()).strip()
    domains = [str(item).strip() for item in (domains or []) if str(item).strip()]
    limit = min(max(int(max_results or 5), 1), 10)
    base: dict[str, Any] = {
        "query": query,
        "results": [],
        "provider": _setting("WEB_SEARCH_PROVIDER", "duckduckgo"),
        "enabled": _enabled(),
        "recency_days": recency_days,
    }
    if not query:
        base.update({"status": "invalid_query", "message": "query must not be empty"})
        return base
    if not base["enabled"]:
        base.update({"status": "disabled", "message": "Web Search is disabled by WEB_SEARCH_ENABLED/FOODGUARD_OFFLINE."})
        return base
    provider = str(base["provider"]).casefold()
    try:
        if provider in {"duckduckgo", "ddg", "local"}:
            results = _search_duckduckgo(query, domains, limit)
        elif provider == "brave":
            results = _search_brave(query, domains, limit)
        elif provider == "tavily":
            results = _search_tavily(query, domains, limit)
        elif provider in {"serper", "google"}:
            results = _search_serper(query, domains, limit)
        else:
            raise WebSearchError(f"Unsupported WEB_SEARCH_PROVIDER: {provider}")
        base.update({"status": "ok" if results else "no_results", "results": results})
    except (requests.RequestException, ValueError, WebSearchError) as exc:
        base.update({"status": "unavailable", "message": f"Web Search unavailable: {exc}"})
    return base


class _PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "nav", "footer", "header", "aside", "form"}:
            self._skip += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "nav", "footer", "header", "aside", "form"} and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if not value or self._skip:
            return
        if self._in_title:
            self.title.append(value)
        else:
            self.parts.append(value)


def fetch_web_page(url: str, max_chars: int = 12000) -> dict[str, Any]:
    """Fetch readable HTTP(S) text only; never downloads or executes files."""

    url = str(url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"status": "invalid_url", "url": url, "text": ""}
    try:
        response = requests.get(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=_timeout(),
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return {"status": "unsupported_content", "url": url, "text": ""}
        parser = _PageTextParser()
        parser.feed(response.text)
        text = " ".join(parser.parts)
        return {
            "status": "ok",
            "url": response.url,
            "title": " ".join(parser.title),
            "text": text[: max(500, min(int(max_chars or 12000), 30000))],
        }
    except (requests.RequestException, ValueError) as exc:
        return {"status": "unavailable", "url": url, "text": "", "message": str(exc)}
