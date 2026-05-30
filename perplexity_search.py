"""
perplexity_search.py — Emergent web access via Perplexity AI
=============================================================
Pure I/O layer. The brain decides WHEN to search (via SearchCortex pressure
neuron) and WHAT to search (current peak semantic activation or queued
unknown-word / pronunciation target). This module just executes the query
and returns Perplexity's synthesized answer.

Why Perplexity:
  Returns a coherent, fact-grounded answer (not raw snippets) plus citation
  URLs. Far higher ingestion quality for the semantic dictionary than
  fragmented Google snippets — the brain hears a real sentence, not noise.

Auth:
  Set PERPLEXITY_API_KEY in your .env (export PERPLEXITY_API_KEY=pplx-...).
  Without it, this backend reports status="disabled" and never fires —
  the SearchCortex still builds pressure but submit() returns False, which
  is the correct emergent behavior (no API key = the world is unreachable).

All searches run in a daemon thread so the 20Hz brain loop never blocks
on network I/O. Result delivery is via callback; the SearchCortex queues
the query, the worker resolves it asynchronously, then calls back with
the answer (which the brain ingests as auditory + semantic-dict updates).
"""

from __future__ import annotations

import os
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


PPLX_URL   = "https://api.perplexity.ai/chat/completions"
PPLX_MODEL = "sonar-pro"   # deeper online-search model — richer, browser-like answers

# Browser-like system prompt: the brain should get a real, informative answer
# the way a person reading a web page would — not a one-line stub. The girls
# ingest the full text (vocabulary into the semantic dict + auditory feedback),
# so a fuller answer means more for them to absorb. Still grounded and factual.
SYSTEM_PROMPT = (
    "You are a knowledgeable research assistant with live web access, acting "
    "as the user's window onto the internet. Answer the question directly and "
    "informatively in a few clear sentences, the way someone would after "
    "reading the top sources — include the key facts, names, and numbers that "
    "matter. If the query asks how to pronounce a word, include the IPA "
    "transcription and a simple phonetic spelling."
)


@dataclass
class SearchResult:
    query:   str
    snippet: str           # synthesized answer (truncated to 600 chars)
    source:  str           # "perplexity" | "fallback"
    ok:      bool


@dataclass
class _Request:
    speaker:  str
    query:    str
    callback: Callable[[str, SearchResult], None]


class PerplexitySearchBackend:
    """
    Thread-safe async Perplexity dispatcher. Same interface as the previous
    GoogleSearchBackend so SearchCortex doesn't care about the swap.
    """

    QUEUE_MAX        = 8
    TIMEOUT_S        = 20.0   # sonar-pro answers take longer than the terse sonar
    MAX_SNIPPET_CHAR = 1200   # browser-like: keep the fuller answer for ingestion
    MAX_TOKENS       = 600    # let sonar-pro give a real, multi-sentence answer

    def __init__(self):
        self._api_key = os.environ.get("PERPLEXITY_API_KEY", "").strip()
        self._enabled = bool(self._api_key and _HAS_REQUESTS)
        self._q: "queue.Queue[Optional[_Request]]" = queue.Queue(maxsize=self.QUEUE_MAX)
        self._worker: Optional[threading.Thread] = None
        self._running = False

    def status(self) -> str:
        if not _HAS_REQUESTS:
            return "disabled (requests not installed)"
        if not self._api_key:
            return "disabled (PERPLEXITY_API_KEY not set)"
        return f"perplexity:{PPLX_MODEL}"

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._loop, name="perplexity-worker", daemon=True,
        )
        self._worker.start()

    def stop(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def submit(self, speaker: str, query: str,
               callback: Callable[[str, SearchResult], None]) -> bool:
        if not query or not query.strip():
            return False
        if not self._enabled:
            return False
        req = _Request(speaker=speaker, query=query.strip()[:160], callback=callback)
        try:
            self._q.put_nowait(req)
            return True
        except queue.Full:
            return False

    # ── Worker ─────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                req = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if req is None:
                break
            try:
                result = self._do_search(req.query)
            except Exception as e:
                result = SearchResult(query=req.query,
                                      snippet=f"(error: {e})",
                                      source="fallback", ok=False)
            try:
                req.callback(req.speaker, result)
            except Exception:
                pass

    def _do_search(self, query: str) -> SearchResult:
        if not self._enabled:
            return SearchResult(query=query, snippet="(disabled)",
                                source="fallback", ok=False)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        body = {
            "model": PPLX_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            "max_tokens": self.MAX_TOKENS,
            "temperature": 0.2,
        }
        try:
            resp = requests.post(PPLX_URL, headers=headers, json=body,
                                 timeout=self.TIMEOUT_S)
        except Exception as e:
            return SearchResult(query=query, snippet=f"(network error: {e})",
                                source="fallback", ok=False)

        if resp.status_code != 200:
            return SearchResult(query=query,
                                snippet=f"(HTTP {resp.status_code}: {resp.text[:120]})",
                                source="fallback", ok=False)
        try:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return SearchResult(query=query, snippet="(empty response)",
                                    source="fallback", ok=False)
            content = (choices[0].get("message") or {}).get("content", "").strip()
            # Append citation URLs if present, so the TUI can show provenance.
            cits = data.get("citations") or []
            if cits:
                cite_str = " [src: " + ", ".join(cits[:4]) + "]"
                content  = (content + cite_str)[:self.MAX_SNIPPET_CHAR]
            else:
                content = content[:self.MAX_SNIPPET_CHAR]
            return SearchResult(query=query, snippet=content,
                                source="perplexity", ok=True)
        except Exception as e:
            return SearchResult(query=query, snippet=f"(parse error: {e})",
                                source="fallback", ok=False)
