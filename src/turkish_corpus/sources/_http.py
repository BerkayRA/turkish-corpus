"""A small polite HTTP client shared by the data-acquisition downloaders.

The crawler (``turkish_corpus.crawl``) gets politeness from Scrapy; the source *downloaders*
(DergiPark OAI-PMH, Resmî Gazete, TBMM, …) fetch known URLs/APIs directly, so they need a
lighter equivalent. :class:`PoliteSession` provides:

- a **per-host minimum delay** (token-bucket-style: it sleeps only as long as needed since
  that host's last request), so we never hammer a single server;
- **retry with exponential backoff** on 429 / 5xx, honouring ``Retry-After`` when present;
- an **honest, contactable User-Agent** (placeholder contact must be replaced before a real
  run — there's a guard in the downloader CLIs);
- optional **robots.txt** compliance (on by default for HTML scraping; disable for APIs like
  OAI-PMH that are designed for harvesting).

The timing/backoff logic is pure and unit-tested; ``requests`` is imported lazily (the
``sources`` extra) and a ``sleep`` hook is injectable so tests never actually wait or hit the
network.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["PoliteSession", "DEFAULT_USER_AGENT", "RetryPolicy"]

# Placeholder contact — the download CLIs refuse to run against live hosts until this is
# replaced with a real, monitored address (same policy as the crawler's USER_AGENT).
DEFAULT_USER_AGENT = (
    "TurkishCorpusBot/1.0 (research corpus; +https://example.org/bot; contact@example.org)"
)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff for transient failures (429 / 5xx)."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds; doubled each retry
    max_delay: float = 60.0
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504, 408})
    )

    def backoff(self, attempt: int) -> float:
        """Delay before retry ``attempt`` (0-based): base*2**attempt, capped at max_delay."""
        return min(self.base_delay * (2**attempt), self.max_delay)


class PoliteSession:
    """A rate-limited, retrying ``requests`` session that respects robots.txt.

    Parameters
    ----------
    user_agent:
        Sent as the ``User-Agent`` header and used for robots.txt matching.
    min_delay:
        Minimum seconds between requests to the *same host*.
    respect_robots:
        Fetch and honour each host's robots.txt before requesting (default True). Set False
        for OAI-PMH / API endpoints designed for harvesting.
    retry:
        :class:`RetryPolicy` for transient errors.
    timeout:
        ``(connect, read)`` seconds passed to ``requests``.
    sleep:
        Injectable sleep function (defaults to ``time.sleep``); tests pass a no-op recorder.
    clock:
        Injectable monotonic clock (defaults to ``time.monotonic``); tests control time.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        min_delay: float = 1.0,
        respect_robots: bool = True,
        retry: RetryPolicy | None = None,
        timeout: tuple[float, float] = (10.0, 30.0),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.user_agent = user_agent
        self.min_delay = min_delay
        self.respect_robots = respect_robots
        self.retry = retry or RetryPolicy()
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._session = None  # lazily created requests.Session

    # -- politeness helpers (pure / testable) ------------------------------------------

    def _host(self, url: str) -> str:
        return urllib.parse.urlsplit(url).netloc

    def _throttle(self, host: str) -> None:
        """Sleep just long enough to honour ``min_delay`` since this host's last request."""
        last = self._last_request.get(host)
        if last is not None:
            wait = self.min_delay - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    def allowed(self, url: str) -> bool:
        """Whether robots.txt permits fetching ``url`` for our user agent.

        Always True when ``respect_robots`` is False. A robots.txt that can't be fetched is
        treated as allow-all (standard crawler convention).
        """
        if not self.respect_robots:
            return True
        host = self._host(url)
        if host not in self._robots:
            self._robots[host] = self._load_robots(url)
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    # -- network (lazy requests) -------------------------------------------------------

    def _ensure_session(self):
        if self._session is None:
            import requests  # noqa: PLC0415  (sources extra; lazy)

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.user_agent})
        return self._session

    def _load_robots(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        split = urllib.parse.urlsplit(url)
        robots_url = f"{split.scheme}://{split.netloc}/robots.txt"
        # The robots.txt fetch is itself a request to this host — throttle it so it counts
        # against the per-host min-delay budget (host = the netloc).
        self._throttle(split.netloc)
        try:
            resp = self._ensure_session().get(robots_url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(resp.text.splitlines())
            return rp
        except Exception:
            return None  # unreachable robots.txt -> allow-all

    def get(self, url: str, **kwargs):
        """GET ``url`` politely (throttle + robots + retry). Returns the ``requests.Response``.

        Raises ``PermissionError`` if robots.txt disallows the URL, and re-raises the last
        error after exhausting retries.
        """
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        import requests  # noqa: PLC0415

        host = self._host(url)
        last_exc: Exception | None = None
        for attempt in range(self.retry.max_retries + 1):
            self._throttle(host)
            try:
                resp = self._ensure_session().get(url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self.retry.max_retries:
                    raise
                self._sleep(self.retry.backoff(attempt))
                continue
            if resp.status_code in self.retry.retry_statuses and attempt < self.retry.max_retries:
                self._sleep(self._retry_after(resp) or self.retry.backoff(attempt))
                continue
            return resp
        if last_exc is not None:  # pragma: no cover - defensive
            raise last_exc
        raise RuntimeError("unreachable")  # pragma: no cover

    def _retry_after(self, resp) -> float | None:
        """Parse a numeric ``Retry-After`` header (seconds), if present and valid."""
        value = resp.headers.get("Retry-After")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
