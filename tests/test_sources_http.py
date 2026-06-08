"""Tests for the polite HTTP session (timing/backoff/robots), no network, no real sleeps."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("requests")

from turkish_corpus.sources._http import PoliteSession, RetryPolicy  # noqa: E402


class TestRetryPolicy:
    def test_backoff_doubles_and_caps(self):
        p = RetryPolicy(base_delay=1.0, max_delay=10.0)
        assert p.backoff(0) == 1.0
        assert p.backoff(1) == 2.0
        assert p.backoff(2) == 4.0
        assert p.backoff(10) == 10.0  # capped


class TestThrottle:
    def test_sleeps_to_honour_min_delay_per_host(self):
        slept: list[float] = []
        clock_values = iter([100.0, 100.3, 100.3])  # store1, wait-calc, store2
        s = PoliteSession(
            min_delay=1.0,
            respect_robots=False,
            sleep=slept.append,
            clock=lambda: next(clock_values),
        )
        s._throttle("example.com")  # first hit: no wait
        s._throttle("example.com")  # 0.3s elapsed -> sleep 0.7s
        assert slept == [pytest.approx(0.7)]

    def test_no_sleep_for_distinct_hosts_first_hit(self):
        slept: list[float] = []
        s = PoliteSession(
            min_delay=1.0, respect_robots=False, sleep=slept.append, clock=lambda: 5.0
        )
        s._throttle("a.com")
        s._throttle("b.com")
        assert slept == []


class TestRobots:
    def test_respect_robots_false_allows_all(self):
        s = PoliteSession(respect_robots=False)
        assert s.allowed("https://anywhere.gov.tr/x") is True

    def test_disallowed_url_raises_on_get(self):
        s = PoliteSession(respect_robots=True, sleep=lambda _: None)
        # Force a robots parser that disallows everything for our UA.
        rp = MagicMock()
        rp.can_fetch.return_value = False
        s._robots["x.gov.tr"] = rp
        with pytest.raises(PermissionError, match="robots"):
            s.get("https://x.gov.tr/secret")

    def test_robots_fetch_throttles_the_host(self):
        # The robots.txt GET must count against the host's min-delay budget: the immediately
        # following request to the same host should then wait for min_delay.
        slept: list[float] = []
        # clock(): robots _throttle store, get's _throttle wait-calc, get's _throttle store.
        clock_values = iter([10.0, 10.0, 10.0])
        s = PoliteSession(
            respect_robots=True,
            min_delay=1.0,
            sleep=slept.append,
            clock=lambda: next(clock_values),
        )
        fake_session = MagicMock()
        robots_resp = MagicMock()
        robots_resp.status_code = 200
        robots_resp.text = ""  # empty robots.txt → allow-all
        page_resp = MagicMock()
        page_resp.status_code = 200
        page_resp.headers = {}
        fake_session.get.side_effect = [robots_resp, page_resp]
        s._session = fake_session

        s.get("https://x.gov.tr/page")

        # The robots fetch recorded a last-request time for the host, so the page fetch waits a
        # full min_delay (no time elapsed between them in the injected clock).
        assert slept == [pytest.approx(1.0)]


class TestGetRetry:
    def _resp(self, status: int):
        r = MagicMock()
        r.status_code = status
        r.headers = {}
        return r

    def test_retries_on_429_then_succeeds(self):
        slept: list[float] = []
        s = PoliteSession(
            respect_robots=False,
            sleep=slept.append,
            clock=lambda: 0.0,
            retry=RetryPolicy(max_retries=3, base_delay=1.0),
        )
        fake_session = MagicMock()
        fake_session.get.side_effect = [self._resp(429), self._resp(200)]
        s._session = fake_session
        resp = s.get("https://api.example.org/oai")
        assert resp.status_code == 200
        assert fake_session.get.call_count == 2
        assert slept  # backed off once

    def test_honours_retry_after_header(self):
        slept: list[float] = []
        s = PoliteSession(
            respect_robots=False, sleep=slept.append, clock=lambda: 0.0,
            retry=RetryPolicy(max_retries=2, base_delay=1.0),
        )
        throttled = self._resp(503)
        throttled.headers = {"Retry-After": "5"}
        fake_session = MagicMock()
        fake_session.get.side_effect = [throttled, self._resp(200)]
        s._session = fake_session
        s.get("https://api.example.org/oai")
        assert 5.0 in slept
