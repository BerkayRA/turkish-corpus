"""Tests for the PURE CC seed-CSV parsing and Scrapy-input builders."""

from turkish_corpus.crawl.seeds import (
    hosts_to_allowed_domains,
    hosts_to_robots_urls,
    load_hosts,
)

CSV = (
    "url_host_name,pages\n"
    "www.example.com.tr,5000\n"
    "haber.example.org,1200\n"
    "tiny.example.net,3\n"
    ",999\n"  # blank host — must be skipped
)


class TestLoadHosts:
    def _write(self, tmp_path, text=CSV):
        path = tmp_path / "tr_hosts.csv"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_skips_header_and_blank_hosts(self, tmp_path):
        hosts = load_hosts(self._write(tmp_path))
        assert hosts == ["www.example.com.tr", "haber.example.org", "tiny.example.net"]

    def test_preserves_rank_order(self, tmp_path):
        hosts = load_hosts(self._write(tmp_path))
        assert hosts[0] == "www.example.com.tr"  # most pages first, as the query sorted

    def test_min_pages_filter(self, tmp_path):
        hosts = load_hosts(self._write(tmp_path), min_pages=1000)
        assert hosts == ["www.example.com.tr", "haber.example.org"]

    def test_nonnumeric_pages_treated_as_zero(self, tmp_path):
        csv = "url_host_name,pages\ngood.example.com,n/a\n"
        hosts = load_hosts(self._write(tmp_path, csv), min_pages=1)
        assert hosts == []


class TestHostsToAllowedDomains:
    def test_strips_www_and_dedupes_preserving_order(self):
        hosts = ["www.example.com", "example.com", "haber.example.org"]
        assert hosts_to_allowed_domains(hosts) == ["example.com", "haber.example.org"]

    def test_lowercases(self):
        assert hosts_to_allowed_domains(["WWW.Example.COM"]) == ["example.com"]


class TestHostsToRobotsUrls:
    def test_builds_https_robots_urls(self):
        urls = hosts_to_robots_urls(["example.com", "haber.example.org"])
        assert urls == [
            "https://example.com/robots.txt",
            "https://haber.example.org/robots.txt",
        ]

    def test_dedupes_preserving_order(self):
        urls = hosts_to_robots_urls(["example.com", "example.com"])
        assert urls == ["https://example.com/robots.txt"]
