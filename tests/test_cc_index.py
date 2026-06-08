"""Tests for the PURE Common Crawl index query/path builders.

No DuckDB, no network — these assert the exact SQL/glob we send to the public CC dataset,
so a refactor cannot silently change the filters or partition path.
"""

import pytest

from turkish_corpus.crawl.cc_index import (
    DEFAULT_CRAWL_ID,
    build_index_path,
    build_tr_host_query,
)

CRAWL_ID = "CC-MAIN-2025-51"


class TestBuildIndexPath:
    def test_interpolates_crawl_id_and_globs_all_parts(self):
        path = build_index_path(CRAWL_ID)
        assert f"crawl={CRAWL_ID}" in path
        assert path.startswith("s3://commoncrawl/cc-index/table/cc-main/warc/")
        assert path.endswith("*.parquet")
        assert "subset=warc" in path

    def test_parts_override(self):
        path = build_index_path(CRAWL_ID, parts="part-00000-*.parquet")
        assert path.endswith("part-00000-*.parquet")
        assert "*.parquet" not in path.replace("part-00000-*.parquet", "")


class TestBuildTrHostQuery:
    def test_contains_crawl_id_and_core_filters(self):
        sql = build_tr_host_query(CRAWL_ID)
        assert CRAWL_ID in sql
        assert "fetch_status = 200" in sql
        assert "content_mime_detected = 'text/html'" in sql
        assert "url_host_name" in sql
        assert "COUNT(*)" in sql

    def test_primary_only_uses_tur_prefix(self):
        sql = build_tr_host_query(CRAWL_ID, primary_only=True)
        assert "content_languages LIKE 'tur%'" in sql
        # Turkish is ISO-639-3 'tur', never 'tr', in this column.
        assert "LIKE 'tr'" not in sql

    def test_all_turkish_uses_substring(self):
        sql = build_tr_host_query(CRAWL_ID, primary_only=False)
        assert "content_languages LIKE '%tur%'" in sql

    def test_group_having_order_limit(self):
        sql = build_tr_host_query(CRAWL_ID, min_pages=42, limit=123)
        assert "GROUP BY url_host_name" in sql
        assert "HAVING pages >= 42" in sql
        assert "ORDER BY pages DESC" in sql
        assert "LIMIT 123" in sql

    def test_default_crawl_id_is_a_valid_snapshot_token(self):
        assert DEFAULT_CRAWL_ID.startswith("CC-MAIN-")


class TestInjectionGuards:
    @pytest.mark.parametrize(
        "bad",
        [
            "CC-MAIN-2025-51') UNION SELECT 1--",
            "CC-MAIN-2025-51'; DROP TABLE x;--",
            "../../etc/passwd",
            "CC-MAIN-2025",  # wrong shape
            "evil",
        ],
    )
    def test_rejects_malicious_crawl_id(self, bad):
        with pytest.raises(ValueError, match="crawl_id"):
            build_index_path(bad)
        with pytest.raises(ValueError, match="crawl_id"):
            build_tr_host_query(bad)

    @pytest.mark.parametrize("bad", ["a'b", "x); DROP", "../p", "a b.parquet"])
    def test_rejects_malicious_parts(self, bad):
        with pytest.raises(ValueError, match="parts"):
            build_index_path(CRAWL_ID, parts=bad)
