"""Tests for the corpus BLEND/MIXER (pure; whitespace counter, no network)."""

import gzip
import json
import warnings

import pytest

from turkish_corpus.blend import (
    BlendSource,
    _shards,
    _write_selected,
    build_blend,
    iter_jsonl,
)
from turkish_corpus.sources.base import SourceInfo, make_record, write_records


def _make_source_dir(tmp_path, name, license_, register, docs):
    """Write ``docs`` (list of (id, text)) as a cleaned gzipped JSONL dir; return its path."""
    info = SourceInfo(name=name, license=license_, register=register)
    records = [make_record(text, doc_id, info) for doc_id, text in docs]
    out = tmp_path / "clean" / name
    write_records(records, str(out), shard_name="00000.jsonl.gz")
    return str(out)


def _read_blend_shard(out_dir, name):
    """Read back the blended gzipped shard for one source as a list of records."""
    path = out_dir / "blend" / f"{name}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestIterJsonl:
    def test_reads_both_plain_and_gz(self, tmp_path):
        info = SourceInfo(name="x", license="L", register="web")
        # One gz shard, one plain shard, in the same dir.
        write_records(
            [make_record("alfa beta", "a", info)], str(tmp_path), shard_name="00000.jsonl.gz"
        )
        write_records(
            [make_record("gama delta", "b", info)], str(tmp_path), shard_name="00001.jsonl"
        )
        ids = sorted(rec["id"] for rec in iter_jsonl(str(tmp_path)))
        assert ids == ["a", "b"]


class TestBuildBlend:
    def test_weighting_manifest_and_outputs(self, tmp_path):
        # wikipedia: 10 docs * 5 words = 50 tokens available (whitespace counter).
        # govlegal:  10 docs * 5 words = 50 tokens available.
        five_words = "bir iki uc dort bes"
        wiki_path = _make_source_dir(
            tmp_path, "wikipedia", "CC BY-SA 3.0", "encyclopedic",
            [(f"w{i}", five_words) for i in range(10)],
        )
        legal_path = _make_source_dir(
            tmp_path, "govlegal", "public", "legal",
            [(f"g{i}", five_words) for i in range(10)],
        )
        out = tmp_path / "blend_out"
        manifest = build_blend(
            [
                BlendSource("wikipedia", wiki_path, weight=0.25, license="CC BY-SA 3.0",
                            register="encyclopedic"),
                BlendSource("govlegal", legal_path, weight=0.75, license="public",
                            register="legal"),
            ],
            str(out),
            target_tokens=40,
            tokenizer_spec=None,
            shuffle_seed=7,
        )

        by_name = {s["name"]: s for s in manifest["sources"]}
        # Targets: 0.25*40=10, 0.75*40=30.
        assert by_name["wikipedia"]["target_tokens"] == 10
        assert by_name["govlegal"]["target_tokens"] == 30
        # Achieved >= target, overshoot bounded by one doc (5 tokens) since docs are 5 each.
        for name, target in (("wikipedia", 10), ("govlegal", 30)):
            achieved = by_name[name]["achieved_tokens"]
            assert achieved >= target
            assert achieved < target + 5  # at most one extra 5-token doc
            assert by_name[name]["shortfall_tokens"] == 0

        # manifest.json written with correct structure.
        manifest_path = out / "manifest.json"
        assert manifest_path.is_file()
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert on_disk["totals"]["tokenizer"] == "whitespace"
        assert on_disk["totals"]["target_tokens"] == 40

        # Totals add up across sources.
        assert manifest["totals"]["achieved_tokens"] == sum(
            s["achieved_tokens"] for s in manifest["sources"]
        )
        assert manifest["totals"]["docs"] == sum(s["docs"] for s in manifest["sources"])

        # Output blend dir has real records carrying provenance.
        wiki_recs = _read_blend_shard(out, "wikipedia")
        assert len(wiki_recs) == by_name["wikipedia"]["docs"]
        assert wiki_recs[0]["metadata"]["source"] == "wikipedia"
        assert wiki_recs[0]["text"] == five_words

    def test_shortfall_reported_when_source_too_small(self, tmp_path):
        # Only 2 docs * 3 words = 6 tokens available, target wants 30.
        small_path = _make_source_dir(
            tmp_path, "tiny", "L", "web", [("t0", "a b c"), ("t1", "d e f")]
        )
        out = tmp_path / "blend_out"
        manifest = build_blend(
            [BlendSource("tiny", small_path, weight=1.0)],
            str(out),
            target_tokens=30,
            tokenizer_spec=None,
        )
        entry = manifest["sources"][0]
        assert entry["target_tokens"] == 30
        assert entry["achieved_tokens"] == 6  # everything taken
        assert entry["available_tokens"] == 6
        assert entry["shortfall_tokens"] == 24
        assert entry["docs"] == 2
        assert manifest["totals"]["shortfall_tokens"] == 24

    def test_weights_need_not_sum_to_one(self, tmp_path):
        # Weights 1 and 3 -> normalized 0.25 / 0.75, same as the weighting test.
        five_words = "bir iki uc dort bes"
        a_path = _make_source_dir(
            tmp_path, "a", "L", "web", [(f"a{i}", five_words) for i in range(10)]
        )
        b_path = _make_source_dir(
            tmp_path, "b", "L", "legal", [(f"b{i}", five_words) for i in range(10)]
        )
        out = tmp_path / "blend_out"
        manifest = build_blend(
            [BlendSource("a", a_path, weight=1.0), BlendSource("b", b_path, weight=3.0)],
            str(out),
            target_tokens=40,
            tokenizer_spec=None,
        )
        by_name = {s["name"]: s for s in manifest["sources"]}
        assert abs(by_name["a"]["weight"] - 0.25) < 1e-9
        assert abs(by_name["b"]["weight"] - 0.75) < 1e-9
        assert by_name["a"]["target_tokens"] == 10
        assert by_name["b"]["target_tokens"] == 30

    def test_deterministic_with_seed(self, tmp_path):
        five_words = "bir iki uc dort bes"
        path = _make_source_dir(
            tmp_path, "s", "L", "web", [(f"s{i}", five_words) for i in range(20)]
        )
        src = [BlendSource("s", path, weight=1.0)]
        out1, out2 = tmp_path / "o1", tmp_path / "o2"
        m1 = build_blend(src, str(out1), target_tokens=25, tokenizer_spec=None, shuffle_seed=3)
        m2 = build_blend(src, str(out2), target_tokens=25, tokenizer_spec=None, shuffle_seed=3)
        ids1 = sorted(r["id"] for r in _read_blend_shard(out1, "s"))
        ids2 = sorted(r["id"] for r in _read_blend_shard(out2, "s"))
        assert ids1 == ids2
        assert m1["sources"][0]["docs"] == m2["sources"][0]["docs"]

    def test_per_source_seed_is_independent_but_reproducible(self, tmp_path):
        # Two sources with IDENTICAL contents must not pick the SAME subset (per-source seed),
        # yet each must pick the same subset across two runs (string seed, not salted hash()).
        five_words = "bir iki uc dort bes"
        docs = [(f"d{i}", five_words) for i in range(30)]
        a_path = _make_source_dir(tmp_path, "a", "L", "web", docs)
        b_path = _make_source_dir(tmp_path, "b", "L", "web", docs)
        src = [BlendSource("a", a_path, weight=1.0), BlendSource("b", b_path, weight=1.0)]

        out1, out2 = tmp_path / "o1", tmp_path / "o2"
        build_blend(src, str(out1), target_tokens=50, tokenizer_spec=None, shuffle_seed=0)
        build_blend(src, str(out2), target_tokens=50, tokenizer_spec=None, shuffle_seed=0)

        a1 = sorted(r["id"] for r in _read_blend_shard(out1, "a"))
        b1 = sorted(r["id"] for r in _read_blend_shard(out1, "b"))
        a2 = sorted(r["id"] for r in _read_blend_shard(out2, "a"))
        b2 = sorted(r["id"] for r in _read_blend_shard(out2, "b"))

        # Reproducible across runs (would fail if seeded with the process-salted hash()).
        assert a1 == a2
        assert b1 == b2
        # Independent shuffles: identical sources do not select the identical subset.
        assert a1 != b1

    def test_happy_path_docs_equal_selected_no_warning(self, tmp_path):
        # On a stable input, pass 2 emits exactly what pass 1 selected → no corruption warning.
        five_words = "bir iki uc dort bes"
        path = _make_source_dir(
            tmp_path, "s", "L", "web", [(f"s{i}", five_words) for i in range(10)]
        )
        out = tmp_path / "blend_out"
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            manifest = build_blend(
                [BlendSource("s", path, weight=1.0)],
                str(out),
                target_tokens=20,
                tokenizer_spec=None,
            )
        # docs reported equals the number of shard records actually written.
        assert manifest["sources"][0]["docs"] == len(_read_blend_shard(out, "s"))


class TestWriteSelected:
    def test_warns_when_selection_references_missing_line(self, tmp_path):
        # A (shard, line_no) that doesn't exist in the shard makes pass 2 under-emit; the
        # mismatch (docs != len(selected)) is what build_blend warns on. Here we exercise the
        # underlying count mismatch directly via _write_selected.
        info = SourceInfo(name="s", license="L", register="web")
        src_dir = tmp_path / "src"
        write_records(
            [make_record(t, i, info) for i, t in (("a", "x y z"), ("b", "u v w"))],
            str(src_dir),
            shard_name="00000.jsonl.gz",
        )
        shard = _shards(str(src_dir))[0]
        # Select line 0 (exists) and line 99 (does not) → only one line written.
        selected = {(shard, 0), (shard, 99)}
        import io

        buf = io.StringIO()
        docs = _write_selected(str(src_dir), selected, buf)
        assert docs == 1
        assert docs != len(selected)  # build_blend turns this inequality into a RuntimeWarning


class TestShards:
    def test_warns_on_duplicate_stem(self, tmp_path):
        info = SourceInfo(name="s", license="L", register="web")
        # Same stem (00000) as both a plain and a gz shard → double-count hazard.
        write_records(
            [make_record("a b c", "a", info)], str(tmp_path), shard_name="00000.jsonl"
        )
        write_records(
            [make_record("d e f", "b", info)], str(tmp_path), shard_name="00000.jsonl.gz"
        )
        with pytest.warns(RuntimeWarning, match="duplicate shard stem"):
            _shards(str(tmp_path))

    def test_no_warning_on_distinct_stems(self, tmp_path):
        info = SourceInfo(name="s", license="L", register="web")
        write_records(
            [make_record("a b c", "a", info)], str(tmp_path), shard_name="00000.jsonl"
        )
        write_records(
            [make_record("d e f", "b", info)], str(tmp_path), shard_name="00001.jsonl.gz"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            shards = _shards(str(tmp_path))
        assert len(shards) == 2
