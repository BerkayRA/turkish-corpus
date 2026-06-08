"""Tests for the end-to-end corpus orchestrator (pure: no network, no datatrove runs).

The heavy stages (run_local, build_blend, the ingest_* functions) are monkeypatched so the
orchestration logic — recipe loading/validation, the raw/clean/blend sequencing, reader
selection, and idempotency — is exercised without any extra installed.
"""

import json

import pytest

from turkish_corpus import orchestrate
from turkish_corpus.orchestrate import (
    CorpusRecipe,
    CorpusStep,
    _parse_token_budget,
    load_recipe,
    main,
    run_corpus,
)


def _recipe_dict(tmp_path, **overrides):
    """A minimal valid recipe dict (one hplt source) with optional field overrides."""
    base = {
        "output_root": str(tmp_path / "corpus"),
        "target_tokens": "1e6",
        "tokenizer": None,
        "tasks": 2,
        "sources": [
            {"name": "hplt", "kind": "hplt", "weight": 0.5, "language": "tur_Latn"},
        ],
    }
    base.update(overrides)
    return base


class TestParseTokenBudget:
    def test_scientific_notation(self):
        assert _parse_token_budget("20e9") == 20_000_000_000

    def test_underscore_separators(self):
        assert _parse_token_budget("20_000_000_000") == 20_000_000_000

    def test_plain_int_passthrough(self):
        assert _parse_token_budget(15) == 15

    def test_plain_int_string(self):
        assert _parse_token_budget("15000000000") == 15_000_000_000

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _parse_token_budget("0")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            _parse_token_budget("not-a-number")


class TestLoadRecipe:
    def test_round_trips_fields(self, tmp_path):
        path = tmp_path / "recipe.json"
        path.write_text(json.dumps(_recipe_dict(tmp_path, target_tokens="2e6", tasks=8)))

        recipe = load_recipe(str(path))

        assert recipe.output_root == str(tmp_path / "corpus")
        assert recipe.target_tokens == 2_000_000
        assert recipe.tokenizer is None
        assert recipe.tasks == 8
        assert len(recipe.sources) == 1
        step = recipe.sources[0]
        assert step.name == "hplt"
        assert step.kind == "hplt"
        assert step.weight == 0.5
        # Flattened kind-specific keys land in params.
        assert step.params["language"] == "tur_Latn"

    def test_flattened_and_nested_params_both_work(self, tmp_path):
        data = _recipe_dict(tmp_path)
        data["sources"] = [
            {"name": "wiki", "kind": "wikipedia", "weight": 1.0, "params": {"dump": "20231101.tr"}},
        ]
        path = tmp_path / "recipe.json"
        path.write_text(json.dumps(data))

        recipe = load_recipe(str(path))
        assert recipe.sources[0].params["dump"] == "20231101.tr"

    def test_missing_top_level_keys_raise(self, tmp_path):
        path = tmp_path / "recipe.json"
        path.write_text(json.dumps({"sources": []}))
        with pytest.raises(ValueError):
            load_recipe(str(path))


class TestRecipeValidate:
    def _step(self, **kw):
        defaults = {"name": "s", "kind": "hplt", "weight": 1.0}
        defaults.update(kw)
        return CorpusStep(**defaults)

    def _recipe(self, sources, **kw):
        defaults = {"output_root": "out", "target_tokens": 1000}
        defaults.update(kw)
        return CorpusRecipe(sources=sources, **defaults)

    def test_rejects_empty_sources(self):
        with pytest.raises(ValueError, match="at least one source"):
            self._recipe([]).validate()

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match="target_tokens"):
            self._recipe([self._step()], target_tokens=0).validate()

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            self._recipe([self._step(name="a"), self._step(name="a")]).validate()

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown kind"):
            self._recipe([self._step(kind="bogus")]).validate()

    def test_rejects_non_positive_weight(self):
        with pytest.raises(ValueError, match="weight"):
            self._recipe([self._step(weight=0)]).validate()

    def test_academic_requires_pdf_dir(self):
        with pytest.raises(ValueError, match="pdf_dir"):
            self._recipe([self._step(name="ac", kind="academic")]).validate()

    def test_jsonl_requires_raw_dir(self):
        with pytest.raises(ValueError, match="raw_dir"):
            self._recipe([self._step(name="j", kind="jsonl")]).validate()

    def test_valid_recipe_passes(self):
        steps = [
            self._step(name="hplt", kind="hplt", weight=0.6),
            self._step(name="ac", kind="academic", weight=0.4, params={"pdf_dir": "/pdfs"}),
        ]
        self._recipe(steps).validate()  # no raise


def _install_mocks(monkeypatch):
    """Patch run_local, build_blend, and the ingest_* functions to record calls.

    Returns a ``calls`` dict capturing what the orchestrator invoked, so tests can assert on
    reader selection, ingest out-dirs, blend sources, etc. — without any real work.
    """
    calls = {"ingest": [], "run_local": [], "build_blend": []}

    def fake_run_local(config):
        calls["run_local"].append(config)
        # Simulate the pipeline writing a populated final dir so the manifest/blend can read it.
        final = config.output_path + "/final"
        import os

        os.makedirs(final, exist_ok=True)
        with open(final + "/00000.jsonl", "w", encoding="utf-8") as fh:
            fh.write('{"id":"x","text":"a","metadata":{}}\n')
        return object()

    def fake_build_blend(sources, out_dir, *, target_tokens, tokenizer_spec, shuffle_seed):
        calls["build_blend"].append(
            {
                "sources": list(sources),
                "out_dir": out_dir,
                "target_tokens": target_tokens,
                "tokenizer_spec": tokenizer_spec,
                "shuffle_seed": shuffle_seed,
            }
        )
        return {
            "sources": [
                {
                    "name": s.name,
                    "achieved_tokens": 100,
                    "target_tokens": 100,
                    "docs": 1,
                    "shortfall_tokens": 0,
                }
                for s in sources
            ],
            "totals": {
                "target_tokens": target_tokens,
                "achieved_tokens": 100 * len(sources),
                "docs": len(sources),
                "shortfall_tokens": 0,
                "tokenizer": tokenizer_spec or "whitespace",
            },
        }

    def make_ingester(kind):
        def _fake(out_dir, **kwargs):
            calls["ingest"].append({"kind": kind, "out_dir": out_dir, "kwargs": kwargs})
            return 3

        return _fake

    # run_local / build_blend are imported lazily INSIDE the package modules, so patch them
    # at their definition site (which the lazy import resolves to).
    monkeypatch.setattr("turkish_corpus.pipeline.run_local", fake_run_local)
    monkeypatch.setattr("turkish_corpus.blend.build_blend", fake_build_blend)
    monkeypatch.setattr(
        "turkish_corpus.sources.wikipedia.ingest_wikipedia", make_ingester("wikipedia")
    )
    monkeypatch.setattr(
        "turkish_corpus.sources.govlegal.ingest_mevzuat", make_ingester("mevzuat")
    )
    monkeypatch.setattr(
        "turkish_corpus.sources.academic.ingest_dergipark", make_ingester("dergipark")
    )
    monkeypatch.setattr(
        "turkish_corpus.sources.academic.ingest_yoktez", make_ingester("yoktez")
    )
    return calls


class TestRunCorpus:
    def _full_recipe(self, tmp_path):
        return CorpusRecipe(
            output_root=str(tmp_path / "corpus"),
            target_tokens=1_000_000,
            tokenizer=None,
            tasks=2,
            sources=[
                CorpusStep("hplt", "hplt", 0.5, params={"language": "tur_Latn"}),
                CorpusStep("wikipedia", "wikipedia", 0.2, params={"dump": "20231101.tr"}),
                CorpusStep("mevzuat", "mevzuat", 0.2),
                CorpusStep(
                    "academic", "academic", 0.1, params={"pdf_dir": "/pdfs", "source": "dergipark"}
                ),
                CorpusStep(
                    "scraped", "jsonl", 0.1, params={"raw_dir": str(tmp_path / "scraped_raw")}
                ),
            ],
        )

    def test_orchestration_calls(self, tmp_path, monkeypatch):
        calls = _install_mocks(monkeypatch)
        recipe = self._full_recipe(tmp_path)

        manifest = run_corpus(recipe)

        # Ingesters run only for wikipedia/mevzuat/academic, into raw/<name>.
        ingest_kinds = {c["kind"] for c in calls["ingest"]}
        assert ingest_kinds == {"wikipedia", "mevzuat", "dergipark"}
        by_kind = {c["kind"]: c for c in calls["ingest"]}
        assert by_kind["wikipedia"]["out_dir"].endswith("raw/wikipedia")
        assert by_kind["mevzuat"]["out_dir"].endswith("raw/mevzuat")
        assert by_kind["dergipark"]["out_dir"].endswith("raw/academic")
        assert by_kind["dergipark"]["kwargs"]["pdf_dir"] == "/pdfs"

        # run_local called once per source with the right reader source + output path.
        assert len(calls["run_local"]) == 5
        cfgs = {c.output_path.split("/")[-1]: c for c in calls["run_local"]}
        assert cfgs["hplt"].reader.source == "hf"
        assert cfgs["hplt"].reader.language == "tur_Latn"
        assert cfgs["wikipedia"].reader.source == "jsonl"
        assert cfgs["wikipedia"].reader.data_path.endswith("raw/wikipedia")
        assert cfgs["scraped"].reader.source == "jsonl"
        assert cfgs["scraped"].reader.data_path == str(tmp_path / "scraped_raw")
        assert all(c.tasks == 2 for c in calls["run_local"])

        # build_blend got one BlendSource per source pointing at clean/<name>/final.
        assert len(calls["build_blend"]) == 1
        blend = calls["build_blend"][0]
        paths = {s.name: s.path for s in blend["sources"]}
        for name in ("hplt", "wikipedia", "mevzuat", "academic", "scraped"):
            assert paths[name].endswith(f"clean/{name}/final")
        weights = {s.name: s.weight for s in blend["sources"]}
        assert weights["hplt"] == 0.5
        assert blend["target_tokens"] == 1_000_000

        # Corpus manifest written to disk and returned.
        manifest_path = tmp_path / "corpus" / "corpus_manifest.json"
        assert manifest_path.is_file()
        assert manifest["recipe"]["target_tokens"] == 1_000_000
        assert manifest["blend"]["totals"]["docs"] == 5

    def test_yoktez_source_dispatch(self, tmp_path, monkeypatch):
        calls = _install_mocks(monkeypatch)
        recipe = CorpusRecipe(
            output_root=str(tmp_path / "corpus"),
            target_tokens=1000,
            sources=[
                CorpusStep("thesis", "academic", 1.0, params={"pdf_dir": "/t", "source": "yoktez"}),
            ],
        )
        run_corpus(recipe)
        assert calls["ingest"][0]["kind"] == "yoktez"

    def test_idempotent_skip_and_force(self, tmp_path, monkeypatch):
        calls = _install_mocks(monkeypatch)
        recipe = CorpusRecipe(
            output_root=str(tmp_path / "corpus"),
            target_tokens=1000,
            sources=[CorpusStep("hplt", "hplt", 1.0)],
        )

        # First run cleans once.
        run_corpus(recipe)
        assert len(calls["run_local"]) == 1

        # Second run skips because clean/hplt/final exists and is populated.
        run_corpus(recipe)
        assert len(calls["run_local"]) == 1

        # force=True re-runs cleaning.
        run_corpus(recipe, force=True)
        assert len(calls["run_local"]) == 2


class TestMain:
    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        assert "recipe" in capsys.readouterr().out

    def test_runs_and_prints_summary(self, tmp_path, monkeypatch, capsys):
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(_recipe_dict(tmp_path)))

        captured = {}

        def fake_run_corpus(recipe, *, force=False):
            captured["recipe"] = recipe
            captured["force"] = force
            return {
                "recipe": {"output_root": recipe.output_root},
                "blend": {
                    "sources": [
                        {
                            "name": "hplt",
                            "achieved_tokens": 50,
                            "target_tokens": 50,
                            "docs": 1,
                            "shortfall_tokens": 0,
                        }
                    ],
                    "totals": {
                        "achieved_tokens": 50,
                        "target_tokens": 50,
                        "docs": 1,
                        "shortfall_tokens": 0,
                        "tokenizer": "whitespace",
                    },
                },
            }

        monkeypatch.setattr(orchestrate, "run_corpus", fake_run_corpus)

        rc = main(["--recipe", str(recipe_path), "--target-tokens", "5e6", "--tasks", "3"])

        assert rc == 0
        # Overrides applied.
        assert captured["recipe"].target_tokens == 5_000_000
        assert captured["recipe"].tasks == 3
        out = capsys.readouterr().out
        assert "TOTAL" in out
        assert "corpus_manifest.json" in out
