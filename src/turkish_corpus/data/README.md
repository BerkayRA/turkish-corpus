# Calibration artifacts

This directory holds the **derived** corpus-quality-filter artifacts produced by
`scripts/calibrate_filters.py` (the FineWeb-2 "canary language" method): stopwords and
Gopher/quality thresholds DERIVED from a clean Turkish reference corpus (Turkish Wikipedia)
rather than hand-set.

Expected files (committed once you run the calibration):

- `turkish_stopwords.txt` — one Turkish-lowercased stopword per line (UTF-8).
- `turkish_thresholds.json` — the numeric Gopher/quality threshold kwargs.
- `calibration_report.json` — provenance (n_docs, quantiles, chosen values).

`turkish_corpus.filters` loads these transparently via `importlib.resources`. When the
files are absent (as on a fresh checkout, before calibration) it falls back to the curated
defaults baked into `filters.py`, so the package works either way.

Regenerate by re-running `scripts/calibrate_filters.py`; see `docs/calibration.md`.
