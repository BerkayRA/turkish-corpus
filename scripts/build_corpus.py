"""CLI: run the whole corpus pipeline (ingest -> clean -> blend) from one recipe file.

Thin wrapper around :func:`turkish_corpus.orchestrate.main` (the ``tc-build-corpus`` console
entry); kept so the build is runnable as a loose script too::

    uv run --extra pipeline --extra sources --extra academic \\
        python scripts/build_corpus.py --recipe examples/recipe.json

See ``docs/end-to-end.md`` for the recipe schema, output layout, and idempotency/--force.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.orchestrate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
