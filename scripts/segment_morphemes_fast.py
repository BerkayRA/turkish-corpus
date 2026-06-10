"""CLI: hardened, fast, hang-proof morpheme segmentation for sp_morph / morpheme-BPE data.

A thin ``argparse`` wrapper over :func:`turkish_corpus.morph_segment.run_segmentation`. It
replaces ``turkish-llm``'s fragile ``segment_morphemes.py`` and emits the identical on-disk
representation (one line in -> one line out; morphemes joined by ``"▁"`` within a word; words
separated by spaces; blank / word-less lines dropped), so an ``sp_morph`` SentencePiece model
trains on the same text.

What makes it hang-proof / orphan-proof (see ``morph_segment`` for the full rationale):

* a per-word ``SIGALRM`` timeout (``--timeout``) abandons a word that stalls ``tr_api``;
* words longer than ``--max-len`` are passed through verbatim (never fed to the chart parser);
* unique words are segmented once (Turkish ~5.5% type/token ratio -> ~18x fewer analyses);
* the worker pool is torn down via a ``with`` block, and ``main()`` traps SIGINT/SIGTERM.

**Run it in its own process group** so a hard stop reaps every worker, e.g.::

    setsid uv run python scripts/segment_morphemes_fast.py corpus.txt -o out.morph.txt &
    kill -TERM -$!        # negative PID => signal the whole process group

Examples
--------
::

    uv run python scripts/segment_morphemes_fast.py data/corpus.clean.txt \\
        -o data/sample.morph.txt --max-lines 150000 --workers 8 \\
        --turkish-tokenizer-path /path/to/turkish-tokenizer
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

# Make the package importable when run as a loose script (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turkish_corpus.morph_segment import (  # noqa: E402
    DEFAULT_MAX_LEN,
    DEFAULT_PROGRESS_EVERY,
    DEFAULT_TIMEOUT_S,
    run_segmentation,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="segment_morphemes_fast",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input text file, one document per line.")
    p.add_argument(
        "-o",
        "--output",
        default="data/sample.morph.txt",
        help="Destination for the morpheme-segmented text.",
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Cap on input lines read (default: all).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Process count; 0 = all CPUs, 1 = in-process (no pool).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-word analysis timeout in seconds before passthrough (default: %(default)s).",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Words longer than this are passed through verbatim (default: %(default)s).",
    )
    p.add_argument(
        "--repo-path",
        default=None,
        help=(
            "Path to a BerkayRA/turkish-tokenizer clone (else set TURKISH_TOKENIZER_PATH)."
        ),
    )
    return p


def _install_kill_handlers() -> None:
    """Trap SIGINT/SIGTERM and raise ``KeyboardInterrupt`` to trigger the pool's ``with`` exit.

    Re-raising as an exception (rather than ``sys.exit`` in the handler) lets the
    ``with mp.Pool(...)`` block in :func:`run_segmentation` run its teardown
    (``pool.terminate()``), so workers are not orphaned on Ctrl-C / ``kill``.
    """

    def _raise(signum, frame):  # noqa: ANN001, ARG001 (signal handler signature)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise)
    # SIGINT already raises KeyboardInterrupt by default, but set it explicitly for symmetry.
    signal.signal(signal.SIGINT, _raise)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    _install_kill_handlers()

    try:
        stats = run_segmentation(
            args.input,
            args.output,
            max_lines=args.max_lines,
            workers=args.workers,
            timeout=args.timeout,
            max_len=args.max_len,
            repo_path=args.repo_path,
            progress_every=DEFAULT_PROGRESS_EVERY,
        )
    except KeyboardInterrupt:
        # The pool has already been terminated by the with-block teardown at this point.
        print("[segment_morphemes_fast] interrupted; workers terminated.", file=sys.stderr)
        return 130

    print(
        f"[segment_morphemes_fast] {stats['lines_in']:,} lines -> "
        f"{stats['lines_out']:,} segmented ({stats['lines_dropped']:,} dropped); "
        f"{stats['unique_words']:,} unique words "
        f"({stats['passthrough_words']:,} passthrough); "
        f"{stats['elapsed_s']:.0f}s, {stats['words_per_s']:.0f} unique words/s "
        f"-> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
