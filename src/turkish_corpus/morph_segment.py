"""Hardened, hang-proof morpheme segmenter for ``sp_morph`` / morpheme-BPE training data.

This replaces ``turkish-llm``'s fragile ``segment_morphemes.py``. It produces the exact same
on-disk representation (one input line -> one output line; within a word, morphemes joined by
``"▁"`` (U+2581); words separated by a single space; blank/word-less lines dropped) so a
SentencePiece ``sp_morph`` model — or the morpheme-BPE trainer — sees identical text.

Why a rewrite (the four failure modes it fixes)
-----------------------------------------------
Segmenting raw Turkish web text with ``tr_api`` at scale repeatedly stalled because:

1. **Pathological long tokens** — URLs / concatenated junk (seen up to 215 chars) blow up
   ``tr_api``'s chart parser and hang *forever* on a single word. We never feed a word longer
   than ``max_len`` to the analyzer (pass it through verbatim as one token), and we arm a
   per-word ``SIGALRM`` timer so even a short word that pathologically stalls is abandoned and
   passed through. See :func:`segment_word`.
2. **The analyzer is slow** (~115 words/s). Turkish text has a ~5.5% type/token ratio, so we
   segment each *unique* word form exactly once and reconstruct every line by dict lookup —
   roughly 18x fewer analyzer calls. See :func:`run_segmentation`.
3. **``imap(chunksize=500)`` buffers output** behind the first ordered chunk, so nothing hits
   disk for a long time. We segment *unique words* with :meth:`~multiprocessing.pool.Pool.
   imap_unordered` (order is irrelevant — results land in a dict), stream progress to stderr,
   and flush periodically so progress is always visible.
4. **Orphaned spawn workers** saturated the CPU after a kill. The pool is managed by a ``with``
   block (terminates on context exit / exception) and ``main()`` installs SIGINT/SIGTERM
   handlers that raise to trigger that teardown. Run the CLI in its own process group so
   ``kill -TERM -<pgid>`` reaps every worker (documented on the CLI).

``signal.setitimer(ITIMER_REAL)`` / ``SIGALRM`` only fire on the **main thread of a process**.
That is fine here: ``segment_word`` runs on the main thread of the main process (single-worker
/ in-test path) *and* on the main thread of each ``multiprocessing`` worker process (the pool
runs the task inline on the worker's main thread, never a sub-thread), so the alarm is always
deliverable. It would *not* work if ``segment_word`` were called from a secondary thread.

The pure, in-process-testable core (:func:`segment_word`, :func:`segment_line`,
:func:`build_cache`) is deliberately separated from the multiprocessing orchestrator
(:func:`run_segmentation`) so the hang-proofing and dedup logic can be unit-tested without
spawning processes or importing the real ``tr_api``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "MORPH_SEP",
    "build_cache",
    "install_alarm_handler",
    "run_segmentation",
    "segment_line",
    "segment_word",
]

logger = logging.getLogger(__name__)

# Within-word morpheme separator. Word boundaries stay as spaces so the morpheme-BPE trainer
# (and sp_morph) can merge morphemes WITHIN a word but never across word boundaries, e.g.
# "geliyorum kitabı" -> "gel▁iyor▁um kitab▁ı". Must match turkish-llm/segment_morphemes.py.
MORPH_SEP = "▁"

# Defaults tuned for raw web text. A word over DEFAULT_MAX_LEN chars is treated as junk
# (URL / concatenated run-on) and passed through untouched rather than fed to tr_api's parser.
DEFAULT_TIMEOUT_S = 1.0
DEFAULT_MAX_LEN = 70
DEFAULT_PROGRESS_EVERY = 20_000


class _SegTimeout(Exception):
    """Raised from the SIGALRM handler to abandon a single pathological word analysis."""


# Only raise from the handler while a word is *actively* being analysed. Cleared before the
# timer is disarmed, so a signal delivered during/after disarm (a classic SIGALRM race) becomes
# a no-op instead of escaping uncaught and killing the worker. Per-process global; workers are
# single-threaded so no lock is needed.
_alarm_armed = False


def _on_alarm(signum: int, frame: Any) -> None:  # noqa: ARG001 (signal handler signature)
    """SIGALRM handler: convert the timer expiry into a catchable per-word timeout.

    Guarded by :data:`_alarm_armed` so a late/stray delivery (e.g. the timer firing during the
    ``finally`` disarm, after we have already left the analysis) is ignored rather than raised.
    """
    if _alarm_armed:
        raise _SegTimeout


_alarm_installed = False


def install_alarm_handler() -> None:
    """Register the module-level SIGALRM handler (idempotent).

    Called by the multiprocessing worker initializer and *lazily* by :func:`segment_word` if
    no handler is installed yet, so the in-process / single-worker path is hang-proof without
    any explicit setup. Must run on a process's main thread (signal handlers are process-wide
    and only deliverable there).
    """
    global _alarm_installed
    if not _alarm_installed:
        signal.signal(signal.SIGALRM, _on_alarm)
        _alarm_installed = True


def segment_word(
    word: str,
    tokenizer: Any,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_len: int = DEFAULT_MAX_LEN,
) -> str:
    """Segment one word into ``"▁"``-joined morphemes, hang-proof and crash-proof.

    Matches ``turkish-llm``'s ``MorphemeBPEAdapter`` per-word call exactly: analyse with
    ``split_clitics=False`` into morpheme ``"chunk"`` strings, fall back to ``[word]`` when the
    parse fails, and join with :data:`MORPH_SEP`.

    Hardening (this never hangs or raises out):

    * ``len(word) > max_len`` -> return ``word`` unchanged *without* calling the tokenizer.
      Long tokens are URLs / concatenated junk that blow up ``tr_api``'s chart parser; they
      carry no real morphology, so passing them through as a single token is both correct and
      safe.
    * Otherwise arm a ``timeout``-second ``SIGALRM`` timer around the analysis. On timeout
      (``_SegTimeout``) or *any* tokenizer exception, return ``word`` (single-token
      passthrough). The timer is always disarmed in a ``finally`` block so it can never leak
      into the next word.

    Parameters
    ----------
    word:
        A single whitespace-free token.
    tokenizer:
        Anything exposing ``tokenize(word, suggest=..., tail_repair=..., alternatives=...,
        split_clitics=...) -> {"parsed": bool, "morphemes": [{"chunk": str}, ...]}`` — the real
        ``tr_api.Tokenizer`` or a fake in tests.
    timeout:
        Per-word wall-clock budget in seconds before the analysis is abandoned.
    max_len:
        Words longer than this are passed through verbatim, never analysed.
    """
    if len(word) > max_len:
        # Junk / URL: do not feed the chart parser. One opaque token.
        return word

    global _alarm_armed
    install_alarm_handler()  # lazy: makes the in-process / single-worker path hang-proof.
    _alarm_armed = True
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = tokenizer.tokenize(
            word,
            suggest=False,
            tail_repair=False,
            alternatives=False,
            split_clitics=False,
        )
        morphs = (
            [m["chunk"] for m in result["morphemes"]] if result.get("parsed") else []
        )
        if not morphs:
            morphs = [word]
        return MORPH_SEP.join(morphs)
    except Exception:
        # _SegTimeout (pathological stall) OR any tr_api parse error -> safe passthrough so a
        # single bad word can never hang or crash the whole batch.
        return word
    finally:
        # Disarm the handler FIRST (so a timer firing during the setitimer(0) call below is a
        # no-op, not an uncaught _SegTimeout escaping the finally), then ALWAYS stop the timer
        # so a pending expiry never fires on the next word / unrelated main-process code.
        _alarm_armed = False
        signal.setitimer(signal.ITIMER_REAL, 0)


def segment_line(line: str, cache: dict[str, str]) -> str | None:
    """Reconstruct one segmented output line from a ``{word: segmented}`` cache.

    Pure (no analysis): split the line on whitespace, look up each word's pre-segmented form,
    and join with single spaces. Returns ``None`` for blank / word-less lines so the caller
    drops them (matching the original tool). A word missing from the cache falls back to
    itself, so reconstruction can never ``KeyError``.
    """
    words = line.split()
    if not words:
        return None
    return " ".join(cache.get(word, word) for word in words)


def build_cache(
    words: Iterable[str],
    segment_fn: Callable[[str], str],
) -> dict[str, str]:
    """Segment each *unique* word once via ``segment_fn`` into a ``{word: segmented}`` dict.

    Factored out (and taking an injectable per-word function) so the dedup logic is unit-
    testable with a fake segmenter and reused by the single-worker path of
    :func:`run_segmentation`. Deduping happens in the caller (``words`` should already be the
    unique set); this simply maps each one.
    """
    return {word: segment_fn(word) for word in words}


def _unique_words(lines: list[str], max_len: int) -> set[str]:
    """Collect the unique word forms across all lines.

    Words longer than ``max_len`` are still included (so :func:`segment_line` can look them
    up), but their segmentation will be a cheap passthrough — they are never sent to the
    analyzer by :func:`segment_word`.
    """
    seen: set[str] = set()
    for line in lines:
        seen.update(line.split())
    return seen


# --- Multiprocessing worker plumbing (module-level so it is picklable for spawn) ------------

# Per-worker state, populated by the initializer. Each worker builds its own Tokenizer once
# (constructing it is expensive) and reuses it for every word the pool hands it.
_WORKER_TOKENIZER: Any = None
_WORKER_TIMEOUT: float = DEFAULT_TIMEOUT_S
_WORKER_MAX_LEN: int = DEFAULT_MAX_LEN


def _build_tokenizer(repo_path: str | None) -> Any:
    """Construct a fast-config ``tr_api.Tokenizer`` (lazy import of the optional sibling repo).

    Fast config mirrors the rest of the project: ``suggest_on_oov=False``,
    ``include_alternatives=False``, ``correct_tail_typos=False`` — the suggestion /
    alternatives / typo-repair passes are slow and irrelevant to segmentation.
    """
    from turkish_corpus.morphology import (  # noqa: PLC0415 (lazy: avoids hard tr_api dep)
        ensure_tr_api_importable,
    )

    ensure_tr_api_importable(repo_path)
    from tr_api import Tokenizer, TokenizerConfig  # noqa: PLC0415 (optional, lazy)

    return Tokenizer(
        TokenizerConfig(
            suggest_on_oov=False,
            include_alternatives=False,
            correct_tail_typos=False,
        )
    )


def _init_worker(repo_path: str | None, timeout: float, max_len: int) -> None:
    """Pool initializer: install the SIGALRM handler and build this worker's Tokenizer once."""
    global _WORKER_TOKENIZER, _WORKER_TIMEOUT, _WORKER_MAX_LEN
    install_alarm_handler()
    _WORKER_TIMEOUT = timeout
    _WORKER_MAX_LEN = max_len
    _WORKER_TOKENIZER = _build_tokenizer(repo_path)


def _segment_word_task(word: str) -> tuple[str, str]:
    """Pool task: segment one unique word, returning ``(word, segmented)`` for the result dict.

    Returns the key alongside the value because ``imap_unordered`` does not preserve order.
    """
    return word, segment_word(
        word,
        _WORKER_TOKENIZER,
        timeout=_WORKER_TIMEOUT,
        max_len=_WORKER_MAX_LEN,
    )


def _read_lines(input_path: str | Path, max_lines: int | None) -> list[str]:
    """Read up to ``max_lines`` raw lines (newline-stripped) from ``input_path``."""
    path = Path(input_path)
    lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for i, raw in enumerate(handle):
            if max_lines is not None and i >= max_lines:
                break
            lines.append(raw.rstrip("\n"))
    return lines


def _segment_unique_parallel(
    words: list[str],
    *,
    repo_path: str | None,
    workers: int,
    timeout: float,
    max_len: int,
    progress_every: int,
) -> dict[str, str]:
    """Segment unique ``words`` across a process pool into a ``{word: segmented}`` dict.

    Uses ``imap_unordered`` (order is irrelevant — everything lands in a dict) so a slow word
    never blocks completed results from being recorded, and reports progress to the logger.
    The ``with`` block guarantees the pool is terminated on normal exit *and* on exception
    (e.g. the KeyboardInterrupt that ``main()``'s signal handlers raise), so workers are never
    orphaned. Chunk size is bounded so progress updates stay frequent on small inputs.
    """
    cache: dict[str, str] = {}
    total = len(words)
    chunksize = max(1, min(200, total // (workers * 4) or 1))
    start = time.monotonic()
    done = 0

    with mp.Pool(
        workers,
        initializer=_init_worker,
        initargs=(repo_path, timeout, max_len),
    ) as pool:
        for word, segmented in pool.imap_unordered(
            _segment_word_task, words, chunksize=chunksize
        ):
            cache[word] = segmented
            done += 1
            if done % progress_every == 0:
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed else 0.0
                logger.info(
                    "segmented %s/%s unique words (%.0f words/s, %.0fs elapsed)",
                    f"{done:,}",
                    f"{total:,}",
                    rate,
                    elapsed,
                )
    return cache


def _segment_unique_single(
    words: list[str],
    *,
    repo_path: str | None,
    timeout: float,
    max_len: int,
    progress_every: int,
) -> dict[str, str]:
    """In-process single-worker fallback (``workers == 1``): no pool, fully signal-safe.

    Also the path that tests can drive with a monkeypatched ``tr_api`` (no real
    multiprocessing). Builds one Tokenizer and segments each unique word with the same
    hang-proof :func:`segment_word`.
    """
    tokenizer = _build_tokenizer(repo_path)
    install_alarm_handler()
    total = len(words)
    start = time.monotonic()
    cache: dict[str, str] = {}
    for done, word in enumerate(words, start=1):
        cache[word] = segment_word(word, tokenizer, timeout=timeout, max_len=max_len)
        if done % progress_every == 0:
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed else 0.0
            logger.info(
                "segmented %s/%s unique words (%.0f words/s, %.0fs elapsed)",
                f"{done:,}",
                f"{total:,}",
                rate,
                elapsed,
            )
    return cache


def _write_output(
    lines: list[str],
    cache: dict[str, str],
    output_path: str | Path,
    *,
    flush_every: int,
) -> tuple[int, int]:
    """Reconstruct and stream every output line, flushing periodically; return ``(in, out)``.

    Streaming + periodic ``flush()`` means progress is visible on disk during a long run (no
    "0 bytes for 20 minutes" mystery). Word-less lines are dropped (``segment_line`` -> None).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            n_in += 1
            segmented = segment_line(line, cache)
            if segmented is not None:
                handle.write(segmented + "\n")
                n_out += 1
            if n_in % flush_every == 0:
                handle.flush()
    return n_in, n_out


def run_segmentation(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_lines: int | None = None,
    workers: int = 0,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_len: int = DEFAULT_MAX_LEN,
    repo_path: str | None = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
) -> dict[str, Any]:
    """Morpheme-segment a corpus file, deduping unique words and streaming output.

    Pipeline: read lines -> collect the unique word set -> segment each unique word *once*
    (in parallel unless ``workers == 1``) into a ``{word: segmented}`` cache -> reconstruct
    every line by dict lookup and stream it to ``output_path``. One input line maps to one
    output line; blank / word-less lines are dropped.

    Parameters
    ----------
    input_path, output_path:
        Source corpus (one document per line) and destination for segmented text.
    max_lines:
        Cap on input lines read (``None`` = all).
    workers:
        Process count; ``0`` -> ``cpu_count()``. ``1`` runs in-process (no pool), which is the
        fully signal-safe path and what tests exercise with a fake ``tr_api``.
    timeout, max_len:
        Per-word hardening knobs forwarded to :func:`segment_word`.
    repo_path:
        Path to a ``turkish-tokenizer`` clone (else resolved from ``TURKISH_TOKENIZER_PATH``).
    progress_every:
        Log a progress line every this many *unique words* segmented.

    Returns
    -------
    dict
        Stats: ``lines_in``, ``lines_out``, ``lines_dropped``, ``unique_words``,
        ``passthrough_words`` (unique words returned unchanged — junk / unparsed / timed out),
        ``elapsed_s``, ``words_per_s`` (unique words / second).
    """
    resolved_workers = workers or mp.cpu_count()
    start = time.monotonic()

    lines = _read_lines(input_path, max_lines)
    words = sorted(_unique_words(lines, max_len))

    if resolved_workers == 1:
        cache = _segment_unique_single(
            words,
            repo_path=repo_path,
            timeout=timeout,
            max_len=max_len,
            progress_every=progress_every,
        )
    else:
        cache = _segment_unique_parallel(
            words,
            repo_path=repo_path,
            workers=resolved_workers,
            timeout=timeout,
            max_len=max_len,
            progress_every=progress_every,
        )

    n_in, n_out = _write_output(
        lines, cache, output_path, flush_every=progress_every
    )

    elapsed = time.monotonic() - start
    unique = len(words)
    # A word is "passthrough" when its segmented form has no morpheme separator and equals the
    # word itself: junk (>max_len), an unparsed single token, or a timed-out analysis.
    passthrough = sum(
        1 for word, seg in cache.items() if seg == word and MORPH_SEP not in seg
    )
    return {
        "lines_in": n_in,
        "lines_out": n_out,
        "lines_dropped": n_in - n_out,
        "unique_words": unique,
        "passthrough_words": passthrough,
        "elapsed_s": elapsed,
        "words_per_s": unique / elapsed if elapsed else 0.0,
    }


def segment_corpus(
    lines: Iterable[str],
    segment_fn: Callable[[str], str],
    *,
    max_len: int = DEFAULT_MAX_LEN,
) -> tuple[Iterator[str], dict[str, str]]:
    """In-memory dedup+reconstruct over ``lines`` using an injected per-word ``segment_fn``.

    The pure counterpart to :func:`run_segmentation` (no I/O, no multiprocessing): builds the
    unique-word cache once via ``segment_fn`` then yields one reconstructed output line per
    non-empty input line. Returns ``(output_iterator, cache)`` so callers can inspect the
    ``{word: segmented}`` cache (e.g. to confirm a word was segmented exactly once).
    """
    materialized = list(lines)
    cache = build_cache(_unique_words(materialized, max_len), segment_fn)

    def _iter() -> Iterator[str]:
        for line in materialized:
            segmented = segment_line(line, cache)
            if segmented is not None:
                yield segmented

    return _iter(), cache
