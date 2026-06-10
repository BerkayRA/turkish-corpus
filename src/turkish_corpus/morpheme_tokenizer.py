"""Production morpheme-BPE tokenizer: integer vocabulary, encode/decode, byte-fallback.

This turns the fertility-only :class:`turkish_corpus.morpheme_bpe.MorphemeBPE` (which only
emits piece *strings*) into a real LLM tokenizer that an embedding table can consume: a
fixed-size integer vocabulary, ``encode(text) -> list[int]`` / ``decode(ids) -> str``,
total coverage via byte-fallback, and an optional precomputed ``word -> pieces`` lookup
table so the hot path never touches the slow ``tr_api`` analyzer.

Why this module exists
----------------------
``MorphemeBPE`` is the project's best/most-unique tokenizer (held-out fertility 1.158,
morpheme-aligned by construction) but cannot train or serve an LLM: it has no vocab/ids, no
word-boundary handling, no OOV coverage, and needs a ~500-700 words/sec ``tr_api`` analysis
per word. :class:`MorphemeTokenizer` adds exactly those missing layers on top of the proven
merge engine without re-implementing it.

Vocabulary layout (fixed-size, default 64000)
---------------------------------------------
Integer ids are assigned in a fixed order so the layout is stable and reproducible:

* ``0..4``   — specials: PAD, UNK, BOS, EOS, WORD_BOUNDARY. WORD_BOUNDARY is the ``▁``
  "preceded-by-space" marker emitted before every word (so decode can re-insert spaces).
* ``5..260`` — the 256 byte tokens, one per byte value ``0x00..0xFF``. These give *total*
  coverage: any morpheme not in the vocab is encoded as its raw UTF-8 bytes, so the
  tokenizer can represent (and round-trip) arbitrary text.
* ``261..(vocab_size-1)`` — morpheme/merge PIECES, ranked by corpus frequency (most
  frequent first), filling the remaining id space up to ``vocab_size``.

Special and byte tokens cannot collide with real morpheme pieces because they are keyed
internally by sentinel tuples (``("<special>", name)`` / ``("<byte>", value)``), never by a
plain string, while morpheme pieces are keyed by their plain ``str`` surface. ``piece_to_id``
maps both the sentinel tuples and the morpheme strings to ids; ``id_to_piece`` is the inverse
list (sentinel tuple for specials/bytes, ``str`` for morphemes).

Round-trip guarantee
--------------------
``decode(encode(text))`` reconstructs the surface text for whitespace-normalized input: words
are re-joined with single spaces, in-word pieces are concatenated, and byte-token runs are
decoded back to their UTF-8 string. The leading WORD_BOUNDARY (before the first word) is
stripped so there is no spurious leading space. Input is assumed already
whitespace-normalized by the upstream corpus pipeline — this tokenizer does NOT re-normalize
(NFC, casing, etc.); it only splits on whitespace.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .morpheme_bpe import MorphemeBPE

if TYPE_CHECKING:
    import collections

__all__ = [
    "MorphemeTokenizer",
    "DEFAULT_VOCAB_SIZE",
    "MORPH_SEP",
]

# Within-word morpheme separator carried in the merges file / used to serialize the table.
# Matches turkish_corpus.morph_segment.MORPH_SEP and the merges file's "morph_sep" key.
MORPH_SEP = "▁"

# Default fixed vocabulary size (specials + 256 bytes + morpheme pieces).
DEFAULT_VOCAB_SIZE = 64_000

# Words longer than this are junk (URLs / concatenated run-ons) that blow up tr_api's chart
# parser; treat them as a single piece (byte-fallback at id time). Matches morph_segment.
MAX_WORD_LEN = 70

# Default per-word lru_cache size for the (slow) analyzer path. Covers ~96% of tokens by
# frequency for Turkish text; bounds memory for long-running serving. Mirrors the counter.
DEFAULT_CACHE_SIZE = 1_000_000

# Ordered special-token names; their ids are their index (PAD=0 ... WORD_BOUNDARY=4). The
# WORD_BOUNDARY marker is emitted before each word so decode can re-insert word spacing.
_SPECIAL_NAMES = ("PAD", "UNK", "BOS", "EOS", "WORD_BOUNDARY")

# Number of byte tokens (one per possible byte value).
_N_BYTES = 256

# Sentinel piece-key prefixes so specials/bytes can live in piece_to_id without ever
# colliding with a real morpheme string key (a tuple is never equal to a str).
_SPECIAL_TAG = "<special>"
_BYTE_TAG = "<byte>"

# On-disk format version for tokenizer.json.
_FORMAT_VERSION = 1

# Filenames written by save() / read by load().
_TOKENIZER_FILE = "tokenizer.json"
_TABLE_FILE = "table.tsv"


class MorphemeTokenizer:
    """Morpheme-BPE tokenizer with an integer vocabulary, byte-fallback, and a fast table.

    Wraps the pure :class:`MorphemeBPE` merge engine (the proven, unit-tested merge loop) with
    everything an LLM needs: a fixed-size integer vocabulary, ``encode``/``decode`` over ids,
    total coverage via byte-fallback for OOV morphemes, optional BOS/EOS, and an optional
    precomputed ``word -> pieces`` table so the hot path is a dict lookup instead of the slow
    ``tr_api`` analysis.

    Construct via :meth:`build` (from a piece-frequency Counter) or :meth:`load` (from a saved
    directory) rather than directly; the ``__init__`` takes already-assembled vocab structures.

    Parameters
    ----------
    bpe:
        The merge engine (reconstructed from the merges list).
    piece_to_id:
        Maps each vocab entry to its integer id: sentinel tuples ``("<special>", name)`` and
        ``("<byte>", value)`` for specials/bytes, plain ``str`` for morpheme pieces.
    id_to_piece:
        Inverse of ``piece_to_id`` as a list indexed by id (sentinel tuple or ``str``).
    morph_sep:
        Within-word morpheme separator from the merges file (informational; pieces are stored
        already merged).
    merges:
        The ordered merge pairs (kept so :meth:`save` can serialize them).
    table:
        Optional precomputed ``word -> piece strings`` map; when set, encoding a tabled word is
        a dict lookup with no analyzer call.
    repo_path:
        Optional path to the ``turkish-tokenizer`` clone for the lazy ``tr_api`` bridge. Only
        used when a word is neither tabled nor cached, so :meth:`load` works without ``tr_api``.
    cache_size:
        Per-word ``lru_cache`` size for the analyzer fallback path (``None`` = unbounded,
        ``0``/negative = disabled).
    """

    def __init__(
        self,
        bpe: MorphemeBPE,
        piece_to_id: dict[str | tuple[str, str | int], int],
        id_to_piece: list[str | tuple[str, str | int]],
        *,
        morph_sep: str = MORPH_SEP,
        merges: list[tuple[str, str]] | None = None,
        table: dict[str, list[str]] | None = None,
        repo_path: str | None = None,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._bpe = bpe
        self.piece_to_id = piece_to_id
        self.id_to_piece = id_to_piece
        self.morph_sep = morph_sep
        self._merges = list(merges) if merges is not None else []
        self._table: dict[str, list[str]] = dict(table) if table is not None else {}
        self._repo_path = repo_path

        # Special-token ids by name (PAD=0 ... WORD_BOUNDARY=4) for fast access in encode/decode.
        self._special_ids: dict[str, int] = {
            name: piece_to_id[(_SPECIAL_TAG, name)] for name in _SPECIAL_NAMES
        }
        # First byte-token id; byte value v -> id (byte_offset + v). Stored to decode runs.
        self._byte_offset: int = piece_to_id[(_BYTE_TAG, 0)]

        # The tr_api Tokenizer is built lazily on first analyzer use (so load() needs no repo).
        self._tokenizer = None

        # Per-word memoization of the analyzer path (the bottleneck). The table is checked
        # first, so this only memoizes non-tabled words. Bound to the instance so it is GC'd
        # with the tokenizer; keyed only on the word (merges/table are fixed per instance).
        maxsize = cache_size if cache_size is None else max(int(cache_size), 0)
        self._cache_enabled = maxsize is None or maxsize > 0
        self._analyze_word_cached = (
            functools.lru_cache(maxsize=maxsize)(self._analyze_word_uncached)
            if self._cache_enabled
            else self._analyze_word_uncached
        )

    # --- Vocabulary introspection ----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of ids in the vocabulary (specials + bytes + morpheme pieces)."""
        return len(self.id_to_piece)

    def token_to_id(self, piece: str) -> int | None:
        """Id for a morpheme-piece *string* (specials/bytes use the sentinel keys instead)."""
        return self.piece_to_id.get(piece)

    # --- Word -> pieces (the fast / cached analysis path) ----------------------------------

    def _ensure_tokenizer(self) -> None:
        """Build the ``tr_api.Tokenizer`` lazily (only when an uncached/un-tabled word needs it).

        Keeping this lazy is why :meth:`load` works without ``tr_api`` on ``sys.path``: pure
        table/cache lookups never trigger it. Uses the fast config (suggestion / alternatives /
        typo-repair passes disabled) like the rest of the project.
        """
        if self._tokenizer is not None:
            return
        from .morphology import ensure_tr_api_importable  # noqa: PLC0415  (lazy optional dep)

        ensure_tr_api_importable(self._repo_path)
        from tr_api import Tokenizer, TokenizerConfig  # noqa: PLC0415  (optional, lazy)

        self._tokenizer = Tokenizer(
            TokenizerConfig(
                suggest_on_oov=False,
                include_alternatives=False,
                correct_tail_typos=False,
            )
        )

    def _analyze_word_uncached(self, word: str) -> tuple[str, ...]:
        """Analyze one word with ``tr_api`` and apply the merges, as an immutable tuple.

        Mirrors :meth:`MorphemeBPETokenCounter._encode_word_uncached`: ``split_clitics=False``
        keeps the flat ``{"morphemes": [...]}`` shape, fall back to ``[word]`` on a failed
        parse, and any analyzer error / timeout degrades to ``[word]`` (byte-fallback at id
        time) so a single bad word can never crash encoding. Returned immutable so the
        lru_cache can safely share it.
        """
        self._ensure_tokenizer()
        try:
            analysis = self._tokenizer.tokenize(
                word,
                suggest=False,
                tail_repair=False,
                alternatives=False,
                split_clitics=False,
            )
            morphs = (
                [m["chunk"] for m in analysis.get("morphemes", [])]
                if analysis.get("parsed")
                else []
            )
        except Exception:
            morphs = []
        if not morphs:
            morphs = [word]
        return tuple(self._bpe.encode(morphs))

    def _word_pieces(self, word: str) -> list[str]:
        """Piece strings for one word, fast-table first then the cached analyzer.

        Resolution order (cheapest first):

        1. ``table`` hit  -> the precomputed pieces (no analyzer call at all).
        2. ``len(word) > MAX_WORD_LEN`` -> a single piece (the word itself); byte-fallback at
           id time. Matches ``morph_segment``: long tokens are junk that hang the chart parser.
        3. otherwise the per-word ``lru_cache``-d ``tr_api`` analysis + merges.
        """
        tabled = self._table.get(word)
        if tabled is not None:
            return list(tabled)
        if len(word) > MAX_WORD_LEN:
            return [word]
        return list(self._analyze_word_cached(word))

    # --- Encode ----------------------------------------------------------------------------

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode whitespace-normalized ``text`` to a list of integer ids.

        Each whitespace-split word is prefixed with the WORD_BOUNDARY id, then each of its
        pieces becomes either its vocab id (known morpheme) or, for an OOV morpheme, the
        sequence of its UTF-8 byte-token ids (byte-fallback — so any morpheme is representable
        and the text always round-trips). Optionally wrapped with BOS / EOS.

        Input is assumed already normalized by the upstream corpus pipeline; this method only
        splits on whitespace and does NOT re-normalize (no NFC / casing).
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self._special_ids["BOS"])

        wb = self._special_ids["WORD_BOUNDARY"]
        for word in text.split():
            ids.append(wb)
            for piece in self._word_pieces(word):
                piece_id = self.piece_to_id.get(piece)
                if piece_id is not None:
                    ids.append(piece_id)
                else:
                    ids.extend(self._byte_ids(piece))

        if add_eos:
            ids.append(self._special_ids["EOS"])
        return ids

    def _byte_ids(self, piece: str) -> list[int]:
        """Byte-fallback: a piece's UTF-8 bytes as their byte-token ids (total coverage)."""
        return [self._byte_offset + b for b in piece.encode("utf-8")]

    # --- Decode ----------------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to surface text, the inverse of :meth:`encode`.

        PAD / BOS / EOS / UNK are dropped. WORD_BOUNDARY separates words with a single space.
        Morpheme pieces are concatenated within a word; runs of byte tokens are gathered and
        decoded together as UTF-8 (``errors="replace"``) so a multi-byte character split across
        byte tokens is reassembled correctly. The leading space from the first WORD_BOUNDARY is
        stripped, so ``decode(encode(text)) == text`` for whitespace-normalized ``text``.
        """
        out: list[str] = []
        byte_run: list[int] = []

        def flush_bytes() -> None:
            if byte_run:
                out.append(bytes(byte_run).decode("utf-8", errors="replace"))
                byte_run.clear()

        byte_end = self._byte_offset + _N_BYTES
        drop = {self._special_ids[name] for name in ("PAD", "BOS", "EOS", "UNK")}
        wb = self._special_ids["WORD_BOUNDARY"]

        for tid in ids:
            if self._byte_offset <= tid < byte_end:
                byte_run.append(tid - self._byte_offset)
                continue
            flush_bytes()  # a non-byte token ends any in-progress byte run.
            if tid in drop:
                continue
            if tid == wb:
                out.append(" ")
                continue
            piece = self.id_to_piece[tid] if 0 <= tid < len(self.id_to_piece) else None
            if isinstance(piece, str):
                out.append(piece)
        flush_bytes()  # trailing byte run.

        return "".join(out).lstrip(" ")

    # --- Fast table ------------------------------------------------------------------------

    def set_table(self, table: dict[str, list[str]]) -> None:
        """Install a precomputed ``word -> piece strings`` table (replaces any existing one).

        Stored as piece *strings* (not ids) so a later vocab change does not invalidate the
        table — the hot path is the dict lookup, with id resolution / byte-fallback applied at
        encode time against the current vocab.
        """
        self._table = dict(table)

    def load_table(self, path: str) -> None:
        """Load a ``table.tsv`` (``word\\tpiece1▁piece2...``) into the fast table."""
        self.set_table(_read_table(Path(path)))

    @property
    def has_table(self) -> bool:
        """Whether a fast ``word -> pieces`` table is currently loaded."""
        return bool(self._table)

    def cache_info(self) -> functools.CacheInfo | None:
        """``lru_cache`` stats for the analyzer fallback path (``None`` if caching disabled)."""
        info = getattr(self._analyze_word_cached, "cache_info", None)
        return info() if info is not None else None

    # --- Persistence -----------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Write ``tokenizer.json`` (and ``table.tsv`` if a table is set) to ``directory``.

        ``tokenizer.json`` carries everything needed to reconstruct the vocab and merge engine:
        the format version, vocab size, ``morph_sep``, the ordered ``merges``, the special-token
        ids, the byte-token offset, and the ordered morpheme ``pieces`` (id 261.. in order).
        Specials and bytes are NOT listed in ``pieces`` — their ids are fixed by the layout and
        rebuilt deterministically on :meth:`load`.
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        spec = {
            "version": _FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "morph_sep": self.morph_sep,
            "merges": [list(pair) for pair in self._merges],
            "specials": dict(self._special_ids),
            "byte_offset": self._byte_offset,
            "pieces": self._ordered_morpheme_pieces(),
        }
        (out_dir / _TOKENIZER_FILE).write_text(
            json.dumps(spec, ensure_ascii=False), encoding="utf-8"
        )

        if self._table:
            _write_table(out_dir / _TABLE_FILE, self._table, self.morph_sep)

    def _ordered_morpheme_pieces(self) -> list[str]:
        """The morpheme pieces in id order (id 261.. ), excluding specials and byte tokens."""
        first = self._byte_offset + _N_BYTES
        return [piece for piece in self.id_to_piece[first:] if isinstance(piece, str)]

    @classmethod
    def load(
        cls,
        directory: str,
        *,
        repo_path: str | None = None,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> MorphemeTokenizer:
        """Reconstruct a tokenizer saved by :meth:`save`.

        Rebuilds the :class:`MorphemeBPE` engine from the stored merges, restores the full
        vocabulary deterministically from the layout + stored ``pieces``, and loads ``table.tsv``
        if present. The ``tr_api`` bridge is set up lazily (only on the first uncached /
        un-tabled word), so ``load`` itself never requires ``tr_api`` on ``sys.path``.
        """
        in_dir = Path(directory)
        spec = json.loads((in_dir / _TOKENIZER_FILE).read_text(encoding="utf-8"))

        merges = [tuple(pair) for pair in spec["merges"]]
        bpe = MorphemeBPE(merges)
        piece_to_id, id_to_piece = _build_vocab(spec["pieces"])

        table = None
        table_path = in_dir / _TABLE_FILE
        if table_path.is_file():
            table = _read_table(table_path)

        return cls(
            bpe,
            piece_to_id,
            id_to_piece,
            morph_sep=spec.get("morph_sep", MORPH_SEP),
            merges=merges,
            table=table,
            repo_path=repo_path,
            cache_size=cache_size,
        )

    # --- Factory ---------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        merges_path: str,
        piece_freqs: collections.Counter[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        table: dict[str, list[str]] | None = None,
        repo_path: str | None = None,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> MorphemeTokenizer:
        """Assemble a tokenizer from merges + a piece-frequency Counter (pure, unit-testable).

        The vocabulary is built deterministically:

        * specials at ids 0..4, the 256 byte tokens at 5..260;
        * the remaining id space (261..vocab_size-1) filled with the most-frequent morpheme
          pieces from ``piece_freqs`` (ties broken by piece string, for reproducibility).

        Corpus iteration / ``tr_api`` segmentation are deliberately kept OUT of this method —
        the build script computes ``piece_freqs`` (and optionally the ``word -> pieces`` table)
        and passes them in, so this stays pure and testable with a synthetic Counter.

        Parameters
        ----------
        merges_path:
            Path to a ``morpheme_bpe_<V>.json`` (the merge engine + ``morph_sep``).
        piece_freqs:
            ``piece string -> corpus frequency``; the most common fill the post-byte id space.
        vocab_size:
            Total fixed vocabulary size (must leave room for specials + 256 byte tokens).
        table:
            Optional precomputed ``word -> piece strings`` fast table.
        """
        first_piece_id = len(_SPECIAL_NAMES) + _N_BYTES
        if vocab_size <= first_piece_id:
            raise ValueError(
                f"vocab_size={vocab_size} too small: needs > {first_piece_id} to leave room "
                f"for {len(_SPECIAL_NAMES)} specials + {_N_BYTES} byte tokens."
            )

        data = json.loads(Path(merges_path).read_text(encoding="utf-8"))
        merges = [tuple(m) for m in data["merges"]]
        morph_sep = data.get("morph_sep", MORPH_SEP)
        bpe = MorphemeBPE(merges)

        n_pieces = vocab_size - first_piece_id
        ordered_pieces = _rank_pieces(piece_freqs, n_pieces)
        piece_to_id, id_to_piece = _build_vocab(ordered_pieces)

        return cls(
            bpe,
            piece_to_id,
            id_to_piece,
            morph_sep=morph_sep,
            merges=merges,
            table=table,
            repo_path=repo_path,
            cache_size=cache_size,
        )


# --- Pure vocab / table helpers (no I/O of the tokenizer's own concern) ---------------------


def _rank_pieces(piece_freqs: collections.Counter[str], n: int) -> list[str]:
    """The top-``n`` pieces by frequency (descending), ties broken by piece string.

    Frequency-then-string ordering makes the assigned ids reproducible for a given Counter,
    independent of insertion order.
    """
    ranked = sorted(piece_freqs.items(), key=lambda kv: (-kv[1], kv[0]))
    return [piece for piece, _freq in ranked[:n]]


def _build_vocab(
    ordered_pieces: list[str],
) -> tuple[dict[str | tuple[str, str | int], int], list[str | tuple[str, str | int]]]:
    """Build ``(piece_to_id, id_to_piece)`` for the fixed layout + given ordered morpheme pieces.

    Layout: specials 0..4 (sentinel ``("<special>", name)``), byte tokens 5..260 (sentinel
    ``("<byte>", value)``), then ``ordered_pieces`` from 261 onward (plain ``str``).
    """
    id_to_piece: list[str | tuple[str, str | int]] = []
    piece_to_id: dict[str | tuple[str, str | int], int] = {}

    def append(key: str | tuple[str, str | int]) -> None:
        piece_to_id[key] = len(id_to_piece)
        id_to_piece.append(key)

    for name in _SPECIAL_NAMES:
        append((_SPECIAL_TAG, name))
    for value in range(_N_BYTES):
        append((_BYTE_TAG, value))
    for piece in ordered_pieces:
        # Skip a morpheme piece that would shadow nothing but is a dup (keep first/highest rank).
        if piece not in piece_to_id:
            append(piece)

    return piece_to_id, id_to_piece


def _write_table(path: Path, table: dict[str, list[str]], morph_sep: str) -> None:
    """Serialize a ``word -> pieces`` table as ``word\\tpiece1▁piece2...`` lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for word, pieces in table.items():
            handle.write(f"{word}\t{morph_sep.join(pieces)}\n")


def _read_table(path: Path) -> dict[str, list[str]]:
    """Parse a ``table.tsv`` (``word\\tpiece1▁piece2...``) into a ``word -> pieces`` dict.

    Lines without a tab (or with an empty piece field) are skipped defensively so a malformed
    line can never poison the lookup. The separator is the ``▁`` morpheme separator.
    """
    table: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if "\t" not in line:
                continue
            word, joined = line.split("\t", 1)
            if not joined:
                continue
            table[word] = joined.split(MORPH_SEP)
    return table
