"""Production morpheme-BPE tokenizer: integer vocabulary, lossless encode/decode, byte-BPE.

This turns the fertility-only :class:`turkish_corpus.morpheme_bpe.MorphemeBPE` (which only
emits piece *strings*) into a real LLM tokenizer that an embedding table can consume: a
fixed-size integer vocabulary, ``encode(text) -> list[int]`` / ``decode(ids) -> str``, total
coverage via a GPT-2-style **byte-level BPE** fallback, **casing markers** that preserve
case, and an optional precomputed ``word -> pieces`` lookup table so the hot path never
touches the slow ``tr_api`` analyzer.

Why this module changed (held-out validation findings)
------------------------------------------------------
Two deployability flaws were found against real web text:

1. **Raw-byte fallback exploded fertility.** ~1/3 of web words don't parse cleanly into known
   morpheme pieces; the old raw-byte fallback spent ~10 byte tokens per such word (47% of all
   tokens were single bytes), so *real* fertility was 2.056 vs an idealized 1.082. The fix is
   a **byte-level BPE** fallback: rare/unparsed words cost a few sub-word tokens, not ~10 raw
   bytes, while still giving total, exact-bytes coverage.
2. **Round-trip was lossy.** The analyzer lowercases and drops apostrophes, so
   ``"Türkiye'de" -> "türkiyede"`` and casing was lost. The fix is a **casing-marker scheme**
   plus a **fidelity check**: any word that does not cleanly + losslessly morpheme-encode is
   routed to the byte-BPE on its ORIGINAL surface, guaranteeing ``decode(encode(x)) == x``.

Fused word-boundary scheme (SentencePiece / GPT style)
------------------------------------------------------
For a cleanly morpheme-encoded word the boundary is FUSED into the first piece: a word's bare
pieces ``[p0, p1, ...]`` become ``["▁"+p0, p1, ...]``. ``"▁ev"`` / ``"▁gel"`` carry the
boundary, so the common-case token count is exactly the sum of pieces per word (no ``+1`` per
word), preserving morpheme_bpe's fertility advantage. The segmentation table stays
VOCAB-AGNOSTIC (it stores BARE pieces); fusion happens in :meth:`encode`.

Casing markers
--------------
The morpheme analyzer works on the LOWERCASED word, so casing is restored with two special
markers emitted *before* the word's morpheme ids: ``CAP`` (word is Titlecase, e.g. ``Ankara``)
and ``UPPER`` (word is ALLCAPS, e.g. ``TÜRKİYE``). Lowercase words emit no marker. MIXED-case
words (e.g. ``iPhone``) cannot be expressed by a single marker, so they take the byte-BPE path
on the original surface. Casing uses Turkish-aware rules (``ı``/``İ``/``i``/``I``).

Byte-level BPE fallback (total, lossless coverage)
--------------------------------------------------
A word that is MIXED-case, or whose morpheme pieces fail the fidelity check (don't concatenate
back to the lowercased form, or contain an OOV piece — e.g. an apostrophe the analyzer
dropped), is emitted as ``[WORD_BOUNDARY] + byte-BPE(original surface)``. Byte-BPE maps each
UTF-8 byte to a distinct printable unicode char (the GPT-2 ``bytes_to_unicode`` bijection),
applies learned byte merges, and maps the resulting byte-char pieces to ids. Every byte-char
is always in the vocab (256 base tokens), so this NEVER fails and reproduces the exact bytes.
There is no raw single-byte path for OOV morpheme pieces anymore.

Vocabulary layout (fixed-size, default 64000), in id order
----------------------------------------------------------
* ``0`` PAD, ``1`` UNK, ``2`` BOS, ``3`` EOS — standard specials.
* ``4`` WORD_BOUNDARY — precedes a byte-BPE word (the fallback path).
* ``5`` CAP, ``6`` UPPER — casing markers for the following morpheme word.
* ``7..262`` — the 256 byte-char tokens (one per byte value, GPT-2 ``bytes_to_unicode``).
* ``263..(263 + n_byte_merges - 1)`` — byte-BPE merge pieces (byte-char strings), in merge
  rank order.
* ``(263 + n_byte_merges)..(vocab_size-1)`` — morpheme/merge PIECES, frequency-ranked: both
  ``▁``-prefixed word-initial forms and bare word-internal forms.

Specials and byte-chars are keyed internally by sentinel tuples (``("<special>", name)`` /
``("<byte>", value)``) so they can never collide with a real ``str`` piece — even though a
byte-char and a morpheme piece could in principle be the same string, their ids are kept
distinct (byte-chars via the sentinel, byte-BPE/morpheme pieces via their plain ``str`` key).

Round-trip guarantee
--------------------
``decode(encode(text)) == text`` for ANY input whose words are whitespace-separated: every
word is either (a) cleanly morpheme-encoded with an optional casing marker — and only when the
fidelity check confirms the pieces concatenate back to the lowercased form and are all in
vocab — or (b) byte-BPE'd from the exact original surface. Whitespace is normalized to single
spaces between words (this tokenizer splits on whitespace and does not otherwise re-normalize).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .morpheme_bpe import MorphemeBPE
from .normalization import turkish_lower, turkish_title, turkish_upper

if TYPE_CHECKING:
    import collections

__all__ = [
    "MorphemeTokenizer",
    "DEFAULT_VOCAB_SIZE",
    "MORPH_SEP",
    "bytes_to_unicode",
]

# Within-word morpheme separator carried in the merges file / used to serialize the table.
# Matches turkish_corpus.morph_segment.MORPH_SEP and the merges file's "morph_sep" key.
MORPH_SEP = "▁"

# Default fixed vocabulary size (specials + 256 byte-chars + byte-BPE pieces + morpheme pieces).
DEFAULT_VOCAB_SIZE = 64_000

# Words longer than this are junk (URLs / concatenated run-ons) that blow up tr_api's chart
# parser; route them to the byte-BPE fallback. Matches morph_segment.
MAX_WORD_LEN = 70

# Default per-word lru_cache size for the (slow) analyzer path. Covers ~96% of tokens by
# frequency for Turkish text; bounds memory for long-running serving.
DEFAULT_CACHE_SIZE = 1_000_000

# Ordered special-token names; their ids are their index. WORD_BOUNDARY precedes a byte-BPE
# word; CAP/UPPER are casing markers for the following morpheme word.
_SPECIAL_NAMES = ("PAD", "UNK", "BOS", "EOS", "WORD_BOUNDARY", "CAP", "UPPER")

# Number of byte-char tokens (one per possible byte value).
_N_BYTES = 256

# Sentinel piece-key prefixes so specials/byte-chars can live in piece_to_id without colliding
# with a real ``str`` piece key (a tuple is never equal to a str).
_SPECIAL_TAG = "<special>"
_BYTE_TAG = "<byte>"

# On-disk format version for tokenizer.json.
_FORMAT_VERSION = 2

# Filenames written by save() / read by load().
_TOKENIZER_FILE = "tokenizer.json"
_TABLE_FILE = "table.tsv"


@functools.lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte<->unicode bijection: each of 256 byte values -> a distinct printable char.

    Reversible (and injective): printable ASCII-range bytes map to themselves; the remaining
    bytes map to codepoints starting at U+0100, so no byte-char is whitespace or a control
    character (which keeps byte-BPE pieces safe to store in a TSV and to split on). This is the
    standard GPT-2 mapping (Radford et al.), reproduced exactly so byte-BPE merges trained the
    same way are interchangeable.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


def _unicode_to_bytes() -> dict[str, int]:
    """Inverse of :func:`bytes_to_unicode`: byte-char -> byte value."""
    return {char: value for value, char in bytes_to_unicode().items()}


class MorphemeTokenizer:
    """Morpheme-BPE tokenizer with integer vocab, casing markers, and a byte-BPE fallback.

    Wraps the pure :class:`MorphemeBPE` merge engine (the proven, unit-tested merge loop) with
    everything an LLM needs: a fixed-size integer vocabulary, lossless ``encode``/``decode``
    over ids, casing markers, a GPT-2-style byte-level BPE fallback for OOV/unparseable words
    (total + exact coverage), optional BOS/EOS, and an optional precomputed ``word -> pieces``
    table so the hot path is a dict lookup instead of the slow ``tr_api`` analysis.

    Construct via :meth:`build` (from frequencies + byte merges) or :meth:`load` (from a saved
    directory) rather than directly; ``__init__`` takes already-assembled vocab structures.

    Parameters
    ----------
    bpe:
        The morpheme merge engine (reconstructed from the morpheme merges list).
    byte_bpe:
        The byte-level merge engine (a second :class:`MorphemeBPE` over byte-char symbols; its
        ``encode`` is symbol-agnostic so the same greedy lowest-rank logic applies to bytes).
    piece_to_id / id_to_piece:
        The vocabulary (see the module layout). Sentinel tuples for specials/byte-chars, plain
        ``str`` for byte-BPE and morpheme pieces.
    morph_sep:
        Within-word morpheme separator from the merges file.
    merges / byte_merges:
        The ordered morpheme / byte merge pairs (kept so :meth:`save` can serialize them).
    table:
        Optional precomputed ``word -> piece strings`` map (BARE pieces, on the LOWERCASED
        form); a tabled word is a dict lookup with no analyzer call.
    repo_path:
        Optional path to the ``turkish-tokenizer`` clone for the lazy ``tr_api`` bridge.
    cache_size:
        Per-word ``lru_cache`` size for the analyzer fallback path.
    """

    def __init__(
        self,
        bpe: MorphemeBPE,
        byte_bpe: MorphemeBPE,
        piece_to_id: dict[str | tuple[str, str | int], int],
        id_to_piece: list[str | tuple[str, str | int]],
        *,
        morph_sep: str = MORPH_SEP,
        merges: list[tuple[str, str]] | None = None,
        byte_merges: list[tuple[str, str]] | None = None,
        table: dict[str, list[str]] | None = None,
        repo_path: str | None = None,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._bpe = bpe
        self._byte_bpe = byte_bpe
        self.piece_to_id = piece_to_id
        self.id_to_piece = id_to_piece
        self.morph_sep = morph_sep
        self._merges = list(merges) if merges is not None else []
        self._byte_merges = list(byte_merges) if byte_merges is not None else []
        self._table: dict[str, list[str]] = dict(table) if table is not None else {}
        self._repo_path = repo_path

        # Special-token ids by name for fast access in encode/decode.
        self._special_ids: dict[str, int] = {
            name: piece_to_id[(_SPECIAL_TAG, name)] for name in _SPECIAL_NAMES
        }
        # First byte-char-token id; byte value v -> id (byte_offset + v). Stored to decode runs.
        self._byte_offset: int = piece_to_id[(_BYTE_TAG, 0)]

        # The GPT-2 byte<->unicode maps (byte-char -> id is via piece_to_id sentinel keys).
        self._byte_to_unicode = bytes_to_unicode()
        self._unicode_to_byte = _unicode_to_bytes()

        # The tr_api Tokenizer is built lazily on first analyzer use (so load() needs no repo).
        self._tokenizer = None

        # Per-word memoization of the analyzer path (the bottleneck). The table is checked
        # first, so this only memoizes non-tabled words.
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
        """Total number of ids in the vocabulary."""
        return len(self.id_to_piece)

    def token_to_id(self, piece: str) -> int | None:
        """Id for a piece *string* (specials/byte-chars use the sentinel keys instead)."""
        return self.piece_to_id.get(piece)

    # --- Word -> pieces (the fast / cached analysis path) ----------------------------------

    def _ensure_tokenizer(self) -> None:
        """Build the ``tr_api.Tokenizer`` lazily (only when an uncached/un-tabled word needs it)."""
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
        """Analyze one (lowercased) word with ``tr_api`` and apply the merges, as a tuple.

        Mirrors :meth:`MorphemeBPETokenCounter._encode_word_uncached`: ``split_clitics=False``
        keeps the flat ``{"morphemes": [...]}`` shape, falls back to ``[word]`` on a failed
        parse, and any analyzer error degrades to ``[word]`` so a bad word can never crash
        encoding. Returned immutable so the lru_cache can safely share it.
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

    def _word_pieces(self, lowered: str) -> list[str]:
        """BARE piece strings for one LOWERCASED word, fast-table first then cached analyzer.

        Resolution order (cheapest first): table hit -> the precomputed pieces; otherwise the
        per-word ``lru_cache``-d ``tr_api`` analysis + merges. (Over-long words never reach
        here — :meth:`encode` routes them straight to the byte-BPE fallback.)
        """
        tabled = self._table.get(lowered)
        if tabled is not None:
            return list(tabled)
        return list(self._analyze_word_cached(lowered))

    # --- Casing ----------------------------------------------------------------------------

    def _casing_marker(self, surface: str, lowered: str) -> int | None | bool:
        """Classify ``surface`` casing against its lowercase form.

        Returns the CAP / UPPER special id (or ``None`` for lowercase) when the word can be
        reconstructed from ``lowered`` + a single marker, or ``False`` for MIXED case (no
        single marker recovers it -> caller must use the byte-BPE fallback).
        """
        if surface == lowered:
            return None
        if surface == turkish_title(lowered):
            return self._special_ids["CAP"]
        if surface == turkish_upper(lowered) and surface != lowered:
            return self._special_ids["UPPER"]
        return False

    # --- Encode ----------------------------------------------------------------------------

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode ``text`` to integer ids, guaranteeing ``decode(encode(text)) == text``.

        Splits on whitespace. For each surface word ``w``:

        1. Lowercase ``lw = turkish_lower(w)`` and classify casing (lowercase / CAP / UPPER /
           MIXED). MIXED, or an over-long word, goes straight to step 3.
        2. Compute the BARE morpheme ``pieces`` of ``lw`` (table or cached analyzer) and fuse
           the boundary into the first piece. FIDELITY CHECK: only if ``"".join(pieces) == lw``
           AND every fused piece is in the morpheme vocab, emit ``[marker?] + [morph ids]``.
        3. Otherwise emit ``[WORD_BOUNDARY] + byte-BPE(w)`` on the ORIGINAL surface ``w`` —
           exact case + punctuation preserved, total coverage, never fails.

        So the common case keeps morpheme_bpe's fertility (no per-word boundary token), while
        any word the morpheme path can't represent losslessly falls back to a few byte-BPE
        tokens. Optionally wrapped with BOS / EOS.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self._special_ids["BOS"])

        for word in text.split():
            ids.extend(self._encode_word(word))

        if add_eos:
            ids.append(self._special_ids["EOS"])
        return ids

    def _encode_word(self, word: str) -> list[int]:
        """Ids for a single surface word (morpheme path with casing marker, else byte-BPE)."""
        lowered = turkish_lower(word)
        marker = self._casing_marker(word, lowered)

        # MIXED case, or junk-length: no clean morpheme path -> byte-BPE the original surface.
        if marker is not False and len(word) <= MAX_WORD_LEN:
            pieces = self._word_pieces(lowered)
            morph_ids = self._try_morpheme_ids(pieces, lowered)
            if morph_ids is not None:
                prefix = [] if marker is None else [marker]
                return prefix + morph_ids

        # Fallback: byte-BPE on the ORIGINAL surface (preserves exact case + punctuation).
        return [self._special_ids["WORD_BOUNDARY"], *self._byte_bpe_encode(word)]

    def _try_morpheme_ids(self, pieces: list[str], lowered: str) -> list[int] | None:
        """Morpheme ids for ``pieces`` if they pass the fidelity check, else ``None``.

        Fidelity: the pieces must concatenate back to the lowercased form (no dropped
        characters — e.g. an apostrophe) AND every fused piece must be in the morpheme vocab
        (no OOV). Either failure returns ``None`` so the caller uses the byte-BPE fallback,
        which is what guarantees losslessness.
        """
        if not pieces or "".join(pieces) != lowered:
            return None
        fused = [self.morph_sep + pieces[0], *pieces[1:]]
        ids: list[int] = []
        for piece in fused:
            piece_id = self.piece_to_id.get(piece)
            if piece_id is None:
                return None
            ids.append(piece_id)
        return ids

    def _byte_bpe_encode(self, surface: str) -> list[int]:
        """Byte-level BPE of a surface string: UTF-8 -> byte-chars -> merges -> ids.

        Every byte-char is always in the vocab (256 base byte tokens), so this NEVER fails and
        preserves the exact bytes. Pieces are byte-char strings; word-initial pieces are NOT
        ``▁``-fused (the WORD_BOUNDARY token already marks the boundary for byte-BPE words).
        """
        byte_chars = [self._byte_to_unicode[b] for b in surface.encode("utf-8")]
        ids: list[int] = []
        for piece in self._byte_bpe.encode(byte_chars):
            piece_id = self.piece_to_id.get(piece)
            if piece_id is not None:
                ids.append(piece_id)
            else:
                # Merge piece not in vocab -> emit its constituent byte-char ids (always present).
                for char in piece:
                    ids.append(self._byte_offset + self._unicode_to_byte[char])
        return ids

    # --- Decode ----------------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to surface text, the exact inverse of :meth:`encode`.

        PAD / BOS / EOS / UNK are dropped. ``CAP`` / ``UPPER`` set a pending case applied to the
        NEXT morpheme word when it flushes. A ``MORPH_SEP``-prefixed piece starts a new morpheme
        word (``▁`` stripped; pending case applied + cleared on flush); bare morpheme pieces
        concatenate onto the current morpheme word. ``WORD_BOUNDARY`` starts a new byte-BPE
        word; following byte-char / byte-BPE pieces accumulate as byte-chars, then map back to
        bytes and decode UTF-8 (``errors="replace"``). Words join with single spaces.
        """
        words: list[str] = []
        morph_word: list[str] | None = None  # accumulating morpheme pieces, or None
        byte_chars: list[str] | None = None  # accumulating byte-chars, or None
        pending_case: str | None = None  # "CAP" / "UPPER" for the next morpheme word

        byte_end = self._byte_offset + _N_BYTES
        drop = {self._special_ids[name] for name in ("PAD", "BOS", "EOS", "UNK")}
        wb = self._special_ids["WORD_BOUNDARY"]
        cap = self._special_ids["CAP"]
        upper = self._special_ids["UPPER"]

        def flush() -> None:
            """Emit the in-progress word (if any), applying + clearing any pending case.

            ``pending_case`` is cleared ONLY when a morpheme word is actually emitted, so a
            marker set just before a ``▁`` piece survives the flush that opens the new word.
            """
            nonlocal morph_word, byte_chars, pending_case
            if morph_word is not None:
                word = "".join(morph_word)
                if pending_case == "CAP":
                    word = turkish_title(word)
                elif pending_case == "UPPER":
                    word = turkish_upper(word)
                words.append(word)
                morph_word = None
                pending_case = None
            elif byte_chars is not None:
                raw = bytes(self._unicode_to_byte[c] for c in byte_chars)
                words.append(raw.decode("utf-8", errors="replace"))
                byte_chars = None

        for tid in ids:
            if tid in drop:
                continue
            if tid == cap:
                flush()
                pending_case = "CAP"
                continue
            if tid == upper:
                flush()
                pending_case = "UPPER"
                continue
            if tid == wb:
                flush()
                pending_case = None  # byte-BPE words carry their own case; drop any stray marker
                byte_chars = []  # a byte-BPE word begins
                continue

            # A byte-char token: part of the current byte-BPE word.
            if self._byte_offset <= tid < byte_end:
                if byte_chars is None:  # defensive: byte-char without a WORD_BOUNDARY
                    flush()
                    byte_chars = []
                byte_chars.append(self._byte_to_unicode[tid - self._byte_offset])
                continue

            piece = self.id_to_piece[tid] if 0 <= tid < len(self.id_to_piece) else None
            if not isinstance(piece, str):
                continue

            # A ``▁``-prefixed piece is ALWAYS a word-initial morpheme piece: byte-BPE pieces
            # are GPT-2 byte-chars and can never contain ``▁`` (U+2581), so this reliably ends
            # any byte-BPE word in progress and opens a new morpheme word.
            if piece.startswith(self.morph_sep):
                flush()
                morph_word = [piece[len(self.morph_sep) :]]
                continue

            if byte_chars is not None:
                # Inside a byte-BPE word: a bare str piece is a byte-BPE merge piece (byte-chars).
                byte_chars.extend(piece)
                continue

            # A bare morpheme piece continues the current morpheme word.
            if morph_word is None:
                morph_word = []
            morph_word.append(piece)

        flush()
        return " ".join(words)

    # --- Fast table ------------------------------------------------------------------------

    def set_table(self, table: dict[str, list[str]]) -> None:
        """Install a precomputed ``word -> piece strings`` table (replaces any existing one).

        Keyed by the LOWERCASED word and stored as BARE piece *strings* so a vocab change does
        not invalidate it; id resolution / fidelity check / fallback are applied at encode time.
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

        ``tokenizer.json`` carries everything needed to reconstruct both merge engines and the
        vocab: format version, vocab size, ``morph_sep``, the ordered morpheme ``merges`` and
        ``byte_bpe_merges``, the special ids, the byte-char offset, ``n_byte_merges``, the
        byte-BPE pieces, and the ordered morpheme ``pieces``. Specials and byte-chars are NOT
        listed (their ids are fixed by the layout and rebuilt deterministically on :meth:`load`).
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        byte_pieces, morph_pieces = self._ordered_pieces()
        spec = {
            "version": _FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "morph_sep": self.morph_sep,
            "merges": [list(pair) for pair in self._merges],
            "byte_bpe_merges": [list(pair) for pair in self._byte_merges],
            "specials": dict(self._special_ids),
            "byte_offset": self._byte_offset,
            "n_byte_merges": len(byte_pieces),
            "byte_pieces": byte_pieces,
            "pieces": morph_pieces,
        }
        (out_dir / _TOKENIZER_FILE).write_text(
            json.dumps(spec, ensure_ascii=False), encoding="utf-8"
        )

        if self._table:
            _write_table(out_dir / _TABLE_FILE, self._table, self.morph_sep)

    def _ordered_pieces(self) -> tuple[list[str], list[str]]:
        """``(byte_bpe_pieces, morpheme_pieces)`` in id order (after specials + byte-chars).

        The byte-BPE section runs for ``len(byte_merges)`` ids; the remainder are morpheme
        pieces. Both are plain ``str`` (the byte-char base tokens are sentinel-keyed, not here).
        """
        first = self._byte_offset + _N_BYTES
        tail = [p for p in self.id_to_piece[first:] if isinstance(p, str)]
        n_byte = len(self._byte_merges)
        return tail[:n_byte], tail[n_byte:]

    @classmethod
    def load(
        cls,
        directory: str,
        *,
        repo_path: str | None = None,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> MorphemeTokenizer:
        """Reconstruct a tokenizer saved by :meth:`save`.

        Rebuilds both the morpheme and byte :class:`MorphemeBPE` engines from the stored merges,
        restores the full vocabulary deterministically from the layout + stored byte-BPE and
        morpheme pieces, and loads ``table.tsv`` if present. ``tr_api`` is set up lazily.
        """
        in_dir = Path(directory)
        spec = json.loads((in_dir / _TOKENIZER_FILE).read_text(encoding="utf-8"))

        merges = [tuple(pair) for pair in spec["merges"]]
        byte_merges = [tuple(pair) for pair in spec.get("byte_bpe_merges", [])]
        bpe = MorphemeBPE(merges)
        byte_bpe = MorphemeBPE(byte_merges)
        piece_to_id, id_to_piece = _build_vocab(
            spec.get("byte_pieces", []), spec["pieces"]
        )

        table = None
        table_path = in_dir / _TABLE_FILE
        if table_path.is_file():
            table = _read_table(table_path)

        return cls(
            bpe,
            byte_bpe,
            piece_to_id,
            id_to_piece,
            morph_sep=spec.get("morph_sep", MORPH_SEP),
            merges=merges,
            byte_merges=byte_merges,
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
        byte_merges: list[tuple[str, str]],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        table: dict[str, list[str]] | None = None,
        repo_path: str | None = None,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> MorphemeTokenizer:
        """Assemble a tokenizer from morpheme merges + a piece Counter + byte-BPE merges.

        The vocabulary is built deterministically in the layout order: specials, the 256
        byte-char tokens, the ``byte_merges`` pieces (byte-char strings), then the most-frequent
        morpheme pieces filling the remaining id space up to ``vocab_size``.

        Corpus iteration / ``tr_api`` segmentation / byte-merge *training* are kept OUT of this
        method — the build script computes ``piece_freqs``, ``byte_merges`` (and optionally the
        table) and passes them in, so this stays pure and testable.

        Parameters
        ----------
        merges_path:
            Path to a ``morpheme_bpe_<V>.json`` (the morpheme merge engine + ``morph_sep``).
        piece_freqs:
            ``piece string -> corpus frequency``; the most common fill the post-byte-BPE space.
        byte_merges:
            Learned byte-level BPE merges (ordered ``(a, b)`` byte-char pairs). Reconstructs the
            byte merge engine; each merge also gets a vocab id (one piece per merge).
        vocab_size:
            Total fixed vocabulary size. Must leave room for at least one morpheme piece after
            the specials + 256 byte-chars + ``len(byte_merges)`` byte-BPE pieces, else ValueError.
        table:
            Optional precomputed ``word -> piece strings`` fast table (lowercased keys).
        """
        n_byte_merges = len(byte_merges)
        first_morph_id = len(_SPECIAL_NAMES) + _N_BYTES + n_byte_merges
        if vocab_size <= first_morph_id:
            raise ValueError(
                f"vocab_size={vocab_size} too small: needs > {first_morph_id} to leave room "
                f"for {len(_SPECIAL_NAMES)} specials + {_N_BYTES} byte-chars + "
                f"{n_byte_merges} byte-BPE pieces, with at least one morpheme piece."
            )

        data = json.loads(Path(merges_path).read_text(encoding="utf-8"))
        merges = [tuple(m) for m in data["merges"]]
        morph_sep = data.get("morph_sep", MORPH_SEP)
        bpe = MorphemeBPE(merges)
        byte_bpe = MorphemeBPE(list(byte_merges))

        # Byte-BPE pieces are the merged byte-char strings, in merge-rank order.
        byte_pieces = [a + b for a, b in byte_merges]

        n_morph = vocab_size - first_morph_id
        morph_pieces = _rank_pieces(piece_freqs, n_morph)
        piece_to_id, id_to_piece = _build_vocab(byte_pieces, morph_pieces)

        return cls(
            bpe,
            byte_bpe,
            piece_to_id,
            id_to_piece,
            morph_sep=morph_sep,
            merges=merges,
            byte_merges=list(byte_merges),
            table=table,
            repo_path=repo_path,
            cache_size=cache_size,
        )


# --- Pure vocab / table helpers ------------------------------------------------------------


def _rank_pieces(piece_freqs: collections.Counter[str], n: int) -> list[str]:
    """The top-``n`` pieces by frequency (descending), ties broken by piece string."""
    ranked = sorted(piece_freqs.items(), key=lambda kv: (-kv[1], kv[0]))
    return [piece for piece, _freq in ranked[:n]]


def _build_vocab(
    byte_pieces: list[str],
    morph_pieces: list[str],
) -> tuple[dict[str | tuple[str, str | int], int], list[str | tuple[str, str | int]]]:
    """Build ``(piece_to_id, id_to_piece)`` for the fixed layout.

    Layout: specials (sentinel ``("<special>", name)``), 256 byte-char tokens (sentinel
    ``("<byte>", value)``), then ``byte_pieces`` (byte-BPE merge strings), then ``morph_pieces``
    (morpheme strings) — the last two as plain ``str`` keys. A duplicate ``str`` piece is kept
    at its first (highest-rank) id.
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
    for piece in byte_pieces:
        if piece not in piece_to_id:
            append(piece)
    for piece in morph_pieces:
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
