"""Faithful text cleaning + filtering + dedup utilities.

Faithful-only by design: clean_text() removes noise (URLs/@/#/emoji/RT, control & bidi marks,
cosmetic tatweel) and applies NFKC (folds presentation-form glyphs like the lam-alef ligature
to standard letters) -- but KEEPS every linguistic letter and all diacritics. No diacritic
stripping, no alef/ya/ta-marbuta folding. So generation targets stay faithful and dialectal.

Verified on 11.5k real Maghrebi/Egyptian/Darija rows: dialect letters (gaf/che/pe/veh/zhe/
keheh/farsi-ya, U+06AF/0686/067E/06A4/0698/06A9/06CC) survive; the only thing removed that
looks like a "letter" is tatweel (U+0640), which is cosmetic elongation with no meaning.

Per-stage filtering guidance (arabic_ratio is a TARGET-language check, use it accordingly):
  * DETECTION (Stage 1): do NOT filter by arabic_ratio. Code-switching (English words in dialect)
    and Arabizi (Latin-script) are normal, authentic social Arabic and are valid dialect signal --
    dropping them biases the router. Clean + dedup only; keep everything labeled.
  * NORMALIZATION (Stage 2): the MSA *target* should be Arabic, so you can require arabic_ratio on
    the MSA side; be lenient on the dialect source (it may be code-switched).
  * SUMMARIZATION (Stage 3): the Arabic article/summary side should be Arabic -- arabic_ratio is OK.
clean_text() never drops code-switched / Arabizi text; only an arabic_ratio/is_arabic FILTER does.

Pure stdlib (re + unicodedata); ASCII source (code points built via chr()). Loaders do I/O;
this does processing. Example:

    from loaders import load_crosssum
    from clean import clean_text, arabic_ratio, ok_summary_pair, dedup

    def clean_summ(pair="arabic-english", split="train"):
        for r in load_crosssum(pair, split):
            art, summ = clean_text(r.article, social=False), clean_text(r.summary, social=False)
            if arabic_ratio(art) > 0.5 and ok_summary_pair(art, summ):
                yield {"article": art, "summary": summ}

    rows = list(dedup(clean_summ(), key=lambda d: d["article"]))   # then YOU split & save
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable, Iterator


def _r(lo: int, hi: int) -> str:
    """A 'lo-hi' char-class fragment built from code points (keeps source ASCII)."""
    return chr(lo) + "-" + chr(hi)


def _cls(*ranges: tuple[int, int]) -> str:
    return "[" + "".join(_r(lo, hi) for lo, hi in ranges) + "]"


TATWEEL = chr(0x0640)  # kashida / cosmetic elongation -- removed (no linguistic content)
# Arabic + Supplement + Extended-A + Presentation Forms A/B:
_ARABIC_CHAR = re.compile(_cls(
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
))
_URL = re.compile(r"(?:https?://|www\.)\S+")
_MENTION = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG = re.compile("#([\\w" + _r(0x0600, 0x06FF) + "_]+)")
_RT = re.compile(r"\bRT\b\s*:?")
_EMOJI = re.compile(_cls(
    (0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x1F1E6, 0x1F1FF), (0x2190, 0x21FF),
    (0x2300, 0x23FF), (0x2B00, 0x2BFF), (0xFE00, 0xFE0F), (0x200D, 0x200D),
) + "+")
_BIDI = re.compile(_cls((0x200E, 0x200F), (0x202A, 0x202E), (0x2066, 0x2069)))
_CTRL = re.compile(_cls((0x0000, 0x001F), (0x007F, 0x009F)))
_WS = re.compile(r"\s+")
_PUNCT = re.compile("[^\\w" + _r(0x0600, 0x06FF) + "]+")


# -- faithful cleaning --------------------------------------------------------
def clean_text(s: str, *, social: bool = True, hashtags: str = "keep_word") -> str:
    """Faithful clean. Removes URLs/@mentions/RT/emoji (when social), control & bidi marks,
    and cosmetic tatweel; NFKC normalizes presentation-form glyphs to standard letters;
    collapses whitespace. KEEPS all letters and diacritics.

    social:   strip tweet noise (URLs/@/#/emoji/RT). Set False for news/article text.
    hashtags: 'keep_word' (drop '#', keep the word) | 'remove' (drop whole tag) | 'keep'.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if social:
        s = _URL.sub(" ", s)
        s = _MENTION.sub(" ", s)
        s = _RT.sub(" ", s)
        if hashtags == "keep_word":
            s = _HASHTAG.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", s)
        elif hashtags == "remove":
            s = _HASHTAG.sub(" ", s)
        s = _EMOJI.sub(" ", s)
    s = _BIDI.sub("", s)
    s = _CTRL.sub(" ", s)
    s = s.replace(TATWEEL, "")
    return _WS.sub(" ", s).strip()


# -- filtering ----------------------------------------------------------------
def arabic_ratio(s: str) -> float:
    """Fraction of non-space characters that are Arabic-script (0..1).
    NOTE: ~0 for Latin-script Arabizi -- don't filter TUNIZI/NArabizi/code-switched with this."""
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    return sum(bool(_ARABIC_CHAR.match(c)) for c in chars) / len(chars)


def is_arabic(s: str, threshold: float = 0.5) -> bool:
    return arabic_ratio(s) >= threshold


def n_tokens(s: str) -> int:
    return len(s.split())


def length_ok(s: str, min_tokens: int = 1, max_tokens: int = 10_000) -> bool:
    return min_tokens <= n_tokens(s) <= max_tokens


def ok_summary_pair(article: str, summary: str, *, min_art=10, max_art=8000,
                    min_sum=1, max_ratio=0.9) -> bool:
    """Keep a summarization pair iff article/summary are sane and the summary is genuinely
    shorter than the article (drops degenerate summary>=article cases)."""
    a, su = n_tokens(article), n_tokens(summary)
    return (min_art <= a <= max_art) and (su >= min_sum) and (su < a * max_ratio)


# -- dedup --------------------------------------------------------------------
def dedup_key(s: str) -> str:
    """Faithful dedup signature: NFKC + drop tatweel + lowercase + strip punctuation +
    collapse whitespace. (Catches case/punctuation/spacing/ligature variants. It does NOT
    fold diacritics, so diacritic-only variants count as distinct -- consistent with faithful.)"""
    s = unicodedata.normalize("NFKC", s).replace(TATWEEL, "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def dedup(records: Iterable, key: Callable[[object], object] = lambda r: r) -> Iterator:
    """Yield records whose dedup signature is seen for the first time. `key` extracts the
    text from each record (default: the record itself is the string)."""
    seen: set[str] = set()
    for r in records:
        raw = key(r)
        sig = dedup_key(raw if isinstance(raw, str) else str(raw))
        if sig and sig not in seen:
            seen.add(sig)
            yield r
