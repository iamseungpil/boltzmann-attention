# -*- coding: utf-8 -*-
"""Tell the model how many documents its query actually matched.

`KB_search_*` returns the top k and says nothing about the denominator, so a query
matching 195 documents and one matching 4 look identical from inside the trajectory.
The first means "narrow your query"; the second means "you have seen all of them".
Only the second can ground a completeness claim, and the model currently cannot tell
them apart (`TRANSFER_INSTRUCTION_FIDELITY_DESIGN_2026_08_03` §2.4).

The counted predicate is content-word AND: every non-stopword token of the query
present somewhere in the document. Deterministic, no threshold, denominator = the
whole corpus, so the number is a fact the engine issues rather than a choice we made.
Phrase matching was measured first and reports zero on 72% of the run's real queries
(`x48_match_coverage.py`); content-word AND reports zero on 25%.

Domain-general: the corpus location comes from the environment or from
`T2_KB_DOCS_DIR`; nothing here names a domain, a tool, or a document.
"""

import glob
import io
import os
import re
import sys

# Minimal closed stopword list. Fixed here rather than tuned, because a tuned list
# would make the count our choice instead of the engine's fact.
STOP = frozenset(
    "a an the of for to in on at by with and or is are was were be been being do does "
    "did how what when where which who whom this that these those i you we they it my "
    "your our can could should would will shall may might must if then than there here "
    "about from into over under after before during any all each".split()
)

_CACHE = {}
_WARNED = False        # 코퍼스 미해결 경고는 프로세스당 1회(로그 포화 방지)


def norm(s):
    """Same closed normalisation the notice predicate uses: lowercase, alphanumeric, spaces."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s or "").lower())).strip()


def content_words(query):
    return [w for w in norm(query).split() if w not in STOP]


def corpus_dir(orch=None):
    """Where the documents live. Configuration, never a literal in this file."""
    d = os.environ.get("T2_KB_DOCS_DIR")
    if d and os.path.isdir(d):
        return d
    for attr in ("kb_docs_dir", "_kb_docs_dir", "documents_dir"):
        for obj in (orch, getattr(orch, "environment", None), getattr(orch, "env", None)):
            v = getattr(obj, attr, None) if obj is not None else None
            if isinstance(v, str) and os.path.isdir(v):
                return v
    return None


def load_corpus(orch=None):
    d = corpus_dir(orch)
    if not d:
        return None
    if d in _CACHE:
        return _CACHE[d]
    docs = []
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if not os.path.isfile(p):
            continue
        try:
            docs.append(norm(io.open(p, encoding="utf-8", errors="replace").read()))
        except Exception:
            pass
    _CACHE[d] = docs
    return docs


def count(query, docs):
    """Documents containing every content word of the query. None if not computable."""
    cw = content_words(query)
    if not cw or docs is None:
        return None
    return sum(1 for d in docs if all(w in d for w in cw))


def shown_in(text):
    """How many results the tool actually returned, read off its own numbering."""
    return len(re.findall(r"(?m)^\s*(\d+)\.\s", str(text or "")))


def note(query, result_text, orch=None):
    """The one line to attach, or None when it would say nothing."""
    docs = load_corpus(orch)
    n = count(query, docs)
    if n is None:
        # ★2026-08-05: 등재 첫 스모크에서 `matches:` 주석이 궤적에 0건이었고 로그에도 아무것도
        #   없었다 — 코퍼스 미해결과 "붙일 말이 없음"이 **같은 침묵**이었기 때문이다. 전자는 설정
        #   결함이므로 한 번은 말하게 한다(sim당 1회·거동 변화 0).
        global _WARNED
        if docs is None and not _WARNED:
            _WARNED = True
            print("[T2_MATCH_COUNT] corpus 미해결 — T2_KB_DOCS_DIR 미설정이거나 경로가 없다: %r"
                  % (os.environ.get("T2_KB_DOCS_DIR"),), file=sys.stderr, flush=True)
        return None
    shown = shown_in(result_text)
    if n == 0:
        # Never phrase this as "nothing relevant exists" — it is a statement about
        # these words, not about the world.
        return "matches: no document contains all of these words; %d shown by ranking" % shown
    if shown and shown >= n:
        return "matches: %d documents contain all of these words; all %d shown" % (n, n)
    return ("matches: %d documents contain all of these words; %d shown (%d not shown)"
            % (n, shown, max(0, n - shown)))
