# -*- coding: utf-8 -*-
"""LB6 - load reduction (mechanism F6, horizon).

One rule: give the model less so the context does not overflow. This engine never speaks; it
returns a transformed view of the history for generation only (the committed history is untouched).
Three deterministic transforms, parameters in A2["LB6"]:

  dedup     a byte-identical tool output is shown once; repeats become one line
  compact   when the total exceeds min_total, old long tool outputs keep only their head
  annotate  {field, note} pairs appended to tool outputs that mention the field

LB7 (give more) is its direct opposite; the two are judged as a pair (canon section 2).
"""

import copy
import hashlib

LB = "LB6"
DEFAULTS = {"keep_recent": 6, "min_len": 800, "head": 400, "annotations": []}
CHARS_PER_TOKEN = 3.5
FOLD_AT = 0.75        # start folding when the view holds this share of the model context


def _text(m):
    c = getattr(m, "content", None)
    return c if isinstance(c, str) else ""


def _with_text(m, text):
    m2 = copy.copy(m)
    m2.content = text
    return m2


def dedup(messages):
    seen, out = {}, list(messages)
    for i, m in enumerate(messages):
        t = _text(m)
        if getattr(m, "role", None) != "tool" or len(t) < 200:
            continue
        h = hashlib.sha1(t.encode("utf-8", "replace")).hexdigest()
        if h in seen:
            out[i] = _with_text(m, "[identical to the tool output at message %d - not repeated]" % seen[h])
        seen.setdefault(h, i)
    return out


def compact(messages, p):
    if sum(len(_text(m)) for m in messages) <= p["min_total"]:
        return messages
    tools = [i for i, m in enumerate(messages) if getattr(m, "role", None) == "tool"]
    out = list(messages)
    for i in tools[:max(0, len(tools) - p["keep_recent"])]:
        t = _text(messages[i])
        if len(t) >= p["min_len"]:
            out[i] = _with_text(messages[i], t[:p["head"]] + "\n[... %d chars folded by the view compactor; "
                                "the full text was read earlier ...]" % (len(t) - p["head"]))
    return out


def annotate(messages, notes):
    out = list(messages)
    for i, m in enumerate(messages):
        t = _text(m)
        hits = [n["note"] for n in notes if getattr(m, "role", None) == "tool" and n.get("field") in t]
        if hits:
            out[i] = _with_text(m, t + "\n[view note] " + " ".join(hits))
    return out


def reduce(a2, messages):
    """Thresholds come from the model context, not from a constant: a constant tuned for a 44k model
    folded at 11% of a 131k one and the model re-read what was folded (old tree, section T-6a)."""
    p = dict(DEFAULTS, min_total=int(int(a2.get("model_context") or 131072) * FOLD_AT * CHARS_PER_TOKEN))
    p.update(a2.get("LB6") or {})
    return annotate(compact(dedup(messages), p), p["annotations"])


def evaluate(turn):
    return []


if __name__ == "__main__":
    class M(object):
        def __init__(self, role, content):
            self.role, self.content = role, content

    big = "x" * 1000
    msgs = [M("tool", big), M("tool", big), M("tool", "y" * 1000), M("assistant", "ok")]
    v = dedup(msgs)
    assert v[1].content.startswith("[identical") and msgs[1].content == big
    v = compact(msgs, dict(DEFAULTS, keep_recent=1, min_total=100, head=10))
    assert v[0].content.startswith("xxxxxxxxxx\n[...") and v[2].content == "y" * 1000
    assert reduce({"model_context": 2000}, msgs)[0].content == big                       # 3000 < 5250: untouched
    assert "[... 600 chars folded" in reduce({"model_context": 100}, [M("tool", big)] * 8)[0].content
    v = annotate([M("tool", "credit_limit: 100")], [{"field": "credit_limit", "note": "cap applies"}])
    assert v[0].content.endswith("[view note] cap applies")
    print("lb6_load self-test OK")
