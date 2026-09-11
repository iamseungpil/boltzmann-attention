# -*- coding: utf-8 -*-
"""LB coordinator - order, conflict resolution and the single exit for engines LB1..LB7.

Canon: research_base/NEW_RESEARCH_BASE.md section 2 (LB1..LB7), section 5 (defects 048, 049).
This directory is a new code base, independent of scripts/distill/tau2 (user instruction 2026-09-08).

Four rules, nothing else:
  1. Engines emit Findings, never text to the model. The only exit is say().
  2. One order per target; facts are unioned (losers are replaced, not deleted). The winner is
     the engine earliest in LB_ORDER, then the stronger evidence grade. Authored prose (grade 5)
     never beats a result computed from state (grade 1-2) - that is what removes defect 049.
  3. A deny is per tool call and always carries its body (fail-closed). Advice is one order per
     turn and only when the window is open (the model is resigning, acting or instructing).
  4. Suppressing another engine needs a warrant declared in A2 (suppression_authority).

Flags: T2_LB1..T2_LB7 (default on). Harness paths: LB_SIDECAR, LB_DOCS_DIR. No other flags exist here.
Conflicts are never hidden: every target with two speakers writes one [LB_CONFLICT] line to stderr
and to the sidecar; lb_report.py folds them into a task x LB table.
"""

import collections
import hashlib
import io
import json
import os
import sys

DENY, SURFACE, PIN = "deny", "surface", "pin"
GRADES = {"execution_ledger": 1, "policy_verbatim": 2, "env_output": 3,
          "retrieved_prose": 4, "authored_prose": 5}
LB_ORDER = ["LB3", "LB1", "LB2", "LB4", "LB5", "LB7"]   # LB6 transforms the view; it never speaks
_RANK = {lb: i for i, lb in enumerate(LB_ORDER)}
_MODULES = {"LB1": "lb1_requirements", "LB2": "lb2_decision", "LB3": "lb3_citation",
            "LB4": "lb4_coverage", "LB5": "lb5_resignation", "LB6": "lb6_load", "LB7": "lb7_material"}
FAILSAFE_DENY = "Error: [POLICY GATE] this call was denied; reason unavailable - do not retry the same call"
ADVICE_BUDGET = 2          # how often one rule may advise in one simulation
DENY_BUDGET = 6            # how often one rule may deny in one simulation; after that the call passes.
                           # A rule that keeps losing to the same model has lost; the simulation must stay
                           # alive (the old tree had twenty-five per-lever caps saying this, one each).


def enabled(lb):
    return os.environ.get("T2_" + lb, "1") != "0"


def engine(lb):
    import importlib
    return importlib.import_module(_MODULES[lb])


# ---- small pure helpers shared by every engine (no regular expressions anywhere) ----------------
def fam(name):
    """Family name: the numeric suffix stripped ('close_card_7834' -> 'close_card')."""
    s = str(name or "")
    i = s.rfind("_")
    return s[:i] if i > 0 and s[i + 1:].isdigit() else s


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            v = json.loads(value)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def records_in(text, key=None):
    """Every JSON object inside a tool output, key-filtered if given. Objects are found by brace
    matching (a tool output often wraps its JSON in prose), then parsed with json.loads only."""
    out, s, i = [], str(text or ""), 0

    def walk(v):
        if isinstance(v, dict):
            if key is None or v.get(key) is not None:
                out.append(v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    while True:
        i = s.find("{", i)
        if i < 0:
            break
        depth, j, quoted = 0, i, False
        while j < len(s):
            ch = s[j]
            if ch == '"' and s[j - 1] != "\\":
                quoted = not quoted
            elif not quoted and ch == "{":
                depth += 1
            elif not quoted and ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        try:
            walk(json.loads(s[i:j + 1]))
            i = j + 1
        except Exception:
            i += 1
    return out


def fill(template, **slots):
    """Fill {slot} placeholders; unfilled lowercase slots are removed."""
    out = str(template or "")
    for k, v in slots.items():
        out = out.replace("{%s}" % k, str(v))
    while True:
        i = out.find("{")
        j = out.find("}", i + 1)
        if i < 0 or j < 0:
            break
        body = out[i + 1:j]
        if body and all(c.islower() or c == "_" for c in body):
            out = out[:i] + out[j + 1:]
        else:
            out = out[:i] + "\x00" + out[i + 1:]
    return out.replace("\x00", "{").strip()


class Finding(object):
    """One verdict of one engine. Structure, not text - the exit builds the sentence."""
    __slots__ = ("lb", "primitive", "target", "call", "order", "facts", "grade", "source",
                 "pin", "force_call")

    def __init__(self, lb, primitive, target=None, call=None, order=None, facts=(), grade=5,
                 source="", pin=None, force_call=False):
        self.lb, self.primitive, self.target, self.call, self.order = lb, primitive, target, call, order
        self.facts = [f for f in facts if f]
        self.grade, self.source = int(grade), source
        self.pin, self.force_call = pin, force_call

    def key(self):
        return ("call", id(self.call)) if self.call is not None else ("target", self.target)


class Turn(object):
    """Read-only state of one turn. Engines see nothing else.

    user_text holds customer messages only - tool output quoting a policy sentence is not the
    customer asking for it (defect 048).
    """

    def __init__(self, a2, messages, am, executed=None, unlocked=(), visible_tools=(),
                 registry=None, corpus=None, extras=None, attempted=None, sim="-", ran=()):
        self.a2 = a2 or {}
        # identity, not capability: an engine can label a record with it and do nothing else. LB4's
        # claims audit is 71% of what our layer does and its rows carried no simulation at all.
        self.sim = sim
        self.messages = list(messages or [])
        self.am = am
        self.calls = list(getattr(am, "tool_calls", None) or [])
        # executed = ran AND the answer was yes; attempted = ran, whatever the answer was. A claim of
        # having done something is backed by attempted (the check ran; "no" is an answer, not an
        # absence); permission to take the next step requires executed.
        self.executed = collections.Counter(executed or {})
        self.attempted = collections.Counter(attempted if attempted is not None else self.executed)
        # the same list as executed, kept in order and with each call's arguments, so a procedure
        # that names its subject can count only the calls that carried that subject
        self.ran = list(ran or ())
        self.unlocked = set(unlocked or ())
        self.visible_tools = set(visible_tools or ())
        self.registry = registry or {"agent": set(), "user": set(), "user_all": set()}
        self.corpus = corpus or {}
        self.extras = extras or {}
        self.user_text = self._text("user")
        self.tool_text = self._text("tool")
        self.am_text = str(getattr(am, "content", "") or "")
        # what we have already told the customer. A required disclosure is satisfied by having said
        # it on an earlier turn, not by saying it in the same breath as the action it gates.
        self.said = self._text("assistant")
        d = self.a2.get("dispatch") or {}
        self.name_args = dict(d.get("name_args") or {})
        self.exec_wrappers = {d.get("agent_call"), d.get("user_call")}
        self.payload_key = d.get("payload_key") or "arguments"

    def _text(self, role):
        return "\n".join(str(getattr(m, "content", "") or "") for m in self.messages
                         if getattr(m, "role", None) == role
                         and isinstance(getattr(m, "content", None), str)).lower()

    def name_of(self, call):
        """The name the environment executes: a dispatcher carries it in its declared argument."""
        nm = str(getattr(call, "name", "") or "")
        key = self.name_args.get(nm) if nm in self.exec_wrappers else None
        return str(as_dict(call.arguments).get(key) or nm) if key else nm

    def named(self, call):
        """The name any wrapper (unlock, give, call) carries in its name argument."""
        key = self.name_args.get(str(getattr(call, "name", "") or ""))
        return str(as_dict(call.arguments).get(key) or "") if key else ""

    def args_of(self, call):
        raw = as_dict(call.arguments)
        inner = raw.get(self.payload_key)
        return as_dict(inner) if inner is not None and as_dict(inner) else raw

    def executed_fams(self):
        return {fam(n) for n in self.executed}

    def attempted_fams(self):
        return {fam(n) for n in self.attempted}

    def tool_outputs(self):
        return [str(getattr(m, "content", "") or "") for m in self.messages
                if getattr(m, "role", None) == "tool"]

    def resigning(self):
        return not self.calls and bool(self.am_text.strip())


class Decision(object):
    def __init__(self):
        self.denies, self.advice, self.advice_rules, self.pins = {}, [], [], []
        self.force_call, self.conflicts, self.trace, self.won = False, [], [], {}

    def empty(self):
        return not (self.denies or self.advice or self.pins)


def evaluate(turn):
    out = collections.OrderedDict()
    for lb in ["LB7"] + [x for x in LB_ORDER if x != "LB7"]:
        try:
            out[lb] = list(engine(lb).evaluate(turn)) if enabled(lb) else []
        except Exception as e:
            print("[LB] %s failed (empty): %r" % (lb, e), file=sys.stderr, flush=True)
            out[lb] = []
    return out


def resolve(findings, turn=None):
    """Group by target, pick one winner per group, merge the rest into facts."""
    if isinstance(findings, dict):
        findings = [f for fs in findings.values() for f in fs]
    d = Decision()
    groups = collections.OrderedDict()
    for f in findings:
        groups.setdefault(f.key(), []).append(f)
        if f.primitive == PIN and f.pin:
            d.pins.append(tuple(f.pin))
        d.force_call = d.force_call or f.force_call
    for g in groups.values():
        g.sort(key=lambda f: (_RANK.get(f.lb, 99), f.grade))
    order_taken = False
    for key, g in sorted(groups.items(), key=lambda kv: (_RANK.get(kv[1][0].lb, 99), kv[1][0].grade)):
        winner = g[0]
        d.won[key] = winner
        if len(g) > 1:
            d.conflicts.append({"target": winner.target or getattr(winner.call, "name", None),
                                "winner": (winner.lb, winner.source, winner.grade),
                                "losers": [(f.lb, f.source, f.grade) for f in g[1:]]})
        if key[0] == "call":
            d.denies[key[1]] = _merge(turn, winner, g) or FAILSAFE_DENY
        else:
            text = _merge(turn, winner, g, order_ok=not (winner.order and order_taken))
            order_taken = order_taken or bool(winner.order)
            if text:
                d.advice.append(text)
                d.advice_rules.append(winner.source.split(":")[0] or winner.lb)
    return d


def _merge(turn, winner, group, order_ok=True):
    facts = []
    for f in group:
        for x in f.facts:
            if x not in facts:
                facts.append(x)
        if f is not winner and f.order and f.order != winner.order and f.order not in facts:
            facts.append("(also noted) " + f.order)
    head = (winner.order or "") if order_ok else ""
    return (head + " " + " ".join(x for x in facts if x != head)).strip()


def window_opened(am, targets, names_of):
    """The model is leaving: resigning (text, no calls), acting on a target, or instructing one.

    A discoverable tool is reached through a dispatcher, so the call's own name is
    unlock_discoverable_agent_tool or call_discoverable_agent_tool and never the target itself.
    Comparing only that name kept this window shut for every discoverable write in the domain: LB7
    carried its policy sentence to the turn the model reached for the tool, resolve built the
    advice, a conflict line was written for it, and the exit dropped it - 085 recorded sixteen of
    those lines and not one write-rule advice, in four simulations out of four. So the names a call
    is judged by include the one it names.
    """
    calls = list(getattr(am, "tool_calls", None) or [])
    text = str(getattr(am, "content", "") or "")
    if not calls and text.strip():
        return True
    for c in calls:
        got = names_of(c)
        for n in ([got] if isinstance(got, str) else list(got or [])):
            if n and n in targets:
                return True
    return any(t and t in text for t in targets)


def admit(owner, tag, text):
    """Same input, same sentence, said once - repetition is keyed on the text, not on a count."""
    seen = owner.__dict__.setdefault("_lb_said", set())
    key = (tag, " ".join(str(text).split()))
    if key in seen:
        return False
    seen.add(key)
    return True


def sim_id(owner):
    """The simulation's own id, so a firing can be joined to the simulation that recorded it.
    id(owner) changes whenever the agent object is rebuilt (a retry), which split one simulation's
    rows across several ids - 11 of the 43 tasks completed on 09-08 carried more ids than they had
    simulations, and none of the ids joined to results.json. The orchestrator owns the id that ends
    up there (tau2 orchestrator.py:124 -> :807); lb_runtime hands it to the agent as _lb_orch."""
    if owner is None:
        return "-"
    sid = getattr(getattr(owner, "_lb_orch", None), "simulation_id", None)
    return sid or owner.__dict__.setdefault("_lb_sim", "%06x" % (id(owner) & 0xFFFFFF))


def sidecar(kind, text, turn=None, **meta):
    path = os.environ.get("LB_SIDECAR") or os.environ.get("T2_FB_SIDECAR")
    if not path:
        return
    row = {"kind": kind, "turn": len(turn.messages) if turn else 0, "len": len(text),
           "sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], "text": text[:4000]}
    row.update({k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))})
    try:
        with io.open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def say(turn, findings, owner=None):
    """The single exit: window -> fingerprint -> budget -> conflict record."""
    d = resolve(findings, turn)
    targets = {f.target for f in d.won.values() if f.target}
    if d.advice and not window_opened(turn.am, targets, lambda c: (
            turn.name_of(c), fam(turn.name_of(c)), fam(turn.named(c) or ""))):
        d.trace.append(("window", "closed", len(d.advice)))
        d.advice, d.advice_rules = [], []
    if owner is not None:
        for k, text in d.denies.items():
            if not admit(owner, "deny", text):
                d.trace.append(("repeat", "deny kept", d.won[("call", k)].source))
        denied = owner.__dict__.setdefault("_lb_deny_fired", collections.Counter())
        for k in list(d.denies):
            rule = d.won[("call", k)].source.split(":")[0] or d.won[("call", k)].lb
            denied[rule] += 1
            if denied[rule] > DENY_BUDGET:
                d.trace.append(("budget", "deny released, rule exhausted in this simulation", rule))
                sidecar("lb-release", d.denies.pop(k), turn, sim=sim_id(owner), source=rule)
        # budget by rule, not by sentence: a rule that phrases itself differently every turn
        # (the claims audit names the claims it found) never hits a per-sentence budget, and one
        # fired thirteen times in a single simulation before this.
        fired = owner.__dict__.setdefault("_lb_advice_fired", collections.Counter())
        kept, rules = [], []
        for text, rule in zip(d.advice, d.advice_rules):
            fired[rule] += 1
            if admit(owner, "advice", text) and fired[rule] <= ADVICE_BUDGET:
                kept.append(text)
                rules.append(rule)
            else:
                d.trace.append(("budget", "rule has spoken enough in this simulation", rule))
        d.advice, d.advice_rules = kept, rules
    sim = sim_id(owner)
    for c in d.conflicts:
        line = "[LB_CONFLICT] target=%s winner=%s:%s(E%d) losers=%s" % (
            c["target"], c["winner"][0], c["winner"][1], c["winner"][2],
            ",".join("%s:%s(E%d)" % l for l in c["losers"]))
        print(line, file=sys.stderr, flush=True)
        sidecar("lb-conflict", line, turn, sim=sim, target=str(c["target"]), winner=c["winner"][0],
                loser=",".join(l[0] for l in c["losers"]), winner_grade=c["winner"][2],
                loser_grade=min([l[2] for l in c["losers"]] or [9]))
    for k, text in d.denies.items():
        w = d.won[("call", k)]
        sidecar("lb-deny", text, turn, sim=sim, lb=w.lb, source=w.source, target=str(w.target))
    for text, rule in zip(d.advice, d.advice_rules):
        sidecar("lb-advice", text, turn, sim=sim, source=rule)
    return d


if __name__ == "__main__":
    class C(object):
        def __init__(self, name, args=None):
            self.name, self.arguments, self.id = name, args or {}, name

    class M(object):
        def __init__(self, role="assistant", content="", calls=()):
            self.role, self.content, self.tool_calls = role, content, list(calls)

    c = C("close_card_7834")
    computed = Finding("LB1", DENY, "close_card", c, "NEXT: retention_offer", grade=1, source="procedure")
    prose = Finding("LB4", DENY, "close_card", c, "SKIP the retention offer", grade=5, source="prose")
    d = resolve([prose, computed])
    assert d.denies[id(c)].startswith("NEXT: retention_offer") and "(also noted) SKIP" in d.denies[id(c)]
    assert d.conflicts[0]["winner"][0] == "LB1"
    d2 = resolve([Finding("LB7", SURFACE, "t2", order="do t2", grade=1, facts=["doc unread"]),
                  Finding("LB5", SURFACE, "t1", order="do t1", grade=1)])
    assert d2.advice == ["do t1", "doc unread"], d2.advice
    assert d2.advice_rules == ["LB5", "LB7"], d2.advice_rules   # no source given: the engine names the rule

    class Owner(object):
        pass

    owner = Owner()
    for i in range(4):                      # one rule, four different sentences, budget two
        say(Turn({}, [], M(content="bye")), [Finding("LB4", SURFACE, "t", order="claim %d missing" % i,
                                             grade=1, source="claims")], owner=owner)
    assert owner._lb_advice_fired["claims"] == 4 and ADVICE_BUDGET == 2
    for i in range(DENY_BUDGET + 2):            # one rule denying forever is released after its budget
        c = C("w_%d" % i)
        dd = say(Turn({}, [], M(calls=[c])), [Finding("LB1", DENY, "w", c, "no %d" % i, grade=1, source="procedure:p")],
                 owner=owner)
    assert not dd.denies and owner._lb_deny_fired["procedure"] == DENY_BUDGET + 2
    assert fill("a {x} b {y_z} c {Keep}", x=1) == "a 1 b  c {Keep}"
    assert records_in('{"rows": [{"id": "r1"}, {"id": "r2"}]}', "id") == [{"id": "r1"}, {"id": "r2"}]
    assert records_in('text\n{"id": "r3"}', "id") == [{"id": "r3"}]
    t = Turn({"dispatch": {"agent_call": "call_x", "name_args": {"call_x": "tool", "unlock": "tool"}}},
             [M("user", "Please close my card"), M("tool", "policy: close my card")], M())
    assert t.name_of(C("call_x", {"tool": "inner_1"})) == "inner_1" and t.named(C("unlock", {"tool": "z"})) == "z"
    assert "close my card" in t.user_text and t.executed_fams() == set()
    assert fam("close_card_7834") == "close_card" and fam("verify_identity") == "verify_identity"
    print("lb_coordinator self-test OK")
