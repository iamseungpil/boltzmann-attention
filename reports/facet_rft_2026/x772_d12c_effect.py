# -*- coding: utf-8 -*-
"""x772 — D12 ⓒ 효과 프로브 (`_delivered_unused_agent_tools` 의 침묵 자격 집합)

★x771 이 D12 ⓐⓑ 를 PROBE-PASS 로 닫으면서 **ⓒ 는 미판정**으로 남겼다:
    *"이 프로브는 ⓒ 의 효과를 재지 않았다. ⓒ 를 실으려면 별도 효과 프로브가 필요하다."*
  이 파일이 그 프로브다. 관문 4칸을 다시 채운다.
    ⑴ 결함이 지금 코드에 있나 — `t2_gate_patch.py:3151` 전수 + 라이브 플래그
    ⑵ 실제 실패에 닿는가     — 014 의 `침묵 안 함` 10회 + 그 sim 의 reward
    ⑶ ★수리 전/후가 갈리나   — 같은 재료에 두 술어를 먹여 **침묵 판정**이 달라지는지 센다
    ⑷ [[70]] 파는 것         — 과잉 침묵으로 죽는 정당한 발화를 회수분 전수로 센다

무엇이 결함인가 (설계서 D12 ⓒ).
  P-A 침묵 게이트(:10480-10505)는 `_delivered_unused_agent_tools` 가 **빈 집합**이면
  *"에이전트가 직접 할 수 있는 일이 남아 있지 않다"* 고 판정하고 `[ACTION]` 문면을
  **그대로 내보낸다**(근거 검사조차 건너뛴다). 그 자격 집합은 `_agent_discoverable(env)`
  — **발견형 레지스트리** 하나뿐이라, `_transfer_tools(a2)` 가 선언하는 이관 도구는
  *에이전트가 할 수 있는 일*로 세어지지 않는다.

수리 후보는 하나가 아니다 — 그래서 **셋 다** 잰다([[77]] 반증조건을 먼저 적는다).
  A  `reg = _agent_discoverable(env) | _transfer_tools(a2)`
       설계서 문면 그대로. 배달된 텍스트에 이름이 있어야 한다는 조건은 **유지**된다.
  B  배달 조건을 이관 도구에는 **면제**한다 — 이관은 발견형이 아니라 처음부터 도구 목록에
       서 있으므로 *"배달되어야 안다"* 가 성립하지 않는다.
  C  `_transfer_tools(a2)` ∩ **손님 발화 축자** − 미실행 ("손님이 요구했는데 미실행").

규격([[78]]·[[67]]): 사본 0. 세 팔 모두 **엔진 자신의 함수 소스**를 잘라 와서, 제안된
  편집 **한 줄만** 치환해 exec 한다. 재타이핑하면 드리프트가 생긴다.
  이관 집합은 `t2_gate_patch._transfer_tools(load_domain_a2(...))` 를 **직접 호출**한다.
  근거 검사(축자 0회)는 `t2_gate_patch` 의 **인라인 블록을 잘라** 실행한다(재작성 0).

★재료의 한계를 먼저 밝힌다([[77]]).
  `_agent_discoverable(env)` 는 **오프라인에서 못 얻는다**(로컬에 `tau2` 모듈이 없다).
  그래서 env 를 레지스트리 ∅ 로 세운다. 이것이 **왜 이 모집단에서 정확한가**:
    이 프로브의 모집단은 라이브가 `[T2_ACTIONREQ] 침묵 안 함` 을 찍은 사건뿐이고,
    그 분기의 조건이 바로 `not _pa_open` — 즉 **라이브에서 PRE 집합이 ∅ 이었다는 사실이
    계기로 박제돼 있다**. PRE=∅ 이면
        POST_A = (reg ∪ XFER) ∩ txt − used = (reg∩txt−used) ∪ (XFER∩txt−used) = ∅ ∪ added
    이므로 reg 를 몰라도 **델타는 정확하다**. `[T2_ACTIONREQ] 침묵:` 을 찍은 사건은
    PRE≠∅ 이 확정이므로 이 근사가 성립하지 않는다 ⇒ **모집단에서 제외**한다.
  ⚠단조성: POST 는 자격 집합을 **넓히기만** 한다. 넓히면 `not _pa_open` 분기는 못 켜지고
    근거 검사만 더 켜진다 ⇒ 판정은 *발화→침묵* 한 방향으로만 움직인다(아래에서 검산).

[[57]] 부정통제 3종:
  NC-1 되돌리기 — PRE 팔을 같은 재료에 먹이면 014 사건이 다시 **발화**하는가
  NC-2 A2 제거  — `_transfer_tools({})` = ∅ 이면 POST 팔이 PRE 와 **판정 동일**인가
                  (= 갈림의 출처가 A2 선언이지 엔진 리터럴이 아니라는 증거 · [[05]])
  NC-3 단조성   — POST 가 *침묵→발화* 로 뒤집는 사건이 하나도 없는가
"""
import gzip
import io
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict

REPO = r"C:\workspace\ba-frft"
ENG = os.path.join(REPO, "scripts", "distill", "tau2")
SIMS = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
OUT = os.path.join(REPO, "reports", "facet_rft_2026", "_ep_work", "x772_d12c_effect.json")
sys.path.insert(0, ENG)

import t2_gate_patch as G                                        # noqa: E402
import gate_interpreter as GI                                    # noqa: E402

NOSIL = "[T2_ACTIONREQ] 침묵 안 함"        # PRE 가 발화를 택한 사건 (= _pa_open 이 ∅)
SIL = "[T2_ACTIONREQ] 침묵: formalized_target"   # PRE 가 이미 침묵한 사건 (= _pa_open ≠ ∅)
RE_TGT = re.compile(r"\(target=([^)]+)\)")


def say(*a):
    print(" ".join(str(x) for x in a))


# ═══════════════════════════════════════════════════════════════════════════
# ⑴ 결함이 지금 코드에 있나
# ═══════════════════════════════════════════════════════════════════════════
SRC = io.open(os.path.join(ENG, "t2_gate_patch.py"), encoding="utf-8").read()


def _func_src(name):
    """엔진 파일에서 `def name(...)` 함수 본문을 통째로 잘라 온다 (사본 0)."""
    i = SRC.index("\ndef %s(" % name) + 1
    j = SRC.index("\ndef ", i + 1)
    return SRC[i:j].rstrip() + "\n", SRC[:i].count("\n") + 1


def gate1():
    say("=" * 86)
    say("[GATE 1] 결함이 지금 코드에 있나")
    src, ln = _func_src("_delivered_unused_agent_tools")
    say("   %s:%d  _delivered_unused_agent_tools" % ("t2_gate_patch.py", ln))
    reg_line = [l for l in src.splitlines() if l.strip().startswith("reg = ")][0]
    say("   축자  %s" % reg_line.strip())
    say("   ⇒ 이 함수가 `_transfer_tools` 를 참조하나 = %s"
        % ("_transfer_tools" in src))
    say("   ⇒ 자격 집합의 유일한 출처 = `_agent_discoverable(env)` (발견형 레지스트리)")

    # 라이브 여부 ([[81]]) — 이 경로는 플래그 뒤에 있다.
    call_i = SRC.index("_pa_on = os.environ.get(\"T2_ACTIONREQ_GROUNDED\")")
    say("   호출부 t2_gate_patch.py:%d  `_pa_on = T2_ACTIONREQ_GROUNDED == \"1\"`"
        % (SRC[:call_i].count("\n") + 1))
    for sh in ("go_stack.sh", "run_ours_task.sh"):
        try:
            t = io.open(os.path.join(ENG, sh), encoding="utf-8").read()
        except Exception:
            continue
        for k, l in enumerate(t.splitlines(), 1):
            if "T2_ACTIONREQ_GROUNDED" in l and l.strip().startswith("export"):
                say("   런처 %s:%d  %s" % (sh, k, l.strip()))
    say("   ⚠ run_ours_task.sh 가 go_stack.sh 를 source 한 **뒤** 덮으므로 라이브 = 1 (경로 살아있음)")
    return src


# ═══════════════════════════════════════════════════════════════════════════
# ⑵ 재료 회수 — trace 는 sim·turn 을 함께 갖는다(`turn = len(state.messages)`)
# ═══════════════════════════════════════════════════════════════════════════
def recover_events():
    """`침묵 안 함` / `침묵` 사건을 (run, sim, turn, target) 으로 회수한다.

    `t2_lever_beat.set_turn` 축자: `_LOCAL.turn = len(state.messages)` ⇒ trace 의 `turn`
    은 **그 시점 메시지 개수**다. 그래서 재료 prefix = `messages[:turn]` 이고,
    별도의 정렬 규칙이 필요 없다.
    """
    fires, sils = [], []
    for fn in sorted(os.listdir(SIMS)):
        if not (fn.startswith("trace_") and fn.endswith(".jsonl.gz")):
            continue
        run = fn[len("trace_"):-len(".jsonl.gz")]
        try:
            fh = gzip.open(os.path.join(SIMS, fn), "rt", encoding="utf-8", errors="replace")
        except Exception:
            continue
        for l in fh:
            if NOSIL not in l and SIL not in l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            line = str(r.get("line") or "")
            sim, turn = r.get("sim"), r.get("turn")
            if sim is None or turn is None:
                continue
            if NOSIL in line:
                m = RE_TGT.search(line)
                fires.append(dict(run=run, sim=sim, turn=int(turn),
                                  utgt=(m.group(1) if m else None)))
            else:
                m = re.search(r"formalized_target=(\S+)", line)
                sils.append(dict(run=run, sim=sim, turn=int(turn),
                                 utgt=(m.group(1) if m else None)))
    return fires, sils


_RESCACHE = {}


def load_sim(run, sim):
    """그 런의 results 에서 그 sim 의 메시지 목록 (없으면 None)."""
    if run not in _RESCACHE:
        p = os.path.join(SIMS, run + ".results.json.gz")
        d = None
        if os.path.exists(p):
            try:
                d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
            except Exception:
                d = None
        idx = {}
        for s in ((d or {}).get("simulations") or []):
            tid, seed = s.get("task_id"), s.get("seed")
            idx["%s#s%s" % (tid, seed)] = s
            idx.setdefault(str(tid), s)
        _RESCACHE[run] = idx
    return _RESCACHE[run].get(sim) or _RESCACHE[run].get(str(sim).split("#")[0])


class _Call(object):
    """회수된 호출 껍데기 — 엔진이 읽는 필드만 갖는다(판단 0)."""
    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = d.get("arguments") or {}
        self.id = d.get("id")


class _Msg(object):
    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [_Call(c) for c in (d.get("tool_calls") or [])]


class _Env(object):
    """레지스트리 ∅ env — 위 docstring 의 정확성 논거를 코드로 고정한다."""
    class tools(object):
        @staticmethod
        def get_discoverable_tools():
            return set()


class _Orch(object):
    def __init__(self):
        self._t2_orch = type("O", (), {"environment": _Env()})()


# ═══════════════════════════════════════════════════════════════════════════
# ⑶ 세 팔 — 엔진 함수 소스에 **제안된 편집 한 줄만** 치환해 exec 한다
# ═══════════════════════════════════════════════════════════════════════════
_REGLINE = ("        reg = _agent_discoverable(getattr(getattr(orch, \"_t2_orch\", None), "
            "\"environment\", None))")
_GUARD = "        if not reg:\n            return []"
_RET = "        return sorted(n for n in reg if n in txt and n not in used)"

# A — 설계서 문면 그대로: 이관 도구가 **자격 레지스트리에 합류**한다.
#     배달 조건(`n in txt`)은 그대로 걸린다.
EDIT_A = [(_REGLINE, _REGLINE + " | set(_transfer_tools(a2) or ())")]

# B — 이관 도구에는 **배달 조건을 면제**한다(발견형이 아니라 처음부터 도구 목록에 있다).
#     ⚠빠른 반환 `if not reg` 도 함께 옮겨야 한다 — 안 옮기면 레지스트리가 빈 순간
#       이관 자격이 통째로 증발하고, 그것은 수리가 아니라 **다른 버그**다.
EDIT_B = [(_REGLINE, _REGLINE + "\n        _xf = set(_transfer_tools(a2) or ())"),
          (_GUARD, "        if not reg and not _xf:\n            return []"),
          (_RET, "        return sorted(set(n for n in reg if n in txt and n not in used)\n"
                 "                      | {n for n in _xf if n not in used})")]

# C — "손님이 요구했는데 미실행": 이관 도구 이름이 **손님 발화 축자**에 있고 미실행.
EDIT_C = [(_REGLINE, _REGLINE + "\n        _xf = set(_transfer_tools(a2) or ())"),
          (_GUARD, "        if not reg and not _xf:\n            return []"),
          (_RET, "        _utx = chr(10).join(str(getattr(m, 'content', '') or '')\n"
                 "                            for m in (messages or [])\n"
                 "                            if getattr(m, 'role', None) == 'user')\n"
                 "        return sorted(set(n for n in reg if n in txt and n not in used)\n"
                 "                      | {n for n in _xf if n in _utx and n not in used})")]


def build_arm(func_src, edit=None, xfer_override=None):
    """엔진 함수 바이트에 **제안된 편집만** 치환해 컴파일한다."""
    code = func_src
    for old, new in (edit or []):
        assert old in code, "치환 대상이 소스에 없다 — 계약 위치가 바뀌었다:\n%s" % old
        code = code.replace(old, new, 1)
    ns = dict(G.__dict__)
    if xfer_override is not None:
        ns["_transfer_tools"] = lambda _a2: xfer_override
    exec(compile(textwrap.dedent(code), "<engine-slice>", "exec"), ns)
    return ns["_delivered_unused_agent_tools"]


# ── 근거 검사(축자 0회)도 엔진 인라인 블록을 잘라서 쓴다 ────────────────────
def build_grounded():
    i = SRC.index("_seen_txt = []")
    j = SRC.index("_utgt = None", i) + len("_utgt = None")
    blk = textwrap.dedent(SRC[SRC.rfind("\n", 0, i) + 1:j])
    # `if ...:` 안의 print 는 stderr 로 나가므로 무해하나, 캡처해서 조용히 만든다.
    compiled = compile(blk, "<engine-grounded>", "exec")

    def grounded_silences(messages, utgt):
        ns = {"state": type("S", (), {"messages": messages})(), "_utgt": utgt,
              "json": json, "_args_dict": G._args_dict,
              "_sys": type("S", (), {"stderr": open(os.devnull, "w")})(),
              "print": lambda *a, **k: None}
        exec(compiled, ns)
        return ns["_utgt"] is None          # None 이 됐다 = 침묵한다
    return grounded_silences


# ═══════════════════════════════════════════════════════════════════════════
def main():
    func_src = gate1()
    a2 = GI.load_domain_a2("banking_knowledge")
    XFER = set(G._transfer_tools(a2) or ())
    say("   `_transfer_tools(load_domain_a2('banking_knowledge'))` = %s   (엔진 직접 호출)"
        % sorted(XFER))

    pre = build_arm(func_src)
    arms = {"A": build_arm(func_src, EDIT_A),
            "B": build_arm(func_src, EDIT_B),
            "C": build_arm(func_src, EDIT_C)}
    grounded_silences = build_grounded()
    orch = _Orch()

    # ── ★계기 검산 (하네스 자기검정) ────────────────────────────────────────
    #   이 함수는 `except Exception: return []` 로 **모든 예외를 삼킨다**. 편집이 터져도
    #   빈 목록이 나오고, 그것은 *"효과 없음"* 과 구별되지 않는다([[55]] 계기 먼저).
    #   그래서 각 팔이 **움직일 수 있다는 것**을 먼저 보인다 — 이건 증거 재료가 아니라
    #   하네스가 죽지 않았다는 확인이다.
    xf1 = sorted(XFER)[0]
    probe_msgs = [_Msg({"role": "tool", "content": "... %s ..." % xf1}),
                  _Msg({"role": "user", "content": "please %s now" % xf1})]
    live = {"PRE": pre(orch, probe_msgs, a2)}
    for k, fn in arms.items():
        live[k] = fn(orch, probe_msgs, a2)
    say("   [하네스 검산] 이관 이름이 도구텍스트·손님발화에 **둘 다** 있는 재료에서")
    say("      PRE=%s · A=%s · B=%s · C=%s" % (live["PRE"], live["A"], live["B"], live["C"]))
    assert not live["PRE"], "PRE 가 ∅-env 에서 비지 않았다 — 하네스가 틀렸다"
    for k in ("A", "B", "C"):
        assert live[k], "POST-%s 가 움직이지 않는다 — 편집이 예외로 삼켜졌을 수 있다" % k
    say("      ⇒ 세 팔 전부 **움직인다**. 이후의 0 은 편집 실패가 아니라 재료의 사실이다.")

    fires, sils = recover_events()
    say("")
    say("=" * 86)
    say("[GATE 2] 재료 — 회수분 전수")
    say("   `침묵 안 함`(PRE 발화) 사건 = %d   ·   `침묵`(PRE 이미 침묵) 사건 = %d"
        % (len(fires), len(sils)))
    say("   ⇒ 모집단 = 발화 사건 %d 뿐 (침묵 사건은 PRE≠∅ 이라 ∅-env 근사가 안 선다)"
        % len(fires))

    rows, skipped = [], Counter()
    for e in fires:
        s = load_sim(e["run"], e["sim"])
        if s is None:
            skipped["results 없음"] += 1
            continue
        allm = s.get("messages") or []
        if e["turn"] > len(allm):
            skipped["turn > 메시지 수"] += 1
            continue
        msgs = [_Msg(m) for m in allm[:e["turn"]]]
        p = pre(orch, msgs, a2)
        if p:                                   # ∅-env 인데 비었지 않다 = 있을 수 없다
            skipped["PRE 비공집합(모순)"] += 1
            continue
        r = dict(e)
        r["task"] = str(e["sim"]).split("#")[0]
        r["reward"] = (s.get("reward_info") or {}).get("reward")
        r["pre_set"] = list(p)
        r["pre_silences"] = False               # PRE 는 ∅ ⇒ 근거 검사 자체가 안 돈다
        gs = grounded_silences(msgs, e["utgt"])
        r["utgt_absent"] = gs                   # 표적이 대화 축자에 0회인가
        for k, fn in arms.items():
            st = fn(orch, msgs, a2)
            r["post_%s_set" % k] = list(st)
            r["post_%s_silences" % k] = bool(st) and gs
        rows.append(r)

    say("   사용 가능한 사건 = %d  (제외 %s)" % (len(rows), dict(skipped)))

    say("")
    say("=" * 86)
    say("[GATE 3] ★수리 전/후로 판정이 갈리나 — 같은 재료 · 엔진 바이트 세 팔")
    say("   PRE  : `침묵 안 함` 분기 = **문면 발화** (자격 집합 ∅ · 근거 검사 건너뜀)")
    say("   표적이 대화 축자에 0회인 사건 = %d/%d"
        % (sum(1 for r in rows if r["utgt_absent"]), len(rows)))
    res = {}
    for k in ("A", "B", "C"):
        flip = [r for r in rows if r["post_%s_silences" % k]]
        nonempty = [r for r in rows if r["post_%s_set" % k]]
        res[k] = flip
        say("   POST-%s : 자격 집합이 비지 않게 된 사건 %3d/%d   ·   **판정 갈림(발화→침묵) %3d**"
            % (k, len(nonempty), len(rows), len(flip)))
        if flip:
            say("            갈림 태스크 %s"
                % Counter(r["task"] for r in flip).most_common())

    say("")
    say("=" * 86)
    say("[GATE 4] [[70]] 파는 것 — 과잉 침묵")
    say("   침묵은 `[ACTION]` 문면을 그 턴에 **안 내보낸다**. 문면이 옳았던 자리를 죽이면 그것이 비용이다.")
    for k in ("A", "B", "C"):
        flip = res[k]
        if not flip:
            say("   POST-%s : 갈림 0 ⇒ 파는 것도 0 (이 수리는 이번 회수분에서 아무것도 안 한다)" % k)
            continue
        byt = Counter(r["task"] for r in flip)
        rw = Counter()
        for r in flip:
            v = r.get("reward")
            rw["reward=1" if v == 1 else ("reward=0" if v == 0 else "reward=?")] += 1
        say("   POST-%s : 갈림 %d 건" % (k, len(flip)))
        say("            그 sim 의 최종 reward 분포 = %s" % dict(rw))
        say("            ★통과한 sim 에서 죽는 발화 = %d 건 (= 파는 것의 상한)"
            % rw.get("reward=1", 0))
        say("            태스크별 = %s" % byt.most_common())

    # ── 016 은 설계 주석이 **발화를 지키라**고 이름 붙인 자리다 ──────────────
    say("")
    say("   ★설계 주석(:10467-10476) 축자 대조 — 016 은 *'되찾을 자리'* 로 발화 38 을 지목했다.")
    for k in ("A", "B", "C"):
        n16 = sum(1 for r in res[k] if r["task"] == "task_016")
        tot16 = sum(1 for r in rows if r["task"] == "task_016")
        say("      POST-%s 가 016 에서 죽이는 발화 = %d / 회수된 016 발화 %d" % (k, n16, tot16))

    # ═══════════════════════════════════════════════════════════════════════
    say("")
    say("=" * 86)
    say("[[57]] 부정통제")
    # NC-1 되돌리기
    t14 = [r for r in rows if r["task"] == "task_014"]
    say("   NC-1 되돌리기: PRE 팔에서 014 사건 %d 건이 전부 **발화**인가 = %s"
        % (len(t14), all(not r["pre_silences"] for r in t14)))
    for r in t14[:3]:
        say("        %s turn=%d target=%s  PRE=발화 · POST_B=%s"
            % (r["sim"], r["turn"], r["utgt"],
               "침묵" if r["post_B_silences"] else "발화"))
    # NC-2 A2 제거
    for k, ed in (("A", EDIT_A), ("B", EDIT_B), ("C", EDIT_C)):
        fn0 = build_arm(func_src, ed, xfer_override=set())
        n = 0
        for r in rows:
            msgs = [_Msg(m) for m in
                    (load_sim(r["run"], r["sim"]).get("messages") or [])[:r["turn"]]]
            if not fn0(orch, msgs, a2):
                n += 1
        say("   NC-2 A2 제거(POST-%s · `_transfer_tools`=∅): PRE 와 판정 동일 = %d/%d"
            % (k, n, len(rows)))
    say("        ⇒ 전건 동일이면 갈림의 출처는 **A2 선언**이지 엔진 리터럴이 아니다([[05]])")
    # NC-3 단조성
    bad = [r for k in ("A", "B", "C") for r in rows
           if r["pre_silences"] and not r["post_%s_silences" % k]]
    say("   NC-3 단조성: POST 가 *침묵→발화* 로 뒤집은 사건 = %d (0 이어야 한다)" % len(bad))

    # ═══════════════════════════════════════════════════════════════════════
    winner = max(("A", "B", "C"), key=lambda k: len(res[k]))
    say("")
    say("=" * 86)
    n = len(res[winner])
    say("[판정] 최대 갈림 = POST-%s  %d/%d  ⇒ **%s**"
        % (winner, n, len(rows), "PROBE-PASS" if n else "PROBE-FAIL (효과 없음)"))
    try:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump({"rows": rows, "flips": {k: len(v) for k, v in res.items()},
                   "n": len(rows), "xfer": sorted(XFER), "skipped": dict(skipped)},
                  io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        say("[산출] %s" % OUT)
    except Exception as e:
        say("[산출] 실패 %r" % (e,))


if __name__ == "__main__":
    main()
