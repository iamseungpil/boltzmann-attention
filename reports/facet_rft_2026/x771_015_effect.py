# -*- coding: utf-8 -*-
r"""x771 — **D11ⓐ 효과 프로브** (표적 = task_015 · pre-give 재생성이 env-변이 호출을 잃는다)

★사용자 관문 네 칸 (2026-09-05):
  ⑴ 결함이 지금 코드에 있나            — 파일:줄 + 축자            → §A
  ⑵ 그 결함이 실제 실패에 닿나          — 회수 궤적의 그 발화·그 칸  → §B
  ⑶ ★수리 전/후로 판정이 갈리나         — 같은 재료 · 두 팔 · 실측   → §C
  ⑷ [[70]] 무엇을 파나                 — 회수분 **전수**            → §D
  [[57]] 부정통제                                                  → §E

★[[78]] 규격 — 프롬프트 저작 0 · 모델 호출 0 · 새 런 0.
  · 재료 = 회수 trace/results 의 **실제 호출**(가짜 입력 0).
  · 변이 도구 집합 = **환경 선언**(`a2/env_surface.json` 의 `mutates`)을
    `t2_forensic.mutating_tools()` 로 그대로 읽는다([[23]] gold 미접촉 · [[67]] 사본 금지).
  · give 도구 이름조차 이 파일에 상수로 쓰지 않는다 — **엔진 소스에서 추출**한다([[05]]).

용법:  PYTHONIOENCODING=utf-8 py -3 reports/facet_rft_2026/x771_015_effect.py
"""
import ast
import collections
import contextlib
import gzip
import io
import json
import os
import re
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ENG = os.path.join(REPO, "scripts", "distill", "tau2")
SIMS = os.path.join(HERE, "sim_results")
sys.path.insert(0, ENG)

import t2_forensic as F                                      # noqa: E402  (엔진 정본 라이브러리)
import t2_gate_patch as G                                    # noqa: E402  (`_exact_tool_name` 정본)

GATE = os.path.join(ENG, "t2_gate_patch.py")

# ─────────────────────────────────────────────────────────────────────────────
# §G 배선 후 재확인 (2026-09-05 수리 `T2_REGEN_KEEP_MUTATING` 이후 추가)
#   ★검정이 코드를 **베껴 적으면** 드리프트가 검정을 통과시킨다([[84]]).
#     그래서 이 절은 `t2_gate_patch.py` 의 그 블록을 **AST 로 떼어 그대로 exec** 한다.
#     즉 §C 의 POST-B 는 *모형*이었고 §G 는 **실제로 배선된 코드**다.
# ─────────────────────────────────────────────────────────────────────────────
WIRED_FLAG = "T2_REGEN_KEEP_MUTATING"


def wired_code(src):
    """`_ap_regen` 안의 D11ⓐ 블록만 떼어 컴파일한다. (code, 소스) — 없으면 (None, "")."""
    fn = next((n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "_ap_regen"), None)
    if fn is None:
        return None, ""
    body = ast.get_source_segment(src, fn) or ""
    i = body.find('if os.environ.get("%s")' % WIRED_FLAG)
    j = body.rfind("return _am2")
    if i < 0 or j < i:
        return None, ""
    blk = textwrap.dedent(body[body.rindex("\n", 0, i) + 1:j])
    return compile(blk, "<t2_gate_patch:_ap_regen:D11a>", "exec"), blk


def wired_run(code, draft, regen, mutset, flag="1"):
    """배선된 블록을 실행한다 → (산출 tool_call 이름들, `restored=` 계기 라인 수).

    재료는 회수분의 **실제 이름**이고, 변이 판정은 환경 선언(`mutset`)을 그대로 쓴다.
    엔진의 `_exact_tool_name` 은 정본을 import 해서 쓴다(사본 0 · [[67]]).
    """
    def _tc(n, i):
        return types.SimpleNamespace(name=n, arguments={}, id="c%d" % i)
    am_ = types.SimpleNamespace(content="draft",
                                tool_calls=[_tc(n, i) for i, n in enumerate(draft)] or None)
    am2_ = types.SimpleNamespace(content="regen",
                                 tool_calls=[_tc(n, 90 + i) for i, n in enumerate(regen)] or None)
    ns = {"os": os, "_sys": sys, "am": am_, "_am2": am2_, "tag": "probe",
          "self": types.SimpleNamespace(_t2_orch=types.SimpleNamespace(
              environment=types.SimpleNamespace(_is_mutating_tool=lambda n: n in mutset))),
          "_exact_tool_name": G._exact_tool_name}
    old, buf = os.environ.get(WIRED_FLAG), io.StringIO()
    try:
        if flag is None:
            os.environ.pop(WIRED_FLAG, None)
        else:
            os.environ[WIRED_FLAG] = flag
        with contextlib.redirect_stderr(buf):
            exec(code, ns)
    finally:
        if old is None:
            os.environ.pop(WIRED_FLAG, None)
        else:
            os.environ[WIRED_FLAG] = old
    names = [str(getattr(t, "name", "")) for t in (getattr(am2_, "tool_calls", None) or [])]
    return names, buf.getvalue().count("[%s] restored=" % WIRED_FLAG)

# ─────────────────────────────────────────────────────────────────────────────
# ⑴ 결함 축자 — 실행 시 파일에서 다시 읽어 대조한다(문서 드리프트 방지 · [[77]])
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_QUOTES = [
    ("usertoolnote 재생성 호출",
     '_new5 = _ap_regen("Note: " + _tpl5.format(tool=_want5), "usertoolnote")'),
    ("usertoolnote 무조건 교체",
     "if _new5 is not None:\n                            am = _new5"),
    ("givequote 재생성 호출",
     '_new1p = _ap_regen(_tpl1.format(tool=_want1 or "this tool", min=_min1),'),
    ("givequote 무조건 교체",
     "if _new1p is not None:\n                            am = _new1p"),
    ("계기가 손실을 이미 세고 있다",
     "[T2_GIVE_QUOTE] retract=%d (give_present_after_reask=%d)"),
]

# 엔진이 스스로 선언한 give 도구 이름을 **소스에서 추출**한다(프로브가 저작하지 않는다).
RX_GIVE_LITERAL = re.compile(
    r"_giv5 = next\(\(t for t in \(am\.tool_calls or \[\]\)\s*\n\s*"
    r'if str\(getattr\(t, "name", ""\)\) == "([A-Za-z0-9_]+)"\), None\)')


# ─────────────────────────────────────────────────────────────────────────────
# 두 팔 — 입력 동일 · 술어만 다르다
#   PRE  (현행) : `_ap_regen` 산출이 None 만 아니면 무조건 `am` 교체
#                 = t2_gate_patch.py 의 `if _new5 is not None: am = _new5` 축자 등가
#   POST (수리) : 초안의 **env-변이 호출**이 산출에 남지 않으면 교체 기각(원본 유지)
#                 = 닫힌 술어 하나(집합 포함). 도구 이름 열거 0 · 값 선택 0 · 새 결정론 0.
# ─────────────────────────────────────────────────────────────────────────────
def arm_pre(draft_names, regen_names, mut):
    return ("REPLACE", [])


def arm_post(draft_names, regen_names, mut):
    lost = sorted({n for n in draft_names if n in mut} - set(regen_names))
    return ("KEEP_ORIGINAL", lost) if lost else ("REPLACE", [])


def arm_post_merge(draft_names, regen_names, mut):
    """수리안 B — 기각 대신 **잃은 변이 호출만 되붙인다**(문면은 재생성 것을 그대로 쓴다).

    엔진 선례: `_ap_regen` 의 C170 부분-수용이 이미 `_am2.tool_calls` 를 재구성한다
    (`t2_gate_patch.py` "partial-accept: dropped %d gate-denied, kept %d call(s)").
    새 결정론 0 — 고르지 않고 **잃은 집합 그대로** 복원한다.
    """
    lost = sorted({n for n in draft_names if n in mut} - set(regen_names))
    return ("REPLACE_MERGED", lost) if lost else ("REPLACE", [])


# ─────────────────────────────────────────────────────────────────────────────
# 회수분 파서 — 계기 라인만 읽는다. 새 계기 0.
# ─────────────────────────────────────────────────────────────────────────────
RX_GEN = re.compile(r"^\[T2_GEN_TRACE\] call=(\S+) .*-> gen=(\d+) prompt=(\d+)"
                    r".*content=(\d+)B tool_calls=(\d+)")
RX_UTN = re.compile(r"^\[T2_USER_TOOL_NOTE\] pre-give note: (\S+)")
RX_GQ = re.compile(r"^\[T2_GIVE_QUOTE\] no verbatim customer span in message before give=(\S+)")
RX_RET = re.compile(r"^\[T2_GIVE_QUOTE\] retract=(\d+) \(give_present_after_reask=(\d+)\)")
RX_CAMPAIGN = re.compile(r"2026090[3-5]")
PREGIVE_TAGS = ("usertoolnote", "givequote")


def campaign_traces():
    """캠페인 = 태그에 2026-09-03~05 가 박힌 banking 런(설계서 §1e 의 범위 정의와 동형)."""
    return [os.path.join(SIMS, fn) for fn in sorted(os.listdir(SIMS))
            if fn.startswith("trace_bank_") and fn.endswith(".jsonl.gz") and RX_CAMPAIGN.search(fn)]


def parse_trace(path):
    """이 런의 생성 이벤트 전수. 한 건 = 한 번의 `_gen`(= `am` 후보 1개)."""
    gens, prev, pend = [], {}, {}
    run = os.path.basename(path)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            L, key, mk = d.get("line") or "", (d.get("sim"), d.get("turn")), d.get("mark")
            if mk == "T2_USER_TOOL_NOTE":
                m = RX_UTN.match(L)
                if m:
                    pend.setdefault(key, []).append((m.group(1), "usertoolnote"))
            elif mk == "T2_GIVE_QUOTE":
                m = RX_GQ.match(L)
                if m:
                    pend.setdefault(key, []).append((m.group(1), "givequote"))
                m = RX_RET.match(L)
                if m and gens and gens[-1]["tag"] == "givequote" \
                        and (gens[-1]["sim"], gens[-1]["turn"]) == key:
                    gens[-1]["retract"] = int(m.group(1))
            elif mk == "T2_GEN_TRACE":
                m = RX_GEN.match(L)
                if not m:
                    continue
                call, clen, tc = m.group(1), int(m.group(4)), int(m.group(5))
                if not call.startswith("agent_response"):
                    continue                    # 서브콜(formalize·claimprov 선언 등)은 `am` 이 아니다
                tag = call[len("agent_response"):].lstrip("_")
                give = None
                if tag in PREGIVE_TAGS:
                    for i, (t, kind) in enumerate(pend.get(key, [])):
                        if kind == tag:
                            give = t
                            pend[key].pop(i)
                            break
                p = prev.get(key)
                gens.append(dict(run=run, sim=d.get("sim"), turn=d.get("turn"), tag=tag,
                                 call=call, clen=clen, tc=tc, give=give, retract=None,
                                 draft_call=(None if p is None else p["call"]),
                                 draft_clen=(None if p is None else p["clen"]),
                                 draft_tc=(None if p is None else p["tc"])))
                prev[key] = gens[-1]
    # (sim,turn) 의 마지막 생성만이 궤적에 커밋될 수 있다
    last = {}
    for g in gens:
        last[(g["sim"], g["turn"])] = g
    for g in gens:
        g["is_last"] = (last[(g["sim"], g["turn"])] is g)
    return gens


def load_committed(run):
    """trace 파일명 → 같은 태그의 results.json.gz 에서 sim별 assistant 메시지 열."""
    tag = run[len("trace_bank_"):-len(".jsonl.gz")]
    p = os.path.join(SIMS, "bank_%s.results.json.gz" % tag)
    if not os.path.exists(p):
        return None
    try:
        res = json.load(gzip.open(p, "rt", encoding="utf-8"))
    except Exception:
        return None
    out = {}
    for s in res.get("simulations") or []:
        seq = [(i, len(m.get("content") or ""),
                [t.get("name") for t in (m.get("tool_calls") or [])])
               for i, m in enumerate(s.get("messages") or []) if m.get("role") == "assistant"]
        out["%s#s%s" % (s.get("task_id"), s.get("seed"))] = (s, seq)
    return out


def align(gens_last, seq):
    """궤적 조인 — **순서 조인**. `is_last` 생성 열 ↔ 커밋 assistant 열을 offset 하나로 맞춘다.

    각 쌍의 `(content 길이, tool_calls 수)` 가 정확히 일치하는 개수를 최대화하는 offset 을
    고르고, 일치율이 낮으면 **모른다**를 돌려준다(추정 금지 · [[77]]).
    반환: (mapping {gen index -> (msg index, names)}, 일치율, offset)
    """
    best = (None, -1.0, None)
    for off in range(0, max(0, len(seq) - len(gens_last)) + 1):
        pairs = list(zip(gens_last, seq[off:off + len(gens_last)]))
        if not pairs:
            continue
        hit = sum(1 for g, s in pairs if g["clen"] == s[1] and g["tc"] == len(s[2]))
        rate = hit / float(len(pairs))
        if rate > best[1]:
            best = ({id(g): (s[0], s[2]) for g, s in pairs}, rate, off)
    if best[1] < 0.9:
        return None, best[1], best[2]
    return best[0], best[1], best[2]


def materials(g, give_name):
    """(draft_names, regen_names, 근거) — regen_names is None 이면 **모른다**([[77]]).

    draft_names : pre-give 레버 라인은 코드상 `_giv5/_giv1 is not None` 안에서만 출력된다
                  ⇒ 그 라인의 존재 자체가 **초안이 give 호출을 갖고 있었다**는 직접 증거.
    regen_names : tc==0 이면 확정 ∅ · givequote 는 엔진 자신의 `retract` 라인이 판정 ·
                  그 밖은 순서-조인으로 커밋 메시지의 실제 이름을 회수.
    """
    draft = [give_name] if g["give"] else []
    if g["tc"] == 0:
        return draft, [], "regen tool_calls=0"
    if g["tag"] == "givequote" and g["retract"] is not None:
        return (draft, ([] if g["retract"] == 1 else list(draft)),
                "engine retract=%d" % g["retract"])
    if g.get("commit"):
        return draft, list(g["commit"][1]), "committed msg[%d]" % g["commit"][0]
    return draft, None, "tool_calls=%d · 이름 미회수" % g["tc"]


def give_delay(g, sims_by_key, give_name):
    """그 재생성 이후 give 가 **실제로 커밋될 때까지** 걸린 메시지 수(같은 sim · 커밋 궤적).

    0    = 이 산출 자체가 give 를 담고 있다(손실 없음)
    n>0  = n 개 메시지 뒤에 같은 도구가 나갔다(= 우회 비용)
    'never' = 그 sim 에서 끝내 안 나갔다
    '?'  = 궤적 조인 실패(모른다)
    """
    ent = sims_by_key.get((g["run"], g["sim"]))
    anchor = g.get("commit") or g.get("turn_commit")   # 같은 턴의 마지막 생성이 커밋된 자리
    if not ent or not anchor:
        return "?"
    msgs = ent[0].get("messages") or []
    i0 = anchor[0]
    if give_name in (anchor[1] or []):
        return 0
    for j in range(i0 + 1, len(msgs)):
        m = msgs[j]
        if m.get("role") != "assistant":
            continue
        if any(t.get("name") == give_name for t in (m.get("tool_calls") or [])):
            return j - i0
    return "never"


def hr(t):
    print("\n" + "=" * 84)
    print(t)
    print("=" * 84)


def main():
    mut = F.mutating_tools()
    src = open(GATE, encoding="utf-8").read()

    # ── §A ⑴ ────────────────────────────────────────────────────────────────
    hr("A. ⑴ 결함이 지금 코드에 있나  —  %s" % GATE)
    alive = True
    for why, q in SOURCE_QUOTES:
        ok = q in src
        alive &= ok
        ln = (src[:src.index(q)].count("\n") + 1) if ok else None
        print("  [%-5s] line=%-6s %-24s %r" % ("ALIVE" if ok else "GONE", ln, why, q[:62]))
    mg = RX_GIVE_LITERAL.search(src)
    give_name = mg.group(1) if mg else None
    print("  엔진 소스에서 추출한 give 도구 이름 = %r  (변이 선언 포함? %s)"
          % (give_name, give_name in mut if give_name else None))
    print("  변이 도구 집합(환경 선언 `mutates`) 크기 = %d" % len(mut))
    # [[81]] 표적 레버가 **정본 런처에 실제로 켜져 있는가** — 아니면 라이브 효과가 0 이다.
    gsp = os.path.join(ENG, "go_stack.sh")
    gs = open(gsp, encoding="utf-8").read() if os.path.exists(gsp) else ""
    for lv in ("T2_USER_TOOL_NOTE", "T2_GIVE_QUOTE"):
        m2 = re.search(r"^export %s=(\S+)" % lv, gs, re.M)
        print("  [[81]] go_stack.sh: %-20s = %s" % (lv, m2.group(1) if m2 else "미등재"))
    print("  => ⑴ 결함 살아있음: %s" % alive)
    if not (alive and give_name):
        print("  ⛔ 코드가 바뀌었다 — 프로브 중단")
        return

    # ── 회수 + 궤적 순서-조인 ───────────────────────────────────────────────
    files = campaign_traces()
    allg, sims_by_key, joined, tried_join = [], {}, 0, 0
    for f in files:
        run = os.path.basename(f)
        gs = parse_trace(f)
        allg.extend(gs)
        com = load_committed(run)
        if not com:
            continue
        bysim = collections.defaultdict(list)
        for g in gs:
            if g["is_last"]:
                bysim[g["sim"]].append(g)
        for simtag, lastgens in bysim.items():
            ent = com.get(simtag)
            if not ent:
                continue
            tried_join += 1
            mp, rate, off = align(lastgens, ent[1])
            if mp is None:
                continue
            joined += 1
            sims_by_key[(run, simtag)] = ent
            for g in lastgens:
                g["commit"] = mp.get(id(g))
        # 같은 턴의 마지막 생성이 앉은 커밋 자리를 그 턴의 모든 생성에 붙인다(지연 측정 기준점)
        anchors = {(g["sim"], g["turn"]): g.get("commit") for g in gs if g["is_last"]}
        for g in gs:
            g["turn_commit"] = anchors.get((g["sim"], g["turn"]))
    regens = [g for g in allg if g["tag"]]
    pregive = [g for g in regens if g["tag"] in PREGIVE_TAGS]
    print("\n  회수: 캠페인 trace %d · 생성 이벤트 %d · `_ap_regen` %d · pre-give %d"
          % (len(files), len(allg), len(regens), len(pregive)))
    print("  궤적 순서-조인 성공 sim = %d / %d  (일치율 ≥0.9 인 offset 만 채택)"
          % (joined, tried_join))

    # ── §B ⑵ ────────────────────────────────────────────────────────────────
    hr("B. ⑵ 015 — 결함이 실제 실패에 닿는가 (회수 궤적 축자)")
    RUN15 = "trace_bank_k8143med1_20260904_0135.jsonl.gz"
    KEY15 = (RUN15, "task_015#s626729")
    g015 = [g for g in allg if g["run"] == RUN15 and g["sim"] == "task_015#s626729"
            and 26 <= (g["turn"] or 0) <= 32]
    print("  ▸ 같은 sim 안 세 번의 give 초안 — 앞의 둘만 pre-give 레버가 맞았다 (레버는 sim당 1회)")
    for g in g015:
        c = g.get("commit")
        print("    t%-4s %-34s clen=%-5s tc=%s  give=%-18s retract=%-4s  커밋=%s"
              % (g["turn"], g["call"], g["clen"], g["tc"], g["give"], g["retract"],
                 ("msg[%d] %s" % (c[0], c[1])) if c else "-"))

    sim15 = sims_by_key[KEY15][0] if KEY15 in sims_by_key else None
    if sim15 is not None:
        print("\n  reward = %s" % (sim15.get("reward_info") or {}).get("reward"))
        try:
            diff = F.mutation_diff(sim15, mut)
            for k in ("missing", "wrongarg", "extra", "matched"):
                v = diff.get(k) or []
                print("    %-9s %d  %s" % (k.upper(), len(v), [x.get("name") for x in v][:6]))
        except Exception as ex:
            print("    mutation_diff 실패(모른다): %r" % (ex,))
        print("\n  하류 축자 (같은 sim · 커밋 메시지):")
        for i, m in enumerate(sim15["messages"]):
            if not (27 <= i <= 36):
                continue
            tcs = [(t.get("name"), str(t.get("arguments"))[:56]) for t in (m.get("tool_calls") or [])]
            c = (m.get("content") or "").replace("\n", " ").strip()
            print("    msg[%02d] %-9s tc=%s | %s" % (i, m.get("role"), tcs or "-", c[:88]))

    # ── §C ⑶ ────────────────────────────────────────────────────────────────
    hr("C. ⑶ ★수리 전 ↔ 수리 후 — 같은 재료 · 두 팔 (015)")
    p015 = [g for g in g015 if g["tag"] in PREGIVE_TAGS]
    split = 0
    for g in p015:
        dn, rn, why = materials(g, give_name)
        if rn is None:
            print("  t%-4s %-12s 판정불가 (%s)" % (g["turn"], g["tag"], why))
            continue
        a = arm_pre(dn, rn, mut)[0]
        b, lost = arm_post(dn, rn, mut)
        c = arm_post_merge(dn, rn, mut)[0]
        split += (a != b)
        print("  t%-4s %-12s draft_mut=%s  regen=%s  (%s)"
              % (g["turn"], g["tag"], [n for n in dn if n in mut], rn, why))
        print("        PRE=%-10s POST-A=%-14s POST-B=%-15s lost=%s   => %s"
              % (a, b, c, lost, "★갈림" if a != b else "동일"))
    print("\n  015 갈린 이벤트 = %d / %d" % (split, len(p015)))

    # ── §D ⑷ ────────────────────────────────────────────────────────────────
    hr("D. ⑷ [[70]] 무엇을 파나 — 캠페인 pre-give 회수분 전수 (%d 건)" % len(pregive))
    tally, blocked, kept = collections.Counter(), [], []
    for g in pregive:
        dn, rn, why = materials(g, give_name)
        if not dn:
            tally[(g["tag"], "레버라인 미회수")] += 1
            continue
        if rn is None:
            tally[(g["tag"], "UNKNOWN(이름 미회수)")] += 1
            continue
        a, (b, lost) = arm_pre(dn, rn, mut)[0], arm_post(dn, rn, mut)
        if a != b:
            tally[(g["tag"], "갈림 = 교체 차단")] += 1
            blocked.append(g)
        else:
            tally[(g["tag"], "동일 = 교체 유지")] += 1
            kept.append(g)
    for k in sorted(tally):
        print("  %-12s %-24s %d" % (k[0], k[1], tally[k]))

    print("\n  ★수리가 이번 런에서 실제로 바꾸는 이벤트 전수 (%d):" % len(blocked))
    print("    (delay = 그 give 가 궤적에서 실제 실행되기까지 커밋 메시지 수 · never = 끝내 안 함)")
    for g in blocked:
        dn, rn, why = materials(g, give_name)
        print("    %-32s %-18s t%-5s %-12s give=%-28s %-22s delay=%s"
              % (g["run"][len("trace_bank_"):-len(".jsonl.gz")], g["sim"], g["turn"],
                 g["tag"], g["give"], why, give_delay(g, sims_by_key, give_name)))

    # ── 무엇을 사는가 — 지연/소멸의 실측 분포 ────────────────────────────────
    dly_b = [give_delay(g, sims_by_key, give_name) for g in blocked]
    dly_k = [give_delay(g, sims_by_key, give_name) for g in kept]
    num_b = [d for d in dly_b if isinstance(d, int)]
    print("\n  ▸ 사는 것(측정): 교체가 차단될 %d 건에서 give 실행까지의 지연 —"
          " 중앙값 %s · never %d · 미상 %d"
          % (len(blocked), (sorted(num_b)[len(num_b) // 2] if num_b else "-"),
             sum(1 for d in dly_b if d == "never"), sum(1 for d in dly_b if d == "?")))
    num_k = [d for d in dly_k if isinstance(d, int)]
    print("    대조: 교체가 유지될 %d 건(give 보존) 지연 — 중앙값 %s · never %d · 미상 %d"
          % (len(kept), (sorted(num_k)[len(num_k) // 2] if num_k else "-"),
             sum(1 for d in dly_k if d == "never"), sum(1 for d in dly_k if d == "?")))

    gq = [g for g in pregive if g["tag"] == "givequote" and g["retract"] is not None]
    r1 = [g for g in gq if g["retract"] == 1]
    print("\n  ⚠givequote 의 **사전등록 성공지표**는 `retract`(give 철회율)다 —"
          " t2_gate_patch.py 의 P1 주석 축자:")
    print("    \"사전등록 지표: '인용-불성립 후 give 철회율' … ≈0이면 접는다\"")
    print("    회수분: retract=1 %d · retract=0 %d / %d  ⇒ 수리안 A 는 그 %d 건을 **전부** 없앤다"
          % (len(r1), len(gq) - len(r1), len(gq), len(r1)))

    utn = [g for g in pregive if g["tag"] == "usertoolnote"]
    utn_blocked = [g for g in blocked if g["tag"] == "usertoolnote"]
    utn_unk = tally.get(("usertoolnote", "UNKNOWN(이름 미회수)"), 0)
    print("\n  ⚠수리안 A 가 usertoolnote 에 하는 일: 발화 %d 중 판정 가능한 %d 건이 **전부** 교체 차단"
          % (len(utn), len(utn) - utn_unk))
    print("    이 레버의 문면은 **비커밋**이라(재생성 산출이 유일한 흔적) 교체를 기각하면"
          " 레버가 궤적에 남기는 것이 0 이 된다 ⇒ A 는 `T2_USER_TOOL_NOTE` 를 사실상 **끈다**"
          " ([[60]] 저촉). 수리안 B(merge)는 문면을 살리고 호출만 되붙여 이 매도를 0 으로 만든다.")
    print("    usertoolnote 차단 대상 = %d 건 / givequote 차단 대상 = %d 건"
          % (len(utn_blocked), len(blocked) - len(utn_blocked)))

    # ── §E [[57]] ───────────────────────────────────────────────────────────
    hr("E. [[57]] 부정통제")
    nc1 = 0
    for g in pregive:
        dn, rn, _ = materials(g, give_name)
        if rn is None or not dn:
            continue
        if arm_pre(dn, rn, set())[0] != arm_post(dn, rn, set())[0]:
            nc1 += 1
    print("  NC1 되돌리기 — 변이 선언을 ∅ 로 두면(수리의 유일한 항 제거) 갈림 = %d  (기대 0)" % nc1)

    nc2 = sum(1 for g in kept
              if arm_post(*(materials(g, give_name)[:2] + (mut,)))[0] == "KEEP_ORIGINAL")
    print("  NC2 과발화 — give 가 **살아남은** 실물 %d 건에 같은 술어: 차단 = %d  (기대 0)"
          % (len(kept), nc2))

    nod = [g for g in regens if g["draft_tc"] == 0 and g["tag"] not in PREGIVE_TAGS]
    nc3 = sum(1 for _ in nod if arm_post([], [], mut)[0] == "KEEP_ORIGINAL")
    print("  NC3 무관면 — 초안에 호출이 없던 재생성 %d 건: 차단 = %d  (기대 0)" % (len(nod), nc3))

    wide = [g for g in regens if (g["draft_tc"] or 0) >= 1 and g["tc"] == 0]
    print("  NC4 술어 확장 — 변이 제한을 풀고 '호출 손실 전부' 로 넓히면 차단 = %d / 재생성 %d"
          % (len(wide), len(regens)))
    print("      ⇒ 변이-제한 항이 %d 건을 덜 건드린다(= 그 항이 실제로 일한다)"
          % (len(wide) - len(blocked)))
    wt = collections.Counter(g["tag"] for g in wide)
    print("      (확장 팔이 건드릴 채널: %s)" % dict(wt.most_common(8)))

    # ── §G ⑥ 배선 후 재확인 — **엔진 소스 그대로 실행** ─────────────────────
    hr("G. ⑥ 수리 배선 후 재실행 — `t2_gate_patch.py` 의 그 블록을 떼어 **그대로 실행**")
    code, blk = wired_code(src)
    if code is None:
        print("  블록 미발견 — 수리가 아직 배선되지 않았다(§C 는 모형 상태).")
        wired_split = None
    else:
        gstxt = open(gsp, encoding="utf-8").read() if os.path.exists(gsp) else ""
        m3 = re.search(r"^export %s=(\S+)" % WIRED_FLAG, gstxt, re.M)
        print("  [[81]] go_stack.sh: %-24s = %s" % (WIRED_FLAG, m3.group(1) if m3 else "미등재"))
        print("  떼어낸 블록 %d 줄 · 계기 = `[%s] restored=`" % (blk.count("\n") + 1, WIRED_FLAG))
        wired_hits, mismatch = [], []
        for g in pregive:
            dn, rn, _ = materials(g, give_name)
            if rn is None or not dn:
                continue
            names, cnt = wired_run(code, dn, rn, mut)
            model_blocks = (arm_pre(dn, rn, mut)[0] != arm_post(dn, rn, mut)[0])
            if cnt:
                wired_hits.append(g)
            if bool(cnt) != model_blocks:
                mismatch.append((g["run"], g["sim"], g["turn"], cnt, model_blocks))
        print("  G1 배선된 코드가 복원하는 pre-give 이벤트 = %d  (§D 모형의 갈림 %d)"
              % (len(wired_hits), len(blocked)))
        print("     모형 ↔ 배선 불일치 = %d %s" % (len(mismatch), mismatch[:3] if mismatch else ""))
        wired_split = 0
        for g in p015:
            dn, rn, why = materials(g, give_name)
            if rn is None:
                continue
            names, cnt = wired_run(code, dn, rn, mut)
            wired_split += (cnt > 0)
            print("  G2 015 t%-4s %-12s draft=%s regen=%s → 배선 산출=%s restored=%d"
                  % (g["turn"], g["tag"], dn, rn, names, cnt))
        nc_off = sum(wired_run(code, *(materials(g, give_name)[:2]), mutset=mut, flag="0")[1]
                     for g in blocked)
        print("  G3 [[57]] 부정통제 — 같은 %d 건에 플래그 0: 복원 = %d  (기대 0)"
              % (len(blocked), nc_off))
        nc_kept = sum(wired_run(code, *(materials(g, give_name)[:2]), mutset=mut)[1]
                      for g in kept)
        print("  G4 과발화 — give 가 살아남은 %d 건: 복원 = %d  (기대 0)" % (len(kept), nc_kept))
        nc_mut = sum(wired_run(code, *(materials(g, give_name)[:2]), mutset=set())[1]
                     for g in blocked)
        print("  G5 되돌리기 — 변이 선언을 ∅ 로(수리의 유일한 항 제거): 복원 = %d  (기대 0)" % nc_mut)
        print("  ⚠[[70]] 배선판은 **문면을 채택하고 호출만 되붙인다**(수리안 B) ⇒"
              " `T2_USER_TOOL_NOTE` 는 꺼지지 않는다. 대신 `[T2_GIVE_QUOTE] retract=1`"
              " %d 건은 구조적으로 0 이 되고 같은 양을 `restored=` 라인이 싣는다." % len(r1))

    # ── 판정 ────────────────────────────────────────────────────────────────
    hr("F. 판정")
    print("  ⑴ 결함 살아있음  : %s" % alive)
    print("  ⑵ 실패에 닿음    : §B — 커밋 길이 유일 일치 · 하류 env 오류 2회 · 같은 인자 재실행")
    print("  ⑶ 전/후 갈림     : 015 %d / %d 이벤트 (모형)" % (split, len(p015)))
    print("  ⑥ 배선 후 재확인  : %s"
          % ("미배선" if wired_split is None
             else "015 %d / %d 이벤트 — 엔진 소스 블록을 그대로 실행" % (wired_split, len(p015))))
    print("  ⑷ 파는 것        : 회수분 전수 차단 %d 건 (givequote 사전등록 철회 %d · usertoolnote %d)"
          % (len(blocked), len(r1), len(utn_blocked)))
    print("  VERDICT = %s" % ("PROBE-PASS" if (alive and split > 0) else "PROBE-FAIL"))
    print("\n  ⛔이 프로브가 재지 **못한** 것([[77]]): 원본을 유지했을 때 그 give 가 실행되는가는"
          " 오프라인에서 확정 불가다. 가장 가까운 증거는 015 의 **같은 sim 자연대조** —"
          " 같은 형상(post-SIGNATURE unified_regen · clen=0 tc=1)의 t30 초안이"
          " 레버에 안 맞고 msg[35] 로 커밋돼 실행됐다(n=1).")


if __name__ == "__main__":
    main()
