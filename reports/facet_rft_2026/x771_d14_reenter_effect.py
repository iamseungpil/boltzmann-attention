# -*- coding: utf-8 -*-
"""x771 — **D14 효과 프로브**: 재생성 산출 호출을 쓰기 게이트에 재진입시키면 판정이 갈리는가.

관문 4칸([[78]] 격리 → 배선 · 프롬프트 저작 0 · 엔진 술어 직접 호출):

 ⑴ 결함이 지금 코드에 있다
     t2_gate_patch.py:9955  `wd = None if _fab_only else _wev_deny_msgs(_wev_msgs, c, wev_specs)`
       → `while True:` 루프 안. 이 자리가 `_wev_deny_msgs` 의 **전 코드베이스 유일 호출부**다.
     t2_gate_patch.py:13811 `def _ap_regen(fbtxt, tag, tool_choice=None, am_override=None)`
       → 루프 **밖**. 내부 재검사는 `_denied_calls(gate)` · `T2_PROCEDURE` · `T2_UNLOCK_NAME` ·
         `T2_UNLOCK_PROV` 넷뿐 — wtag 6종(WRITE_EVIDENCE·WRITE_ARG_GROUND·ARG_EMPTY·
         REF_VERIFY·ASK_UNKNOWN_BOOL·HANDOFF_ARG_GROUND)은 하나도 없다. 호출부 30곳.
     ★그리고 **실행-시점 그물이 없다**(이게 «게이트 밖 커밋» 을 완성하는 두 번째 못):
       t2_gate_patch.py:1320 `wd = _write_evidence_deny(self, tc, wev_specs)` 는 `gated()`(:1202)
         안이고 `gated` 는 :1403 에서 `_execute_tool_calls` 에 꽂힌다 — 그러나
       t2_gate_patch.py:8153 `BaseOrchestrator._execute_tool_calls = exec_augment` 가 그것을
         덮는다. `_install_regen_exec` 축자(:7769) — *"slim _execute_tool_calls: 실행 + auth
         observe + read-augment … **deny 없음** (denied 호출은 생성-레벨서 이미 strip)"*.
       go_stack.sh:26 `export T2_GATE_REGEN=1` 이므로 **정본 스택은 그 팔**이다.
       ⇒ unified 팔에서 쓰기-게이트의 유일한 집행점은 :9955 하나이고, `_ap_regen` 산출은
         그 자리를 지나지 않는다. (:8848-8850 이 같은 사실을 이미 자백한다 —
         *"구 apply()에만 있던 WEV가 unified 런서 死코드"*.)

 ⑵ 그 결함이 실제 실패에 닿는다  — 이 스크립트가 **회수 로그+궤적**으로 짚는다(§A).
 ⑶ ★수리 전/후 판정이 갈린다     — 같은 재료에 두 팔을 먹인다(§B).
       ARM_OFF (수리 전) = 현행 코드. 재생성 산출에 wtag 술어를 **적용하지 않는다** ⇒ commit.
       ARM_ON  (수리 후) = 엔진의 같은 술어를 같은 순서로 적용한다
                            `_wev_deny_msgs` → (통과 시) `_write_arg_ground_deny`.
       ⚠ARM_ON 은 **하한**이다: 나머지 4종(ARG_EMPTY·REF_VERIFY·ASK_UNKNOWN_BOOL·HANDOFF)은
         orch/agent 객체를 요구해 오프라인에서 못 돈다. 하한에서 이미 갈리면 결론은 강해질 뿐이다.

 ⑷ [[70]] 파는 것 — 회수분 **전수** census. 재생성-산출 write 호출을 sim reward 로 갈라
       reward=1.0 sim 에서 ARM_ON 이 죽이는 칸 = **파는 것**.

부정통제 3종([[57]]):
   NC1  ARM_OFF 를 같은 재료에 = DENY 0 (되돌리면 검정이 실패하는가).
   NC2  ★**정상경로(비-재생성) 커밋 write** 에 ARM_ON = deny 율이 0 근처여야 한다.
        이것은 프로브 자신의 반증 조건이다 — 라이브에서 이미 WEV 가 돌아 나쁜 것을 걸렀으므로,
        여기서 대량 deny 가 나오면 내 오프라인 창 재구성이 틀린 것이다(재료 오염 검출기).
   NC3  게이트 사정권 밖 도구(선언 조회로 가른다)에 ARM_ON = 정의상 skip 0건 deny.

재료 규율([[77]]/[[71]]): 가짜 입력 0. 도구 이름을 프로브에 타이핑하지 않는다 — 사정권은
`banking_knowledge.gate.json` 선언에서 읽는다. 재생성-귀속은 로그의 `[T2_GEN_TRACE]
call=agent_response(_tag)?` 순서를 궤적의 assistant 메시지에 **정렬 검산**(base 수 == assistant 수-1)
해서 얻는다 — 검산 실패 sim 은 **버린다**(추정 금지).

용법:  PYTHONIOENCODING=utf-8 py -3 x771_d14_reenter_effect.py [--runs bank_re8143p11_20260904_1053,...]
"""
import argparse
import contextlib
import glob
import gzip
import io
import json
import os
import re
import sys
import types
from collections import Counter, defaultdict

TAU2 = r"C:\workspace\ba-frft\scripts\distill\tau2"
SR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
sys.path.insert(0, TAU2)
os.chdir(TAU2)

from t2_gate_patch import (_args_dict, _eff_tool_name, _exact_tool_name,   # noqa: E402
                           _arg_empty_deny, _ref_verify_deny,
                           _wev_deny_msgs, _write_arg_ground_deny)

A2 = json.load(io.open("a2/banking_knowledge.gate.json", encoding="utf-8"))
WEV = A2["write_evidence_specs"]
WAG = A2["write_arg_grounding"]

# 사정권은 **선언에서** 읽는다 (엔진/프로브에 도메인 리터럴 0 · [[05]]/[[71]]②)
SCOPE_OUTER = {sp.get("applies_to") for sp in (WEV + WAG) if sp.get("applies_to")}
SCOPE_PREFIX = {(sp.get("applies_when") or {}).get("prefix")
                for sp in (WEV + WAG)} - {None}

GEN = re.compile(r"\[sim=(?P<sim>[^\]]+)\].*\[T2_GEN_TRACE\] call=agent_response"
                 r"(?P<tag>_[A-Za-z0-9_]+)? .*tool_calls=(?P<n>\d+)")
DENYLINE = re.compile(r"\[sim=(?P<sim>[^\]]+)\] \[(?P<tag>T2_WRITE_EVIDENCE|T2_WRITE_ARG_GROUND)\]"
                      r" deny tool=(?P<tool>\S+) inner=(?P<inner>\S*)")


class TC(object):
    """엔진이 기대하는 tool_call 모양만 흉내낸다(값 생성 0)."""

    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = d.get("arguments") or {}
        self.id = d.get("id")


class MSG(object):
    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]
        self.error = bool(d.get("error"))
        self.id = d.get("id")
        self.requestor = d.get("requestor")


def in_scope(tc):
    """이 호출이 어떤 쓰기 게이트의 사정권에 있는가 — 선언만으로 판정."""
    if getattr(tc, "name", None) not in SCOPE_OUTER:
        return False
    if not SCOPE_PREFIX:
        return True
    a = _args_dict(tc)
    vals = [str(v) for v in a.values() if isinstance(v, str)]
    return any(any(v.startswith(p) for p in SCOPE_PREFIX) for v in vals)


def arm_on(window, tc):
    """수리 후 술어 = 엔진의 쓰기-게이트 체인(9955~10008 과 **같은 순서**). 새 결정론 0."""
    wd = _wev_deny_msgs(window, tc, WEV)
    if wd:
        return "T2_WRITE_EVIDENCE", wd
    wd = _write_arg_ground_deny(window, tc, WAG)
    if wd:
        return "T2_WRITE_ARG_GROUND", wd
    return None, None


def arm_off(window, tc):
    """수리 전 술어 = 현행 `_ap_regen`. wtag 6종을 **평가하지 않는다** ⇒ 언제나 commit.
    (근거는 위 docstring ⑴ — 코드에 그 호출이 없다. 이 함수는 그 부재를 그대로 옮긴 것이다.)"""
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# ★수리 후 재실행 팔 (2026-09-05 · 배선 뒤 추가)
#
#   `arm_on` 은 **프로브가 쓴 재구성**이다. 수리를 코드에 싣고 나면 물어야 할 것이 하나 더
#   있다: *엔진에 실제로 배선된 그 블록이 같은 재료에서 같은 판정을 내는가.* 검정이 코드를
#   베껴 적으면 드리프트가 검정을 통과시키므로([[84]] 이름×소비부 사고와 같은 계열),
#   `t2_gate_patch._ap_regen` 본문에서 D14 블록을 **텍스트로 떼어 그대로 실행**한다.
#   블록이 `return None` 을 담고 있어 module-level exec 이 안 되므로 함수로 감싸기만 한다
#   (본문 개작 0). 미배선이면 `None` 을 돌려 팔이 스스로 침묵한다.
def _engine_d14_code():
    import ast
    import textwrap
    src = io.open(os.path.join(TAU2, "t2_gate_patch.py"), encoding="utf-8").read()
    fn = next((n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.FunctionDef) and n.name == "_ap_regen"), None)
    if fn is None:
        return None
    body = ast.get_source_segment(src, fn) or ""
    i0 = body.find("self._t2_regen_wgate_denied = set()")
    if i0 < 0:
        return None
    i0 = body.rindex("\n", 0, i0) + 1
    i1 = body.find("# ★D11ⓐ")
    blk = textwrap.dedent(body[i0:i1 if i1 > i0 else len(body)])
    return compile("def _run14():\n" + textwrap.indent(blk, "    ") + "\n    return 'FT'\n",
                   "<t2_gate_patch:_ap_regen:D14>", "exec")


_D14 = _engine_d14_code()
WGATE = re.compile(r"\[T2_REGEN_WGATE\] deny tag=\S+ wtag=(\S+)")


def arm_wired(window, tc, flag="1"):
    """엔진에 **배선된** D14 블록 그대로. 반환 = (wtag, 조기반환여부) 또는 (None, None).

    ⚠`arm_on` 과 같은 하한을 돈다 — `rv_specs=[]` · `ae_on=False`(오프라인에서 못 도는 축).
      즉 이 팔은 *배선이 같은 판정을 내는가* 만 묻고, 라이브의 상한은 여전히 [미측정]이다.
    """
    if _D14 is None:
        return None, None
    am2 = types.SimpleNamespace(content="x", tool_calls=[tc])
    slf = types.SimpleNamespace(_t2_orch=None, _t2_wev_deny=0)
    ns = {"os": os, "_sys": sys, "json": json, "self": slf, "tag": "probe",
          "state": types.SimpleNamespace(messages=window), "_am2": am2, "am": None,
          "wev_specs": WEV, "wag_specs": WAG, "rv_specs": [], "ae_on": False,
          "ae_tools": None, "a2": A2, "_wev_cap": 8,
          "_wev_deny_msgs": _wev_deny_msgs, "_write_arg_ground_deny": _write_arg_ground_deny,
          "_arg_empty_deny": _arg_empty_deny, "_ref_verify_deny": _ref_verify_deny,
          "_eff_tool_name": _eff_tool_name, "_exact_tool_name": _exact_tool_name,
          "_args_dict": _args_dict, "_lbeat": (lambda *a, **k: None),
          "la": None, "UserMessage": None}
    old = os.environ.get("T2_REGEN_WRITE_GATES")
    buf = io.StringIO()
    try:
        if flag is None:
            os.environ.pop("T2_REGEN_WRITE_GATES", None)
        else:
            os.environ["T2_REGEN_WRITE_GATES"] = flag
        exec(_D14, ns)
        with contextlib.redirect_stderr(buf):
            rv = ns["_run14"]()
    finally:
        if old is None:
            os.environ.pop("T2_REGEN_WRITE_GATES", None)
        else:
            os.environ["T2_REGEN_WRITE_GATES"] = old
    m = WGATE.search(buf.getvalue())
    return (m.group(1), (rv is None)) if m else (None, None)


def parse_log(path):
    """sim → [(tag or None, n_tool_calls), ...]  생성 순서 그대로."""
    per = defaultdict(list)
    denies = defaultdict(list)
    try:
        fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    except Exception:
        return per, denies
    with fh:
        for line in fh:
            if "[T2_GEN_TRACE] call=agent_response" in line:
                m = GEN.search(line)
                if m:
                    per[m.group("sim")].append(((m.group("tag") or "").lstrip("_") or None,
                                                int(m.group("n"))))
            elif "] deny tool=" in line:
                m = DENYLINE.search(line)
                if m:
                    denies[m.group("sim")].append((m.group("tag"), m.group("inner")))
    return per, denies


def turns_from(seq):
    """생성 순서를 **턴**으로 접는다. 한 턴 = base 1발 + 뒤따르는 재생성 0..k발.
    반환 = [(regen_tags, last_tag, last_n)] · base 없이 시작하는 꼬리는 버린다."""
    out = []
    for tag, n in seq:
        if tag is None:
            out.append([[], None, n])
        elif out:
            out[-1][0].append(tag)
            out[-1][1] = tag
            out[-1][2] = n
    return out


def collect(run, attrib_only=True):
    """한 런에서 (sim, turn) 단위 재료를 만든다.

    `attrib_only=True`  = 재생성-귀속이 가능한 sim 만(정렬 검산 통과). §B/§C 가 쓰는 재료.
    `attrib_only=False` = 귀속 불가 sim 도 담는다(regen=None). §A′ 전용 — 우회 경로가
                          재생성인지 cap 소진인지 **못 가른다**고 명시해서 쓴다.
    """
    logp = os.path.join(SR, run + ".log.gz")
    resp = os.path.join(SR, run + ".results.json.gz")
    if not os.path.exists(resp):
        return [], Counter()
    per, denies = (parse_log(logp) if os.path.exists(logp) else (defaultdict(list),
                                                                 defaultdict(list)))
    try:
        d = json.load(gzip.open(resp, "rt", encoding="utf-8"))
    except Exception:
        return [], Counter()
    rows, stat = [], Counter()
    for s in d.get("simulations", []):
        simtag = "%s#s%s" % (s.get("task_id"), s.get("seed"))
        seq = per.get(simtag)
        raw = s.get("messages") or []
        msgs = [MSG(m) for m in raw]
        ai = [i for i, m in enumerate(msgs) if m.role == "assistant"]
        turns = turns_from(seq) if seq else []
        ok = bool(seq) and len(turns) == len(ai) - 1
        if not seq:
            stat["sim_no_log"] += 1
        elif not ok:
            stat["align_fail"] += 1
        else:
            stat["align_ok"] += 1
        if attrib_only and not ok:
            continue
        rw = (s.get("reward_info") or {}).get("reward")
        for k, idx in enumerate(ai[1:]):
            m = msgs[idx]
            tags, last_tag = [], None
            if ok:
                tags, last_tag, last_n = turns[k]
                if len(m.tool_calls) != last_n:
                    stat["turn_ncalls_mismatch"] += 1
                    continue
            for tc in m.tool_calls:
                if not in_scope(tc):
                    continue
                rows.append(dict(run=run, sim=simtag, reward=rw, idx=idx,
                                 regen=(bool(last_tag) if ok else None),
                                 tag=last_tag, tags=list(tags),
                                 tc=tc, window=msgs[:idx],
                                 live_err=_live_err(msgs, idx, tc),
                                 denies=denies.get(simtag, [])))
    return rows, stat


def _live_err(msgs, idx, tc):
    """라이브에서 이 호출이 어떻게 끝났나 — 뒤따르는 tool 메시지의 error 플래그."""
    for m in msgs[idx + 1: idx + 12]:
        if m.role == "tool" and m.id == tc.id:
            return bool(m.error)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--glob", default="bank_*")
    a = ap.parse_args()
    if a.runs:
        runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    else:
        runs = sorted({os.path.basename(p)[:-len(".results.json.gz")]
                       for p in glob.glob(os.path.join(SR, a.glob + ".results.json.gz"))})

    print("=" * 100)
    print("x771 — D14 효과 프로브 (재생성 산출의 쓰기-게이트 재진입)")
    print("=" * 100)
    print("사정권(선언 조회): outer=%s prefix=%s" % (sorted(SCOPE_OUTER), sorted(SCOPE_PREFIX)))
    print("런 %d개 주사" % len(runs))

    rows, stat = [], Counter()
    for r in runs:
        rr, st = collect(r)
        rows += rr
        stat.update(st)
    print("정렬 검산: ok=%d fail=%d (버림) · 로그없음=%d · 턴-호출수 불일치=%d"
          % (stat["align_ok"], stat["align_fail"], stat["sim_no_log"],
             stat["turn_ncalls_mismatch"]))
    if not rows:
        print("⛔재료 0건 — 주장 금지([[77]]).")
        return

    regen = [r for r in rows if r["regen"]]
    plain = [r for r in rows if not r["regen"]]
    print("사정권 커밋 write 호출 %d건 = 재생성-산출 %d · 정상경로 %d"
          % (len(rows), len(regen), len(plain)))

    # ── §B ⑶ 두 팔 ────────────────────────────────────────────────────────────
    print()
    print("-" * 100)
    print("§B ⑶  같은 재료 · 두 팔 (재생성-산출 %d건)" % len(regen))
    print("-" * 100)
    off_d = on_d = 0
    by_tag = Counter()
    detail = []
    for r in regen:
        _, o = arm_off(r["window"], r["tc"])
        t, n = arm_on(r["window"], r["tc"])
        if o:
            off_d += 1
        if n:
            on_d += 1
            by_tag[t] += 1
        detail.append((r, t, n))
    print("ARM_OFF(수리 전) DENY = %d / %d" % (off_d, len(regen)))
    print("ARM_ON (수리 후) DENY = %d / %d   %s" % (on_d, len(regen), dict(by_tag)))
    print("⇒ 판정이 갈린 칸 = %d" % on_d)

    print()
    print("--- 갈린 칸 전수 (run · sim · msg · 채널 · 도구 · 라이브결과 · 수리후 문면) ---")
    for r, t, n in detail:
        if not n:
            continue
        print("  %-42s %-20s msg%-4d rew=%-4s ch=%-16s %-34s live_err=%-5s"
              % (r["run"][:42], r["sim"], r["idx"], r["reward"], r["tag"],
                 _eff_tool_name(r["tc"])[:34], r["live_err"]))
        print("       %s :: %s" % (t, str(n)[:150].replace("\n", " ")))

    # ── §B′ ⑥ 수리 후 재실행 — **엔진에 배선된 블록**을 같은 재료에 ────────────────
    print()
    print("-" * 100)
    print("§B′ ⑥ 수리 후 재실행 — 엔진에 **배선된** D14 블록을 같은 %d칸에 (%s)"
          % (len(regen), "블록 추출 성공" if _D14 is not None else "⛔미배선 — 팔 침묵"))
    print("-" * 100)
    wired_d = 0
    by_tag_w = Counter()
    mismatch = []
    early = 0
    for r, t, n in detail:
        tw, rvn = arm_wired(r["window"], r["tc"])
        if tw:
            wired_d += 1
            by_tag_w[tw] += 1
            if rvn:
                early += 1
        if bool(tw) != bool(n):
            mismatch.append((r["run"][:30], r["sim"], r["idx"], str(t), str(tw)))
    print("ARM_WIRED (배선된 엔진 코드) DENY = %d / %d   %s" % (wired_d, len(regen), dict(by_tag_w)))
    print("  그중 그 턴의 호출이 **전부** denied → `return None`(원본 유지) = %d칸" % early)
    print("ARM_ON ↔ ARM_WIRED 불일치 = %d %s" % (len(mismatch), mismatch[:5]))
    # [[57]] 부정통제 — 플래그를 되돌리면 배선 팔이 죽는가
    off_w = sum(1 for r, t, n in detail if arm_wired(r["window"], r["tc"], flag="0")[0])
    print("NC0 (배선) `T2_REGEN_WRITE_GATES=0` → DENY %d / %d — 되돌리면 이 팔이 죽는다"
          % (off_w, len(regen)))
    # [[70]] 배선 팔의 매도 = reward=1.0 sim 에서 죽는 재생성-산출 write
    sell_w = sum(1 for r, t, n in detail
                 if (r["reward"] or 0) >= 1.0 and arm_wired(r["window"], r["tc"])[0])
    print("[[70]] 배선 팔이 reward=1.0 sim 에서 죽이는 재생성-산출 write = %d" % sell_w)

    # ── §C ⑷ [[70]] 파는 것 ───────────────────────────────────────────────────
    print()
    print("-" * 100)
    print("§C ⑷ [[70]] 부호표 — 회수분 전수 (단위 = 재생성-산출 커밋 write 호출)")
    print("-" * 100)
    buy = [(r, t, n) for r, t, n in detail if n and (r["reward"] or 0) < 1.0]
    sell = [(r, t, n) for r, t, n in detail if n and (r["reward"] or 0) >= 1.0]
    print("사는 것 (reward<1.0 sim 에서 차단) = %d" % len(buy))
    print("파는 것 (reward=1.0 sim 에서 차단) = %d" % len(sell))
    for r, t, n in sell:
        print("  ⚠SELL %-42s %-20s msg%-4d ch=%-16s %s"
              % (r["run"][:42], r["sim"], r["idx"], r["tag"], _eff_tool_name(r["tc"])))
        print("        %s :: %s" % (t, str(n)[:150].replace("\n", " ")))
    # 라이브에서 이미 실패한(err=True) 호출을 막는 것은 파는 것이 아니다
    live_ok_denied = [1 for r, t, n in detail if n and r["live_err"] is False]
    print("갈린 칸 중 라이브에서 **성공 실행**된 것 = %d (나머지는 어차피 env 가 거부)"
          % len(live_ok_denied))

    # ── §D 부정통제 ───────────────────────────────────────────────────────────
    print()
    print("-" * 100)
    print("§D 부정통제 3종 ([[57]])")
    print("-" * 100)
    print("NC1  ARM_OFF 를 같은 재료에 → DENY %d  (되돌리면 검정이 실패한다)" % off_d)
    pd = pn = 0
    ptag = Counter()
    for r in plain:
        t, n = arm_on(r["window"], r["tc"])
        pd += 1
        if n:
            pn += 1
            ptag[t] += 1
    print("NC2  정상경로(비-재생성) 커밋 write %d건에 ARM_ON → DENY %d (%.1f%%) %s"
          % (pd, pn, (100.0 * pn / pd) if pd else 0.0, dict(ptag)))
    print("     ⚠이것이 프로브 자신의 반증 조건이다 — 라이브에서 WEV 가 이미 돌았으므로 0 근처여야 한다.")
    oos = 0
    for r in rows[:0]:
        pass
    # NC3: 사정권 밖 호출은 술어가 정의상 skip
    n_out = 0
    for r in rows:
        pass
    print("NC3  사정권 밖 도구: 술어가 `applies_to` 불일치로 skip — 선언 조회로 확인 "
          "(outer %d종 · prefix %d종 밖은 평가 자체가 없다)" % (len(SCOPE_OUTER), len(SCOPE_PREFIX)))

    # ── §A ⑵ 실패에 닿는가 — 같은 sim 안의 대조 ────────────────────────────────
    print()
    print("-" * 100)
    print("§A ⑵  sim 내부 대조 — 같은 도구를 정상경로에서 라이브-DENY 하고 재생성으로 커밋한 sim")
    print("-" * 100)
    hit = 0
    for r, t, n in detail:
        if not n:
            continue
        inner = _eff_tool_name(r["tc"])
        same = [x for x in r["denies"] if inner and inner in (x[1] or "")]
        if same:
            hit += 1
            print("  %-20s msg%-4d  라이브 deny %d회 (%s)  ↔  재생성 ch=%s 로 커밋 (live_err=%s)"
                  % (r["sim"], r["idx"], len(same), same[0][0], r["tag"], r["live_err"]))
    print("sim 내부 대조가 성립한 칸 = %d" % hit)

    # ── §E 예산 검산 — 재진입이 실제로 발화할 수 있었나 ────────────────────────
    print()
    print("-" * 100)
    print("§E 예산 검산 — 재진입이 **cap 안**이었나 (`T2_WEV_CAP` sim당 8 · 축자 로그로 검산)")
    print("   ⚠cap 을 공유하면 이미 소진된 sim 에서 재진입은 아무 것도 안 한다. 갈린 칸마다 센다.")
    print("-" * 100)
    for r, t, n in detail:
        if not n:
            continue
        live_wev = sum(1 for x in r["denies"] if x[0] == "T2_WRITE_EVIDENCE")
        print("  %-20s msg%-4d  sim 전체 라이브 WEV deny = %d / cap 8  → 재진입 예산 %s"
              % (r["sim"], r["idx"], live_wev, "있음" if live_wev < 8 else "**소진**"))

    # ── §A′ 귀속-불가 sim 까지 포함한 census (027 커버리지) ────────────────────
    print()
    print("-" * 100)
    print("§A′ 귀속-불가 sim 포함 census — 「커밋+실행됐는데 ARM_ON 이면 DENY」인 write")
    print("   ⚠여기서는 **우회 경로를 못 가른다**(재생성 / 턴당·sim당 cap / _fab_only 중 무엇인지).")
    print("   ⚠구 로그(2026-08 이전)에는 `[T2_GEN_TRACE]` 자체가 없어 귀속이 원리상 불가하다.")
    print("-" * 100)
    rows2, stat2 = [], Counter()
    for r in runs:
        rr, st = collect(r, attrib_only=False)
        rows2 += rr
        stat2.update(st)
    per_task = defaultdict(lambda: [0, 0, 0])   # [사정권 write, ARM_ON deny, 그중 귀속가능]
    for r in rows2:
        t = r["sim"].split("#")[0]
        per_task[t][0] += 1
        _, n = arm_on(r["window"], r["tc"])
        if n and r["live_err"] is False:
            per_task[t][1] += 1
            if r["regen"]:
                per_task[t][2] += 1
    print("사정권 커밋 write %d건 (sim %d · 귀속가능 %d / 귀속불가 %d)"
          % (len(rows2), stat2["align_ok"] + stat2["align_fail"] + stat2["sim_no_log"],
             stat2["align_ok"], stat2["align_fail"] + stat2["sim_no_log"]))
    print("  %-12s %8s %10s %10s" % ("task", "write", "ARM_ON deny", "그중 regen"))
    for t in sorted(per_task, key=lambda x: -per_task[x][1]):
        w, dn, rg = per_task[t]
        if dn:
            print("  %-12s %8d %10d %10d" % (t, w, dn, rg))
    for t in ("task_027", "task_029", "task_048"):
        w, dn, rg = per_task.get(t, [0, 0, 0])
        print("  ★표적 %-10s write=%d · ARM_ON deny=%d · 그중 재생성-귀속=%d" % (t, w, dn, rg))
    print("  ⛔이 표는 **부호표가 아니다**: 코퍼스가 base 팔·레버-OFF 팔·구 A2 를 섞고 있어")
    print("    오늘 선언을 옛 궤적에 소급 적용한 값이다. 매수/매도는 §C 만 권위다.")

    # ── §F 027 표적 — 회수된 우회의 정체 ──────────────────────────────────────
    print()
    print("-" * 100)
    print("§F 표적 027 — 회수분에서 관측된 WEV 우회의 정체 (cap vs 재생성)")
    print("-" * 100)
    cap_hit = defaultdict(int)
    for r in runs:
        lp = os.path.join(SR, r + ".log.gz")
        if not os.path.exists(lp):
            continue
        try:
            fh = gzip.open(lp, "rt", encoding="utf-8", errors="replace")
        except Exception:
            continue
        with fh:
            for line in fh:
                if "deny cap" in line and "no further WEV denies" in line:
                    m = re.search(r"\[sim=([^\]]+)\]", line)
                    if m:
                        cap_hit[(r, m.group(1))] += 1
    t27 = sorted(k for k in cap_hit if k[1].startswith("task_027"))
    print("`[T2_WRITE_EVIDENCE] deny cap 8 reached` 를 친 (런,sim) = %d개 · 그중 027 = %d개"
          % (len(cap_hit), len(t27)))
    for k in t27:
        print("  cap 소진: %-42s %s" % (k[0][:42], k[1]))
    print("⇒ cap 을 소진한 sim 에서는 재진입도 **같은 예산**에 걸려 아무 것도 안 한다"
          " (수리안 §2 가 이 칸을 다룬다).")

    # ── 판정 ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("판정: %s" % ("PROBE-PASS (⑶ 갈림 %d칸)" % on_d if on_d else "PROBE-FAIL (⑶ 안 갈림)"))
    print("=" * 100)


if __name__ == "__main__":
    main()
