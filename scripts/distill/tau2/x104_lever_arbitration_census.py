# -*- coding: utf-8 -*-
"""What did our own layer say, when did two of our messages collide, and which of them
would a phase- or condition-gate actually silence?

`x95` classifies the strings we *can* send and pairs opposed classes. It missed the pair that
cost `task_022`: [VALUE-ACQUIRE] "give `get_card_last_4_digits` to the customer NOW" arriving in
the same turn as [PROCEDURE] "the policy forbids `get_card_last_4_digits` in this procedure".
The pair is invisible statically because the two strings do not disagree in vocabulary — they
disagree about one tool name, and only at run time.

So this reads the sidecar (what we actually sent, per simulation and turn) together with the
trajectories (what had been executed by then) and reports four things, all mechanical:

  A  census      per lever: firings, and the split over passing/failing simulations
  B  co-firing   lever pairs that arrived in the same (simulation, turn)
  C  prohibition for every firing that names a tool, whether an active procedure's
                 declaration forbids that tool — the target of the speak-time condition
  D  phase       the phase `t2_phase.phase_of` assigns at each firing turn, under the wiring
                 as it exists and under the two branches the module declares but never returns

D is the number the design needs before any lever declares an owning phase: a lever whose
declared phase is never reached is a lever that has been switched off.

  usage: x104_lever_arbitration_census.py <tag>      e.g. 20260806a
"""

import collections
import glob
import gzip
import hashlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import gate_interpreter as GI                                   # noqa: E402
import t2_phase as PH                                           # noqa: E402
import t2_procedure as PR                                       # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260806a"
DOMAIN = os.environ.get("X104_DOMAIN", "banking_knowledge")
SIM_REMOTE = ("/home/woori/workspace_common/boltzmann-attention-pi/"
              "reports/facet_rft_2026/sim_results")
SIM_LOCAL = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                         "reports", "facet_rft_2026", "sim_results"))
SIM = SIM_REMOTE if os.path.isdir(SIM_REMOTE) else SIM_LOCAL

# ── 계기 규약 ────────────────────────────────────────────────────────────────
# 사이드카 `kind`는 채널의 종류다. `reminder-assistant`는 **모델이 재생성한 답변**이지
# 우리가 보낸 지시가 아니다 — 노출을 셀 때 이걸 섞으면 우리 문구 수가 부풀어 재현이 깨진다
# (설계서 §3의 재현 불일치가 이 정의 누락에서 나왔다).
OURS = ("reminder-user", "tool-deny")


# sim 식별자는 처음부터 표시형 문자열 `task_048 t1`로 만든다 — 튜플로 두면 다운스트림
# (`rew`/`sims` 조회·출력 포맷)이 조용히 어긋난다.

# 문구의 레버 정체는 지금 사이드카에 **없다**(채널만 있다). 태그를 본문 앞머리에서 읽는 것은
# 임시방편이고, 그 취약성 자체가 설계서 E3의 "발화에 (레버, 표적) 메타데이터를 실어라"의 근거다.
TAGRE = re.compile(r"\[([A-Z][A-Z0-9_\- ]{2,30})\]")

# ★표적 추출은 **닫힌 집합 대조**로 한다. 초판은 따옴표/백틱에 의존했는데, VALUE-ACQUIRE 문구는
#   도구명을 맨몸으로 쓴다("…running get_card_last_4_digits") — 그래서 표적을 놓치고 대신 그 도구를
#   인용한 PROCEDURE deny 자신을 잡았다. 계기가 표적을 반대로 지목하면 처방도 반대가 된다
#   ([[55]] 계기는 부정통제 없이 신뢰 금지). 이름은 env 레지스트리에서 읽는다([[22]] 닫힌 술어).
def tool_universe(domain):
    p = os.path.join(HERE, "a2", "env_surface.json")
    d = (json.load(open(p, encoding="utf-8")) or {}).get(domain) or {}
    names = set(d.get("tools") or {}) | set(d.get("exposed") or []) \
        | set(d.get("discoverable_user_tools") or [])
    return {n for n in names if isinstance(n, str) and len(n) > 4}


# 레버의 **표적 출처** 분류(설계서 §4.2). push = 표적을 우리가 골랐다 / react = 모델의 호출에
# 대한 판정이라 표적이 이미 모델의 것. 금지 조건은 push에만 건다 — react에 걸면 방금 그 금지를
# 집행한 문장 자신을 지운다(초판 계기가 정확히 그 오류를 냈다).
PUSH = {"ACTION", "ACTION-REQUIRED", "VALUE-ACQUIRE", "DISCOVERY-REQUIRED",
        "FOLLOW-UP", "UNLOCKED-NOT-CALLED", "SEARCH-EXHAUST", "E-PLAN"}
REACT = {"PROCEDURE", "PROCEDURE-INCOMPLETE", "SIGNATURE", "TOOL-CHANNEL", "WRITE-EVIDENCE",
         "WRITE-GROUNDING", "CLAIM-PROVENANCE", "PROVENANCE", "OPERATOR-PROVENANCE",
         "VERDICT", "PROTOCOL", "VERIFY-PERSISTENCE", "TRANSFER-REASON", "GIVE-EXEC",
         "TOOL-CALL ENVELOPE"}


def lever_of(row):
    m = TAGRE.search((row.get("text") or "")[:120])
    return m.group(1) if m else row.get("channel") or "?"


def targets_of(row, universe):
    """문구가 이름을 댄 도구들 = 본문 ∩ 레지스트리. 표적 메타데이터가 없어서 본문을 읽는다(위 주석)."""
    txt = row.get("text") or ""
    return sorted(n for n in universe if re.search(r"(?<![a-z_0-9])%s(?![a-z_0-9])" % re.escape(n), txt))


class _M(object):
    """엔진 함수는 메시지 객체를 기대하고 저장본은 dict다 — 최소 어댑터."""

    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [_TC(t) for t in (d.get("tool_calls") or [])]
        self.requestor = d.get("requestor")


class _TC(object):
    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = d.get("arguments")


def eff_name(tc):
    a = getattr(tc, "arguments", None)
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    if isinstance(a, dict):
        for k in ("agent_tool_name", "discoverable_tool_name"):
            if a.get(k):
                return str(a[k])
    return getattr(tc, "name", None)


def executed_ok(msgs):
    """오류 없이 돌아온 호출만. 도구 응답이 'Error:'로 시작하면 실행되지 않은 것으로 본다."""
    out, pending = set(), []
    for m in msgs:
        for tc in (m.tool_calls or []):
            pending.append(eff_name(tc))
        if m.role == "tool":
            txt = str(m.content or "")
            if pending:
                nm = pending.pop(0)
                if nm and not txt.strip().startswith("Error"):
                    out.add(nm)
    return out


def procedures_state(a2, executed):
    """(unmet, decision_ready) — 새 판정이 아니라 기존 함수 조합(설계서 E2)."""
    unmet = decision = False
    for p in PR.active_procedures((a2 or {}).get("procedures") or [], executed):
        cands, uniq = PR.next_step(p, executed)
        if cands:
            unmet = True
            if uniq:
                decision = True
    return unmet, decision


def discover_pending(a2, executed, msgs):
    """`discover` 분기의 후보 정의 — 선언이 이름을 댄 단계 도구 중 아직 unlock되지 않은 것.

    t2_phase의 docstring은 이 단계를 선언하고도 반환하지 않는다. 여기서는 **가설 구현**을 놓고
    분포만 잰다(엔진 거동은 바꾸지 않는다). 켜기 전에 얼마나 넓은지 보는 것이 목적이다.
    """
    unlocked = set()
    for m in msgs:
        for tc in (m.tool_calls or []):
            if getattr(tc, "name", None) in ("unlock_discoverable_agent_tool",
                                             "give_discoverable_user_tool"):
                unlocked.add(eff_name(tc))
    named = set()
    for p in PR.active_procedures((a2 or {}).get("procedures") or [], executed):
        for n in (p.get("nodes") or []):
            named |= set(PR._tools_of(n))          # 노드 스키마는 t2_procedure가 안다(tool/tool_any)
    return bool(named - unlocked - set(executed))


def main():
    a2 = GI.load_domain_a2(DOMAIN) or {}
    procs = a2.get("procedures") or []

    # ★전수 런 지원 (2026-08-06): 이 도구는 스모크(`bank_smk_*`·nt=1)만 읽었고 sim을 **task_id로**
    #   키잉했다 — nt2 전수에 그대로 돌리면 **두 trial이 겹쳐** 발화가 한쪽으로 접힌다.
    #   그래서 ①파일 glob에 `bank_n97_*`를 추가하고 ②키를 (task_id, trial)로 바꾼다.
    #   사이드카도 전수 드라이버는 `fb_n97_gpu<G>_<TAG>.jsonl`로 **GPU별로** 쓴다(한 파일 아님).
    sims, key, rew = {}, {}, {}
    pats = [os.path.join(SIM, "bank_smk_gpu*_%s.results.json.gz" % TAG),
            os.path.join(SIM, "bank_n97_gpu*%s.results.json.gz" % TAG)]
    dup = 0
    for p in sorted(set(sum([glob.glob(x) for x in pats], []))):
        d = json.load(gzip.open(p, "rt", encoding="utf-8"))
        for s in d["simulations"]:
            msgs = [_M(m) for m in s["messages"]]
            ident = "%s t%s" % (s["task_id"], s.get("trial"))
            sims[ident] = msgs
            rew[ident] = s["reward_info"]["reward"]
            for m in msgs:
                if m.role == "user" and isinstance(m.content, str) and m.content.strip():
                    k = hashlib.sha1(m.content.strip().encode("utf-8")).hexdigest()[:12]
                    if k in key:
                        dup += 1          # 첫 발화가 같은 두 sim = 키 충돌(아래에서 보고)
                    key[k] = ident
                    break
    if not sims:
        print("결과 파일 없음: %s (tag=%s)" % (SIM, TAG))
        return

    fbs = [os.path.join(SIM, "fb_%s.jsonl.gz" % TAG)] +         sorted(glob.glob("/home/woori/scratch/logs/fb_n97_gpu*_%s.jsonl" % TAG))
    rows = []
    for fb in fbs:
        if not os.path.exists(fb):
            continue
        op = gzip.open if fb.endswith(".gz") else io.open
        rows += [json.loads(l) for l in op(fb, "rt", encoding="utf-8") if l.strip()]
    ours = [r for r in rows if r.get("kind") in OURS]

    print("== 계기 규약 ==")
    print("  우리 문구 = kind ∈ %s  (reminder-assistant=모델 재생성분·제외)" % (OURS,))
    print("  레버 정체 = 본문 앞 120자의 [TAG]; 없으면 channel 이름")
    print("  sim 식별 = 첫 user 발화 sha1[:12] (t2_fbsidecar._sim_key와 동일 규약)")
    print("  전체 %d행 중 우리 문구 %d행 / 시뮬 %d개\n" % (len(rows), len(ours), len(sims)))

    # ── A 레버별 노출 ────────────────────────────────────────────────────────
    print("== A. 레버별 발화 (통과/실패 분할은 **교란**됨: 실패 sim이 길어 기회가 많다) ==")
    per = collections.defaultdict(collections.Counter)
    for r in ours:
        per[lever_of(r)][key.get(r["sim"], "?")] += 1
    print("  %-24s %5s %7s %7s  태스크" % ("lever", "발화", "통과sim", "실패sim"))
    for k, c in sorted(per.items(), key=lambda kv: -sum(kv[1].values())):
        p = sum(v for t, v in c.items() if rew.get(t) == 1.0)
        f = sum(v for t, v in c.items() if rew.get(t) == 0.0)
        print("  %-24s %5d %7d %7d  %s" % (k, sum(c.values()), p, f,
                                           ",".join(sorted(t.replace("task_", "") for t in c))))

    # ── B 공발화 ─────────────────────────────────────────────────────────────
    print("\n== B. 같은 (sim, turn)에 함께 도착한 레버 쌍 ==")
    byturn = collections.defaultdict(set)
    for r in ours:
        byturn[(r["sim"], r["turn"])].add(lever_of(r))
    pair, where = collections.Counter(), collections.defaultdict(set)
    for (s, t), ls in byturn.items():
        ls = sorted(ls)
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                pair[(ls[i], ls[j])] += 1
                where[(ls[i], ls[j])].add(key.get(s, "?")[-3:])
    for (x, y), n in pair.most_common(24):
        print("  %3d회  %-22s + %-22s  %s" % (n, x, y, sorted(where[(x, y)])))

    # ── C 금지 조건(E3-②)의 표적 ────────────────────────────────────────────
    print("\n== C. 발화가 이름을 댄 도구를 **활성 절차가 금지**하는가 (speak-time 금지 조건) ==")
    print("  ⚠ speak-time 계약: prohibited()는 `names`가 절차 트리거이기도 하면 그 언급만으로")
    print("     절차를 활성 취급한다. 호출 시점(모델의 호출)과 발화 시점(우리의 권유)은 의미가")
    print("     다르므로, 여기서는 **이미 실행된 것으로 활성을 판정**하고 표적은 금지 대조에만 쓴다.")
    universe = tool_universe(DOMAIN)
    hit = collections.Counter()
    for r in ours:
        task = key.get(r["sim"], "?")
        msgs = sims.get(task) or []
        ex = executed_ok(msgs[:r["turn"]])
        lev = lever_of(r)
        for t in targets_of(r, universe):
            p, nm, spec = PR.prohibited(procs, {t}, ex)
            if p is not None:
                kind = "push(침묵 대상)" if lev in PUSH else (
                    "react(대상 아님)" if lev in REACT else "미분류")
                hit[(task, lev, kind, nm, p.get("id"))] += 1
    if hit:
        print("  %-10s %-20s %-16s %-26s %-22s %s" % ("태스크", "레버", "분류", "표적", "금지 절차", "발화"))
        for (task, lev, kind, nm, pid), n in sorted(hit.items()):
            print("  %-10s %-20s %-16s %-26s %-22s %d" % (task, lev, kind, nm, pid, n))
    else:
        print("  (해당 없음)")
    print("  ⇒ **push 행만** 금지 조건이 지운다. react 행을 지우면 그 금지를 집행한 문장 자신이 사라진다.")

    # ── D 단계 분포 ──────────────────────────────────────────────────────────
    print("\n== D. 발화 시점의 단계 (현행 배선 / 인자 전달 후 / discover 가설 추가) ==")
    dist = collections.defaultdict(collections.Counter)
    for r in ours:
        task = key.get(r["sim"], "?")
        msgs = (sims.get(task) or [])[:r["turn"]]
        ex = executed_ok(msgs)
        now = PH.phase_of(a2, msgs, eff_name, executed=ex)[0]                 # 현행(인자 미전달)
        with_st = PH.phase_of(a2, msgs, eff_name, executed=ex,
                              procedures_state=procedures_state(a2, ex))[0]   # E2-①
        hyp = with_st
        if hyp == "open" and discover_pending(a2, ex, msgs):
            hyp = "discover"                                                   # E2-② 가설
        dist[lever_of(r)][(now, with_st, hyp)] += 1
    print("  %-24s %-10s %-10s %-10s %s" % ("lever", "현행", "+state", "+discover", "발화"))
    for lev, c in sorted(dist.items(), key=lambda kv: -sum(kv[1].values())):
        for (a, b, cph), n in sorted(c.items(), key=lambda kv: -kv[1]):
            print("  %-24s %-10s %-10s %-10s %d" % (lev, a, b, cph, n))
    tot = collections.Counter()
    for c in dist.values():
        for (a, b, cph), n in c.items():
            tot[("현행", a)] += n
            tot[("+state", b)] += n
            tot[("+discover", cph)] += n
    print("\n  전체 분포:")
    for stage in ("현행", "+state", "+discover"):
        line = ", ".join("%s=%d" % (ph, n) for (s, ph), n in sorted(tot.items()) if s == stage)
        print("    %-10s %s" % (stage, line))
    print("\n  ⇒ 어떤 레버가 단계를 선언하기 전에 이 표를 읽는다: 선언한 단계가 여기서 0이면")
    print("     그 선언은 조정이 아니라 **그 레버를 끄는 것**이다.")


if __name__ == "__main__":
    main()
