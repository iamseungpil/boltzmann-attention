# -*- coding: utf-8 -*-
"""bank_frontier_mechanism.py — frontier 모델별 실패궤적 전수 + 우리 매커니즘 극복 판정 (2026-07-16·사용자 지시).

17 frontier 모델(리더보드 제출 banking 궤적·C:/tmp/traj) 실패를 per-model per-step 분해하고,
각 실패 sim을 우리 매커니즘(per-step 균일 연산-loop {GET/FIND/COMPUTE/ASK}+H_min+coverage+suppress)이
극복하는지 tier로 판정.

  Tier-D (결정론 closable·무료): 모든 층 ∈ {COVERAGE, FIND-discovery, COMPUTE, OVER-ACTION}
       = H_min+coverage-track(under-action)·강제열거(reach)·ABox COMPUTE·suppress. 능력비용 0.
  Tier-X (decidable ⋈): 위 + GET-⋈(엔티티 유일결정=결정론 GET/filter·C78 decidable부 100%).
  Tier-A (ASK/의미): GATHER-ASK 층 有 = enum/reason/의미 갭 → 일부 결정론 그라운딩·일부 ASK·일부 경계.
  Tier-Blind: action-check 없음(pure-DB) = 오프라인 관측 불가(C93 blind).

DB-basis 실패·infra(too_many_errors) 제외. bank_perstep_decomp 로직 재사용.
사용: py bank_frontier_mechanism.py
"""
import json, glob, re, sys, io, os
from collections import Counter, defaultdict

import bank_perstep_decomp as P   # 동일 디렉터리·decompose_sim/load_compute_fields 재사용
# stdout utf-8 wrap은 P import가 이미 설정(이중 wrap 금지·buffer 닫힘 방지).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ABOX = os.path.join(_HERE, "a2", "banking_knowledge.gate.json")


DET = {"COVERAGE", "FIND-discovery", "COMPUTE", "OVER-ACTION"}
# 결정론 COMPUTE/GET-lookup 필드 (rule_fit.py gold-fit 확증분·2026-07-16):
#  liability 94.7%·amount_difference=산술 확증·expected_apy/rewards=GET-lookup.
_COMPUTE_LIKE = re.compile(
    r"(amount_difference|expected_apy|actual_apy|_apy$|new_rewards|interest)", re.I)
# ★자격 judgment (rule_fit: 날짜식 69%≈base 65%=NOT deterministic) → F3/의미경계로 강등([[08]]).
_JUDGMENT = re.compile(r"(eligible|provisional_credit|refund_amount)", re.I)


def sim_tier(layers):
    if not layers:
        return "Tier-Blind(pure-DB)"
    st = set(layers)
    if st <= DET:
        return "Tier-D(결정론 closable)"
    if st <= (DET | {"GET-xmatch"}):
        return "Tier-X(+decidable ⋈)"
    if "GATHER-ASK" in st:
        return "Tier-A(ASK/의미 잔여)"
    return "Tier-?(기타)"


# enum-형 필드(NL→정규화=F3 의미참조·literal presence=schema 오염이라 그라운딩 무효)
_ENUM_FIELD = re.compile(
    r"(reason|category|type|action|status|option|design|method|resolution|class|compromised|possession|filed|provided)", re.I)


def sim_tier_grounded(layers, ga_fields, ctx, cmap):
    """[[08]] 교정판(enum-schema 오탐 제거·per-case 감사 반영·2026-07-16).
    GATHER-ASK 필드 유형별: compute-like→결정론 · enum→F3의미경계(그라운딩 무효) ·
    data(id/amount/date)&present→GET그라운딩 · data&absent→ASK.
    sim tier = 최악 필드 (all-layer AND). data present는 tool-record 그라운딩(GET) 정당."""
    st = set(layers)
    if not layers:
        return "Blind(pure-DB)"
    non_ga = st - {"GATHER-ASK"}
    if non_ga - (DET | {"GET-xmatch"}):
        return "기타"
    order = {"det": 0, "grounding": 1, "ASK": 2, "F3": 3}
    worst = "det"
    for (tf, field, val) in ga_fields:
        if _JUDGMENT.search(field):
            cat = "F3"                                   # 자격 judgment(gold-fit 실패=NOT det·[[08]])
        elif _COMPUTE_LIKE.search(field):
            cat = "det"                                  # 결정론 compute/GET-lookup(gold-fit 확증)
        elif _ENUM_FIELD.search(field):
            cat = "F3"                                   # NL→정규화 의미경계(오염이라 그라운딩 무효)
        else:
            pres = P_val_present(val, ctx)               # data 필드=id/amount/date
            cat = "grounding" if pres else ("ASK" if pres is False else "det")
        if order[cat] > order[worst]:
            worst = cat
    return {"det": "D+X+compute(결정론)", "grounding": "그라운딩-closable(data present)",
            "ASK": "ASK(data 부재)", "F3": "F3-의미경계(enum NL→정규화)"}[worst]


def P_val_present(val, ctx):
    v = str(val).strip().lower()
    if not v or v in ("none", "null", "true", "false"):
        return None
    if v in ctx:
        return True
    toks = [t for t in re.split(r"[_\s]+", v) if len(t) >= 4]
    if toks and sum(1 for t in toks if t in ctx) >= max(1, len(toks) - 1):
        return True
    return False


def ga_wrong_fields(s, abox, cmap):
    """이 sim의 GATHER-ASK 틀린 (tool,field,value) 목록."""
    out = []
    calls = P.agent_calls_by_family(s)
    for ac in ((s.get("reward_info") or {}).get("action_checks") or []):
        a = ac.get("action") or {}
        outer = P._nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "")
        if not atn or "arguments" not in outer:
            continue
        tf = P._fam(atn)
        if P.is_read(tf):
            continue
        met = ac.get("action_reward")
        if met is None:
            met = 1.0 if ac.get("action_match") else 0.0
        if float(met) >= 1.0:
            continue
        gold_args = P._nd(outer.get("arguments"))
        mm = P.best_match(gold_args, calls.get(tf, []))
        if mm is None or not mm[1]:
            continue
        for field in mm[1]:
            if P.classify_field(field, tf, cmap) == "GATHER-ASK":
                out.append((tf, field, gold_args.get(field)))
    return out


def main():
    abox = json.load(open(_ABOX, encoding="utf-8"))
    cmap = P.load_compute_fields(abox)
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))

    per_model = {}          # model → Counter(tier)
    per_model_meta = {}     # model → (nsims, npass, ndbfail_decomposed, ninfra)
    agg_tier = Counter()
    agg_layer = Counter()
    for f in files:
        model = f.replace("\\", "/").split("/")[-1].replace("_banking.json", "")
        d = json.load(open(f, encoding="utf-8"))
        sims = d.get("simulations", [])
        npass = sum(1 for s in sims if (s.get("reward_info") or {}).get("reward") == 1.0)
        tc = Counter()
        ninfra = 0
        blind = 0
        for s in sims:
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            if tuple(ri.get("reward_basis") or []) != ("DB",):
                continue
            if str(s.get("termination_reason")) == "too_many_errors":
                ninfra += 1
                continue
            r = P.decompose_sim(s, abox, cmap)
            t = sim_tier(r["layers"])
            tc[t] += 1
            agg_tier[t] += 1
            for op in r["layers"]:
                agg_layer[op] += 1
        per_model[model] = tc
        per_model_meta[model] = (len(sims), npass, sum(tc.values()), ninfra)

    # 출력: 모델별 tier %
    print("=== frontier 모델별 실패궤적 × 우리 매커니즘 극복 tier (DB-basis 실패·infra제외) ===")
    hdr = "%-14s %5s %6s | %7s %7s %7s %7s" % (
        "model", "pass%", "DBfail", "Tier-D", "Tier-X", "Tier-A", "Blind")
    print(hdr); print("-" * len(hdr))
    order = sorted(per_model, key=lambda m: -per_model_meta[m][1] / max(per_model_meta[m][0], 1))
    for m in order:
        nsims, npass, ndf, ninf = per_model_meta[m]
        tc = per_model[m]
        D = tc["Tier-D(결정론 closable)"]; X = tc["Tier-X(+decidable ⋈)"]
        A = tc["Tier-A(ASK/의미 잔여)"]; B = tc["Tier-Blind(pure-DB)"]
        tot = max(ndf, 1)
        print("%-14s %5.1f %6d | %6.1f%% %6.1f%% %6.1f%% %6.1f%%" % (
            m, 100 * npass / max(nsims, 1), ndf,
            100 * D / tot, 100 * X / tot, 100 * A / tot, 100 * B / tot))

    # 집계
    tot = sum(agg_tier.values())
    print("\n=== 전 모델 집계 (DB-basis 실패 %d) ===" % tot)
    for t in ("Tier-D(결정론 closable)", "Tier-X(+decidable ⋈)", "Tier-A(ASK/의미 잔여)", "Tier-Blind(pure-DB)", "Tier-?(기타)"):
        if agg_tier.get(t):
            print("  %-24s %6d (%.1f%%)" % (t, agg_tier[t], 100 * agg_tier[t] / max(tot, 1)))
    cumDX = agg_tier["Tier-D(결정론 closable)"] + agg_tier["Tier-X(+decidable ⋈)"]
    print("\n  ★결정론 극복(Tier-D+X) = %.1f%% (능력비용 0·coverage/FIND/COMPUTE/⋈-decidable)" % (100 * cumDX / max(tot, 1)))
    print("  ★+ASK 층(Tier-A) = %.1f%% (의미 갭·일부 그라운딩/ASK 닫힘·일부 경계·inner router 세분 필요)" % (100 * agg_tier["Tier-A(ASK/의미 잔여)"] / max(tot, 1)))
    print("  ★관측불가(Blind) = %.1f%% (pure-DB·오프라인 상한 밖·DB-replay/live만)" % (100 * agg_tier["Tier-Blind(pure-DB)"] / max(tot, 1)))
    print("\n  전 층 연산 빈도:", dict(agg_layer.most_common()))

    # ── sim-레벨 그라운딩+compute 통합 재판정 (정밀 '극복 가능') ──
    gt = Counter()
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            if tuple(ri.get("reward_basis") or []) != ("DB",):
                continue
            if str(s.get("termination_reason")) == "too_many_errors":
                continue
            r = P.decompose_sim(s, abox, cmap)
            gaf = ga_wrong_fields(s, abox, cmap)
            ctx = " ".join(str(m.get("content")) for m in (s.get("messages") or [])
                           if m.get("role") in ("user", "tool")).lower()
            gt[sim_tier_grounded(r["layers"], gaf, ctx, cmap)] += 1
    gtot = sum(gt.values())
    print("\n=== ★sim-레벨 '우리 매커니즘 극복' 판정 ([[08]] enum-오탐 교정·%d) ===" % gtot)
    order2 = ["D+X+compute(결정론)", "그라운딩-closable(data present)", "ASK(data 부재)", "F3-의미경계(enum NL→정규화)", "Blind(pure-DB)", "기타"]
    det = 0
    for k in order2:
        if gt.get(k):
            print("  %-32s %6d (%.1f%%)" % (k, gt[k], 100 * gt[k] / max(gtot, 1)))
        if k in ("D+X+compute(결정론)", "그라운딩-closable(data present)"):
            det += gt.get(k, 0)
    obs = gtot - gt.get("Blind(pure-DB)", 0)
    print("  " + "─" * 40)
    print("  ★결정론+GET그라운딩 극복 = %.1f%% (관측가능 중 %.1f%%)" % (100 * det / max(gtot, 1), 100 * det / max(obs, 1)))
    print("  ★F3-의미경계(enum NL→정규화) = %.1f%% = LOCKED-frame 잔여(우리도·frontier도 공유·일부 ASK로만)" % (100 * gt.get("F3-의미경계(enum NL→정규화)", 0) / max(gtot, 1)))
    print("  ★ASK(data 부재)/Blind = user-원천/오프라인밖.")
    print("  [[08]] enum literal=schema 오염→그라운딩서 제외(F3로)·data present=tool-record GET 정당·compute=ABox규칙 전제·sim=all-layer AND.")


if __name__ == "__main__":
    main()
