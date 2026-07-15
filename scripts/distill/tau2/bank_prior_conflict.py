# -*- coding: utf-8 -*-
"""bank_prior_conflict.py — 게이트2(Track B Phase-0) 무료 진단: F3 enum 오답이 prior-override인가 (2026-07-16).

리뷰 ❻ 핵심: 스킬이 "prior=스키마일 때만" 작동하면 전이 무효. → banking F3 실패가 실제로 prior-conflict
(agent가 NL-직관 enum 선택·gold는 정책-반직관)인지 기존 데이터로 측정.
방법: dispute enum 필드(dispute_reason/category/card_action)의 (agent_val, gold_val, 고객NL) 삼항 →
  agent_val이 gold_val보다 고객 NL에 더 align하나(=agent가 표면-직관값 선택=prior-override).
prior-override 지배 = Track B가 prior-conflict 필수(❻ 제약 정당) + F3=능력/스킬(프롬프트 불가·[[42]]).
로컬 무료·per-case 근거 포함."""
import json, glob, re, sys, io
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
fam = lambda n: re.sub(r"_\d+$", "", str(n))
ENUM_FIELDS = ["dispute_reason", "dispute_category", "card_action", "transaction_type"]

def toks(v):
    return set(t for t in re.split(r"[_\s]+", str(v).lower()) if len(t) >= 4)

def nl_align(val, nl):
    """enum 값의 토큰이 고객 NL에 얼마나 등장 (표면 정렬 점수)."""
    tk = toks(val)
    if not tk:
        return 0.0
    return sum(1 for t in tk if t in nl) / len(tk)

from collections import defaultdict
cls = Counter(); examples = []
by_field = defaultdict(Counter)   # 필드별 (의미 enum vs 기술 enum 분리)
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        if ri.get("reward") in (None, 1.0): continue
        if tuple(ri.get("reward_basis") or []) != ("DB",): continue
        if str(s.get("termination_reason")) == "too_many_errors": continue
        nl = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") == "user").lower()
        # agent dispute 제출 (transaction_id 키)
        asub = {}
        for m in (s.get("messages") or []):
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") == "call_discoverable_agent_tool" and "dispute" in fam(Nd(tc.get("arguments")).get("agent_tool_name", "")):
                    aa = Nd(Nd(tc.get("arguments")).get("arguments")); t = str(aa.get("transaction_id") or "")
                    if t: asub.setdefault(t, aa)
        for ac in (ri.get("action_checks") or []):
            a = ac.get("action") or {}; outer = Nd(a.get("arguments"))
            if "dispute" not in fam(outer.get("agent_tool_name", "")) or "arguments" not in outer: continue
            ga = Nd(outer.get("arguments")); t = str(ga.get("transaction_id") or "")
            if t not in asub: continue
            aa = asub[t]
            for fld in ENUM_FIELDS:
                gv, av = ga.get(fld), aa.get(fld)
                if gv is None or av is None: continue
                if str(gv).strip().lower() == str(av).strip().lower(): continue   # 정답은 skip
                # 오답 enum: agent값 vs gold값 NL-정렬 비교
                ag, gg = nl_align(av, nl), nl_align(gv, nl)
                if ag > gg:
                    cls["prior-override(agent값이 NL-직관·gold 반직관)"] += 1; by_field[fld]["prior-override"] += 1
                    if len(examples) < 8:
                        examples.append((fld, "agent=%s(%.2f)" % (av, ag), "gold=%s(%.2f)" % (gv, gg)))
                elif ag < gg:
                    cls["gold이 더 NL-정렬(agent 딴값·비-prior)"] += 1; by_field[fld]["gold-aligned"] += 1
                else:
                    cls["동률(판정불가)"] += 1; by_field[fld]["tie"] += 1

tot = sum(cls.values())
print("=== 게이트2(Track B Phase-0): F3 enum 오답의 prior-override 여부 (dispute·%d 오답 enum) ===" % tot)
for k, v in cls.most_common():
    print("  %-44s %5d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
po = cls["prior-override(agent값이 NL-직관·gold 반직관)"]
print("\n  ★prior-override 지배 = %.1f%%" % (100 * po / max(tot, 1)))
print("  예시 (필드·agent값·gold값·NL정렬점수):")
for fld, av, gv in examples:
    print("    %-18s %-28s %s" % (fld, av, gv))
print("\n  필드별 (의미 enum=dispute_reason/category vs 기술 enum=transaction_type):")
for fld, c in by_field.items():
    t = sum(c.values())
    print("    %-20s prior-override=%d gold-aligned=%d tie=%d (n=%d)" % (fld, c["prior-override"], c["gold-aligned"], c["tie"], t))
print("\n  판정: prior-override=surface-plausible 오답(prior 억제 필요·가장 어려움)·gold-aligned=NL에 답 있는데 미-attend")
print("        (attend-schema 스킬로 닫힘)·tie=NL 미-disambig(정책추론/ASK). 셋 다 Track B 표적·비중이 학습 설계 좌우.")
print("  ★[[08]] 약신호(토큰중첩·transaction_type 노이즈). 정본 측정=base 모델 NL→enum eval(서버 필요·게이트2 다음).")
