# -*- coding: utf-8 -*-
"""bank_goldfree_coverage.py — 리뷰 ❹: 라이브 coverage-set을 gold 없이 도출 가능한가 (go/no-go).

오프라인 상한(9.9~29%)은 gold(action_checks)로 "무엇이 빠졌나"를 안다. 라이브 컨트롤러엔 gold 없음
→ H_min 강제열거가 "어느 write를 해야 하나"를 gold-free 신호(user 요청 ∪ discovery ∪ ABox 술어)로 골라야.
이게 되면 Track A 실현·안 되면 상한은 gold-mirage.

측정: 각 gold WRITE 액션의 타깃 엔티티가 gold-free로 복원되나 —
  (A) user 발화에 엔티티 명시(고객이 직접 지목)
  (B) discovery surface + 유일후보(reference_filter류·ABox 술어로 선택)
  (C) 복원 불가(gold 없이 못 고름=mirage)
로컬 무료·per-write-family."""
import json, glob, re, sys, io
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
fam = lambda n: re.sub(r"_\d+$", "", str(n))
READ = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
PROC = re.compile(r"(^log_|_verification$|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_)", re.I)
IDRX = re.compile(r"\b([a-z]{2,6}_[0-9a-f]{6,}|txn_[0-9a-f]+|cc_[0-9a-z_]+|dbc_[0-9a-z_]+|chk_[0-9a-z_]+)\b", re.I)

def entity_of(gold_args):
    """write 액션의 타깃 엔티티 id (가장 특정한 것)."""
    for k in ("transaction_id", "card_id", "account_id", "order_id", "report_id"):
        if gold_args.get(k):
            return str(gold_args[k])
    return None

cls = Counter(); by_fam = defaultdict(Counter); n = 0
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        if ri.get("reward") in (None, 1.0): continue
        if tuple(ri.get("reward_basis") or []) != ("DB",): continue
        if str(s.get("termination_reason")) == "too_many_errors": continue
        utext = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") == "user").lower()
        # discovery = tool result 전체 텍스트(substring 검사·[[08]] 정규식 under-capture 회피·sav_/기타 접두사 포괄)
        disc_text = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") == "tool").lower()
        for ac in (ri.get("action_checks") or []):
            a = ac.get("action") or {}
            outer = Nd(a.get("arguments")); atn = outer.get("agent_tool_name", "")
            if not atn or "arguments" not in outer: continue
            tf = fam(atn)
            if READ.match(tf) or PROC.search(tf): continue   # write만
            ga = Nd(outer.get("arguments")); ent = entity_of(ga)
            if not ent: continue
            n += 1; el = ent.lower()
            if el in utext:
                cls["A. user 발화 명시(gold-free ✓)"] += 1; by_fam[tf]["A"] += 1
            elif el in disc_text:
                cls["B. discovery surface(선택 술어 필요)"] += 1; by_fam[tf]["B"] += 1
            else:
                cls["C. gold-free 복원불가(mirage 위험)"] += 1; by_fam[tf]["C"] += 1

print("=== 리뷰 ❹: gold WRITE 타깃 엔티티의 gold-free 복원성 (DB-basis 실패·%d write) ===" % n)
for k, v in cls.most_common():
    print("  %-38s %6d (%.1f%%)" % (k, v, 100 * v / max(n, 1)))
A = cls["A. user 발화 명시(gold-free ✓)"]; B = cls["B. discovery surface(선택 술어 필요)"]
print("\n  ★gold-free 직접복원(A) = %.1f%% · +discovery후 술어선택(A+B) = %.1f%% · mirage(C) = %.1f%%"
      % (100 * A / max(n, 1), 100 * (A + B) / max(n, 1), 100 * cls["C. gold-free 복원불가(mirage 위험)"] / max(n, 1)))
print("\n  write-family별 A/B/C:")
for tf, c in sorted(by_fam.items(), key=lambda x: -sum(x[1].values()))[:12]:
    tot = sum(c.values())
    print("    %-40s A=%d B=%d C=%d (n=%d)" % (tf, c["A"], c["B"], c["C"], tot))
print("\n  판정: A 높으면 coverage-set gold-free 도출 가능(Track A 실현성 ↑). C 높으면 상한=mirage(라이브 착수 금지).")
print("  B는 ABox 선택술어(reference_filter·⋈-decidable) 필요 — 그 술어 정확도가 별도 관문.")
