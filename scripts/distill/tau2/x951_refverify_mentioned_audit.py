# -*- coding: utf-8 -*-
"""x951 — `_ref_verify_deny._mentioned`(t2_gate_patch.py:2075-2081) 오프라인 전수 감사.

질문 3개(반증자용):
  Q1 이 술어는 실제로 **무엇에 닿나** — 자유 텍스트인가, 선언이 좁힌 닫힌 집합인가.
  Q2 **false-block**(gold 레코드인데 '미언급'으로 판정) 이 회수분에 실재하나.
  Q3 완화(토큰≥5)가 없으면 무엇을 잃나 — 축자만으로 gold 가 몇 건 막히나.

엔진 술어를 **그대로** 복제(축자 이식)해서 회수분 sim 에 돌린다. LLM 0.
Run: py -3 x951_refverify_mentioned_audit.py
"""
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIMR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

# ── 엔진 축자 이식 (t2_gate_patch.py:2075-2081) ──────────────────────────
def _mentioned(val, utext, min_tok=5):
    if not val:
        return False
    if val.lower() in utext:
        return True
    for tok in re.findall(r"[A-Za-z0-9]+", val):
        if len(tok) >= min_tok and tok.lower() in utext:
            return True
    return False


def _exact(val, utext):
    return bool(val) and val.lower() in utext


ROW_RE = re.compile(r"merchant_name:\s*([^\n|]+)")
TXN_RE = re.compile(r"txn_[0-9a-f]{4,}")


def listing_map(msgs):
    """도구 출력에서 txn_id -> merchant_name (프로브 전용 파서. 라이브는 sub_records)."""
    m = {}
    allv = set()
    for msg in msgs:
        if msg.get("role") != "tool":
            continue
        c = str(msg.get("content") or "")
        if "merchant_name" not in c:
            continue
        for blk in re.split(r"\n\s*\n|(?=transaction_id)", c):
            tid = TXN_RE.search(blk)
            mer = ROW_RE.search(blk)
            if mer:
                v = mer.group(1).strip()
                allv.add(v)
                if tid:
                    m.setdefault(tid.group(), v)
    return m, allv


def user_text(msgs):
    raw = "\n".join(str(msg.get("content") or "") for msg in msgs
                    if msg.get("role") == "user")
    return TXN_RE.sub("", raw).lower()          # id 오염 제거(엔진은 merchant 값만 본다)


def filed_ids(msgs):
    out = []
    for msg in msgs:
        for tc in (msg.get("tool_calls") or []):
            if tc.get("name") != "call_discoverable_agent_tool":
                continue
            aa = tc.get("arguments")
            aa = json.dumps(aa) if not isinstance(aa, str) else aa
            if "file_credit_card_transaction_dispute" not in aa:
                continue
            mm = TXN_RE.search(aa)
            if mm:
                out.append(mm.group())
    return out


def gold_ids(sim):
    out = []
    for ac in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ac.get("action") or {}
        s = json.dumps(a, ensure_ascii=False)
        if "file_credit_card_transaction_dispute" in s:
            out.extend(TXN_RE.findall(s))
    return out


def main():
    pats = sys.argv[1:] or ["bank_t739*.results.json.gz", "bank_x599*.results.json.gz",
                            "bank_x644*.results.json.gz", "bank_rall*.results.json.gz",
                            "bank_t738*.results.json.gz"]
    files = []
    for p in pats:
        files.extend(sorted(glob.glob(os.path.join(SIMR, p))))
    files = [f for f in files if "smoke" not in os.path.basename(f)]

    n_sim = n_dispute_sim = 0
    fired = []          # (tag, task, tid, merchant, is_gold, reward)
    passed = []
    relax_only = []     # 완화 덕에 통과한 것
    allval_rows = []
    for f in files:
        tag = os.path.basename(f).split(".")[0]
        try:
            d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        except Exception as e:
            print("  !! skip %s (%s)" % (tag, e))
            continue
        for sim in d.get("simulations", []):
            n_sim += 1
            msgs = sim.get("messages") or []
            fid = filed_ids(msgs)
            if not fid:
                continue
            n_dispute_sim += 1
            ut = user_text(msgs)
            tmap, allv = listing_map(msgs)
            gid = set(gold_ids(sim))
            rw = (sim.get("reward_info") or {}).get("reward")
            if rw is None:
                rw = sim.get("reward")
            task = sim.get("task_id")
            for v in sorted(allv):
                allval_rows.append((tag, task, v, _exact(v, ut), _mentioned(v, ut)))
            for tid in dict.fromkeys(fid):
                mer = tmap.get(tid)
                if not mer:
                    continue                     # rec_val 없음 → 라이브도 skip
                rec = (tag, task, tid, mer, tid in gid, rw)
                if _mentioned(mer, ut):
                    passed.append(rec)
                    if not _exact(mer, ut):
                        relax_only.append(rec)
                else:
                    fired.append(rec)

    print("파일 %d · sim %d · dispute-filing sim %d" % (len(files), n_sim, n_dispute_sim))
    print("\n=== [Q2] 실제 file 호출에 대한 술어 판정 (rec_val 복원된 것만) ===")
    print("  통과(=언급 인정) %d · deny 발화 %d" % (len(passed), len(fired)))
    fb = [r for r in fired if r[4]]
    print("  ⊖ FALSE-BLOCK (gold 인데 deny) = %d" % len(fb))
    for r in fb:
        print("     %s %s %s merchant=%r reward=%s" % (r[0], r[1], r[2][-6:], r[3], r[5]))
    tp = [r for r in fired if not r[4]]
    print("  ⊕ CAUGHT (gold 아닌데 deny) = %d" % len(tp))
    for r in tp[:20]:
        print("     %s %s %s merchant=%r reward=%s" % (r[0], r[1], r[2][-6:], r[3], r[5]))
    miss = [r for r in passed if not r[4]]
    print("  ⊘ MISS (gold 아닌데 통과) = %d" % len(miss))
    for r in miss[:20]:
        print("     %s %s %s merchant=%r reward=%s" % (r[0], r[1], r[2][-6:], r[3], r[5]))
    gpass = [r for r in passed if r[4]]
    print("  ✓ gold 통과 = %d" % len(gpass))

    print("\n=== [Q3] 완화(토큰≥5)가 없었다면: 축자만으로 판정 ===")
    ro_gold = [r for r in relax_only if r[4]]
    print("  완화 덕에 통과한 실제 filing = %d (그중 gold = %d)" % (len(relax_only), len(ro_gold)))
    for r in relax_only[:20]:
        print("     %s %s %s merchant=%r gold=%s reward=%s" % (r[0], r[1], r[2][-6:], r[3], r[4], r[5]))

    print("\n=== [Q1] 후보 집합의 크기 — 술어가 닿는 값 = 도구출력 merchant_name 값 ===")
    ex = sum(1 for r in allval_rows if r[3])
    tk = sum(1 for r in allval_rows if r[4])
    print("  merchant 값 표본 %d · 축자 %d · 완화포함 %d (완화가 늘린 것 %d)"
          % (len(allval_rows), ex, tk, tk - ex))
    seen = set()
    for r in allval_rows:
        if r[4] and not r[3] and r[2] not in seen:
            seen.add(r[2])
            print("     [완화로 '언급'] %s %s merchant=%r" % (r[0], r[1], r[2]))


if __name__ == "__main__":
    main()
