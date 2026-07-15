# -*- coding: utf-8 -*-
"""bank_eplan_r3_recall.py — Ⓐ R3 게이트: CP5 obligation 추출 recall 오프라인 측정 (2026-07-16).

지배 레버(FIND/COVERAGE)의 실현자 = 32B replan 서브콜이 gold write-의무를 뽑나(gold-free).
floor 결과(32B agent 실 transcript + gold action_checks)에 structured_replan_prompt→8140→parse_obligations
돌려 **의무-recall(vs gold write) / precision** 측정. 유료 밤샘 前 무료 게이트(리뷰 Ⓐ·[[09]]).

- write 분류 = 도메인일반 read-prefix + procedural denylist (dag_plan 미러).
- recall = |추출 ∩ gold-write(tool-family)| / |gold-write| · precision = matched / 추출.
- 임계 미달 → 결정론 fallback(qty>executed) 없이 밤샘 금지.

사용(리모트): python bank_eplan_r3_recall.py --results <floor.gz> --base http://localhost:8140/v1 [--max N]
"""
import json, gzip, re, sys, io, os, argparse
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_eplan_patch as E

fam = lambda n: re.sub(r"_\d+$", "", str(n or ""))
_READ = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
_PROC = re.compile(r"(^log_|_verification$|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_|unlock_)", re.I)
def is_write(nm):
    nm = fam(nm)
    return bool(nm) and not _READ.match(nm) and not _PROC.search(nm)

def nd(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return x if isinstance(x, dict) else {}

def gold_writes(sim):
    """gold action_checks의 미충족 write 스텝 (도메인일반 분류·tool-family + entity)."""
    ri = sim.get("reward_info") or {}
    out = []
    for ac in (ri.get("action_checks") or []):
        a = ac.get("action") or {}
        outer = nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "") or a.get("name", "")
        if not atn or not is_write(atn):
            continue
        inner = nd(outer.get("arguments"))
        # entity = primary id-like arg (not user_id)
        ent = ""
        for k in ("transaction_id", "card_id", "account_id", "credit_card_account_id"):
            if inner.get(k):
                ent = str(inner[k]); break
        out.append((fam(atn), ent))
    return out

def load_sims(path):
    op = gzip.open if path.endswith(".gz") else open
    d = json.load(op(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8"))
    return d.get("simulations", [])

# 최소 eplan spec(ledger 요약용·entity_key=transaction_id 대표)
SPEC = {"entity_key": "transaction_id", "list_from_reads": True,
        "dispatch_tool": "call_discoverable_agent_tool"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--failed_only", type=int, default=1)
    a = ap.parse_args()
    from openai import OpenAI
    cl = OpenAI(base_url=a.base, api_key="x")

    sims = load_sims(a.results)
    rows = []
    tot_gold = tot_match = tot_extract = 0
    n = 0
    by_sim = []
    for s in sims:
        ri = s.get("reward_info") or {}
        if a.failed_only and ri.get("reward") in (None, 1.0):
            continue
        gw = gold_writes(s)
        if not gw:
            continue
        n += 1
        if a.max and n > a.max:
            n -= 1; break
        msgs = s.get("messages") or []
        led = E.build_ledger_from_messages(msgs, SPEC, set())
        transcript = E.transcript_text(msgs)
        prompt = E.structured_replan_prompt(led, transcript, "transaction_id")
        try:
            r = cl.chat.completions.create(model=a.model, messages=[{"role": "user", "content": prompt}],
                                           temperature=0, max_tokens=800)
            obls = E.parse_obligations(r.choices[0].message.content or "", "transaction_id") or []
        except Exception as e:
            print("ERR", type(e).__name__, e); break
        ext_fams = Counter(fam(o.get("intent_class")) for o in obls if is_write(o.get("intent_class")))
        gold_fams = Counter(f for f, _ in gw)
        # recall: gold write-family가 추출집합에 있나 (family-level·entity 무시=상한 recall)
        matched = sum(min(gold_fams[f], ext_fams.get(f, 0)) for f in gold_fams)
        gcount = sum(gold_fams.values()); ecount = sum(ext_fams.values())
        tot_gold += gcount; tot_match += matched; tot_extract += ecount
        by_sim.append((s.get("task_id"), gcount, matched, ecount))
        if n <= 8 or n % 20 == 0:
            print("[%d] %s gold=%s extracted=%s matched=%d" %
                  (n, s.get("task_id"), dict(gold_fams), dict(ext_fams), matched), flush=True)
    print("\n=== R3 obligation-recall (32B replan·gold-free·n=%d sims) ===" % n)
    print("  gold write-steps: %d · extracted: %d · matched(family): %d" % (tot_gold, tot_extract, tot_match))
    print("  ★RECALL = %.1f%% (matched/gold)   PRECISION = %.1f%% (matched/extracted)" %
          (100*tot_match/max(tot_gold,1), 100*tot_match/max(tot_extract,1)))
    full = sum(1 for _, g, m, _ in by_sim if m >= g and g > 0)
    print("  전-의무 recall된 sim: %d/%d (%.0f%%)" % (full, len(by_sim), 100*full/max(len(by_sim),1)))
    print("  판정: recall 높음(≥~70%%)=CP5 의무추출 실현→FIND/COVERAGE 라이브 가능. 낮음=결정론 fallback(qty>executed) 필수·[[09]] 밤샘 前.")

if __name__ == "__main__":
    main()
