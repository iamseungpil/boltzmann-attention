# -*- coding: utf-8 -*-
"""bank_cause_census.py — 전수 미충족 gold-write 원인 census (2026-07-17·사용자 "전수 궤적 추적").

각 실패 sim의 미충족 gold-write 스텝을 정밀 원인으로 분해:
  A 무시도(도구 안 부름):  A1 not-surfaced(타깃 record 미조회=REACH/discovery)
                            A2 surfaced-not-written(조회했으나 미제출=COVERAGE/horizon)
  B 시도-오답(도구 부름):   B1 compute / B2 reference(id) / B3 F3-enum / B4 judgment / B5 gather
     (field_ops=ABox·[[05]]·per-field 최악 원인)
원인별 레버·신뢰도 매핑. 도메인일반 write 분류(read-prefix+procedural·dag_plan 미러).
로컬 무료·C:/tmp/traj 17모델.
"""
import json, glob, re, sys, io, os, gzip, argparse
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

fam = lambda n: re.sub(r"_\d+$", "", str(n or ""))
_READ = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
_PROC = re.compile(r"(^log_|_verification$|^kb_|^shell$|discoverable|transfer_to_human|give_|unlock_)", re.I)
isw = lambda n: bool(fam(n)) and not _READ.match(fam(n)) and not _PROC.search(fam(n))
_TXN = re.compile(r"\b((?:txn|btxn|chk|dbc|ccord|dcord|clsr|cli|card|acct|ca)_[0-9a-fA-F]{6,})\b")

def nd(x):
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception: return {}
    return x if isinstance(x, dict) else {}

def load_abox():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "banking_knowledge.gate.json")
    return json.load(open(p, encoding="utf-8"))

def field_op(field, fo):
    if field in set(fo.get("judgment", [])): return "B4_judgment"
    if field in set(fo.get("compute", [])): return "B1_compute"
    if field in set(fo.get("id_ref", [])) or field.endswith("_id"): return "B2_reference"
    if field in set(fo.get("enum", [])): return "B3_F3enum"
    return "B5_gather"

def entity_of(args):
    for k in ("transaction_id", "card_id", "account_id", "credit_card_account_id", "checking_account_id"):
        if args.get(k): return str(args[k])
    return ""

def iter_sims(results):
    """results gz(단일 실행) 또는 C:/tmp/traj glob(frontier 17모델)."""
    if results:
        op = gzip.open if results.endswith(".gz") else open
        d = json.load(op(results, "rt", encoding="utf-8"))
        for s in d.get("simulations", []):
            yield "floor", s
    else:
        for f in sorted(glob.glob("C:/tmp/traj/*_banking.json")):
            model = os.path.basename(f).replace("_banking.json", "")
            d = json.load(open(f, encoding="utf-8"))
            for s in d.get("simulations", []):
                yield model, s


def census(results=None):
    abox = load_abox(); fo = abox.get("field_ops") or {}
    src = results or "C:/tmp/traj (frontier 17모델)"
    cause = Counter(); n_sim = 0; n_write = 0
    by_model = defaultdict(Counter)
    a_vs_b = Counter()
    per_sim_dom = Counter()   # sim의 지배원인(최다 미충족 원인 클래스 A/B)
    if True:
        for model, s in iter_sims(results):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0): continue
            if tuple(ri.get("reward_basis") or []) != ("DB",): continue
            if str(s.get("termination_reason")) == "too_many_errors": continue
            n_sim += 1
            # agent 호출 write 패밀리 + 제출 args(전 dispatcher) + surfaced id 집합
            called = set(); subs = defaultdict(dict); surfaced = set()
            for m in (s.get("messages") or []):
                if m.get("role") in ("tool", "user"):
                    surfaced |= set(_TXN.findall(str(m.get("content"))))
                for tc in (m.get("tool_calls") or []):
                    nm = tc.get("name")
                    if nm == "call_discoverable_agent_tool":
                        outer = nd(tc.get("arguments")); tfam = fam(outer.get("agent_tool_name", ""))
                        called.add(tfam)
                        ia = nd(outer.get("arguments")); e = entity_of(ia)
                        if e: subs[(tfam, e)] = ia
                    elif nm:
                        called.add(fam(nm))
            sim_causes = []
            for ac in (ri.get("action_checks") or []):
                a = ac.get("action") or {}; outer = nd(a.get("arguments"))
                atn = outer.get("agent_tool_name", "") or a.get("name", "")
                if not isw(atn): continue
                met = ac.get("action_reward"); met = met if met is not None else (1.0 if ac.get("action_match") else 0.0)
                if float(met) >= 1.0: continue
                tf = fam(atn); ga = nd(outer.get("arguments")); ent = entity_of(ga)
                n_write += 1
                if tf not in called:
                    # A: never-attempted
                    if ent and ent in surfaced:
                        c = "A2_surfaced_not_written(COVERAGE/horizon)"
                    else:
                        c = "A1_not_surfaced(REACH/discovery)"
                    a_vs_b["A_never_attempted"] += 1
                else:
                    # B: attempted — 어느 필드가 최악 원인
                    agent_args = subs.get((tf, ent), {})
                    wrong = [k for k, gv in ga.items()
                             if k != "transaction_id" and str(agent_args.get(k)) != str(gv)]
                    ops = [field_op(k, fo) for k in wrong] or ["B5_gather"]
                    rank = {"B1_compute": 0, "B2_reference": 1, "B3_F3enum": 2, "B4_judgment": 3, "B5_gather": 4}
                    c = min(ops, key=lambda o: rank[o])  # 최결정론(닫기 쉬운) 원인 우선 귀속
                    a_vs_b["B_attempted_wrong"] += 1
                cause[c] += 1; by_model[model][c[:2]] += 1; sim_causes.append(c[0])
            if sim_causes:
                per_sim_dom[Counter(sim_causes).most_common(1)[0][0]] += 1
    print("=== 전수 미충족 gold-write 원인 census (%s·DB-basis) ===" % src)
    print("실패 sim: %d · 미충족 write: %d" % (n_sim, n_write))
    print("\n[A무시도 vs B시도-오답]")
    for k, v in a_vs_b.most_common():
        print("  %-30s %d (%.0f%%)" % (k, v, 100*v/max(n_write, 1)))
    print("\n[정밀 원인 분해]")
    for k, v in sorted(cause.items(), key=lambda x: -x[1]):
        print("  %-42s %d (%.1f%%)" % (k, v, 100*v/max(n_write, 1)))
    print("\n[sim 지배원인 (미충족 최다 클래스)]")
    for k, v in per_sim_dom.most_common():
        print("  %-4s %d sim (%.0f%%)" % (k, v, 100*v/max(sum(per_sim_dom.values()), 1)))
    print("\n[레버·신뢰도 매핑]")
    print("  A1 REACH        → FIND-enumerate(HARD·강제열거)          · 신뢰=선택술어 정확도 의존")
    print("  A2 COVERAGE     → CP5 리마인더(SOFT) or 결정론 H_min(qty) · 신뢰=낮음(R3 obligation 43%/0-full)")
    print("  B1 compute      → COMPUTE 키스톤(결정론·ABox)             · 신뢰=높음(755replay 90.9%)")
    print("  B2 reference    → reference_filter(⋈·결정론)             · 신뢰=높음(C78 82% unique)")
    print("  B3 F3enum       → schema-classify 스킬(Track B/SFT)       · 신뢰=낮음(prompt-ceiling·C100)")
    print("  B4 judgment/B5 gather → ASK/경계/user-원천                 · 신뢰=경계(결정론 불가)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="단일 실행 results gz(floor). 미지정=C:/tmp/traj frontier")
    a = ap.parse_args()
    census(a.results)
