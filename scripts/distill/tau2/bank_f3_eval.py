# -*- coding: utf-8 -*-
"""bank_f3_eval.py — 게이트3(정본 gate-2): base 모델 "스키마-대령→분류" NL→enum eval (2026-07-16).

질문([[42]]): base 32B가 *제공된 enum 스키마(정의)*를 대령받고 고객 NL을 gold enum으로 분류하나,
아니면 prior로 덮나. Track B(스키마-분류 스킬)의 make-or-break 선진단.
case 3분류: attend(gold토큰 NL에)·prior-conflict(agent surface≠gold)·inference(NL 미-disambig).

  --build : 궤적서 case-set 추출→JSONL (무료·서버 불요). 스키마정의=tool content서 추출.
  --run   : case를 base 모델(OpenAI 호환)에 질의·정확도 채점 (서버 필요·[[09]] 32B 로컬무료).

사용:
  py bank_f3_eval.py --build --out f3_cases.jsonl
  py bank_f3_eval.py --run  --cases f3_cases.jsonl --base http://localhost:8140/v1 --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
"""
import json, glob, re, sys, io, os, argparse
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
fam = lambda n: re.sub(r"_\d+$", "", str(n))
FIELDS = ["dispute_reason", "card_action", "dispute_category"]

def extract_schema(files):
    """tool content서 각 enum 필드의 정의 블록 추출 (values + 설명·도메인지식=제공맥락)."""
    sch = {}
    rx = {f: re.compile(re.escape(f) + r".{0,80}?(?:one of|select).{0,1200}", re.I | re.S) for f in FIELDS}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                c = str(m.get("content"))
                for fld in FIELDS:
                    if fld in sch:
                        continue
                    mt = rx[fld].search(c)
                    if mt:
                        # 값 리스트만 깔끔히: '- 'value': desc' 라인 수집
                        blk = mt.group(0)
                        vals = re.findall(r"'([a-z_]+)'\s*:\s*([^\n'-][^\n]{0,90})", blk)
                        if len(vals) >= 3:
                            sch[fld] = [(v, desc.strip()) for v, desc in vals][:12]
            if len(sch) == len(FIELDS):
                return sch
    return sch

def toks(v):
    return set(t for t in re.split(r"[_\s]+", str(v).lower()) if len(t) >= 4)

def label_case(gold, agent, nl):
    tg, ta = toks(gold), toks(agent)
    ag = sum(1 for t in ta if t in nl) / max(len(ta), 1)
    gg = sum(1 for t in tg if t in nl) / max(len(tg), 1)
    if gg > ag:
        return "attend(gold NL에 있음)"
    if ag > gg:
        return "prior-conflict(agent surface≠gold)"
    return "inference(NL 미-disambig)"

def build_cases(files, schema):
    cases = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0): continue
            if tuple(ri.get("reward_basis") or []) != ("DB",): continue
            nl = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") == "user")
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
                aa = asub.get(t, {})
                for fld in FIELDS:
                    if fld not in schema: continue
                    gv = ga.get(fld)
                    if gv is None: continue
                    av = aa.get(fld)
                    txn = {k: ga.get(k) for k in ("transaction_date", "disputed_amount", "merchant", "description", "transaction_type") if ga.get(k)}
                    lab = label_case(gv, av, nl.lower()) if av and str(av).lower() != str(gv).lower() else "agent-correct"
                    tg = toks(gv); gold_attend = bool(tg) and all(t in nl.lower() for t in tg)  # gold 토큰 전부 NL에
                    cases.append({"field": fld, "gold": str(gv), "agent": str(av) if av else None,
                                  "label": lab, "gold_attend": gold_attend, "nl": nl[:1500], "txn": txn,
                                  "options": [v for v, _ in schema[fld]]})
    return cases

def _short_nl(nl, n=500):
    return re.sub(r"\s+", " ", str(nl)).strip()[:n]


def build_fewshot(demo_pool, schema, k_per_class=1):
    """필드별 few-shot 데모 prefix: 각 enum 클래스 예시 + prior-conflict 반례(있으면).
    demo_pool = test와 disjoint한 case 리스트. 반환 {field: prefix_str}."""
    pref = {}
    for fld in schema:
        seen = {}
        demos = []
        # 클래스별 1개(다양성) + prior-conflict 우선
        pool = [c for c in demo_pool if c["field"] == fld]
        pool.sort(key=lambda c: 0 if c.get("label", "").startswith("prior-conflict") else 1)
        for c in pool:
            g = c["gold"]
            if seen.get(g, 0) >= k_per_class:
                continue
            seen[g] = seen.get(g, 0) + 1
            tag = " (NOTE: customer's words may suggest 'fraud' but the correct category is per the definition)" if c.get("label", "").startswith("prior-conflict") else ""
            demos.append("Customer: %s\nTransaction: %s\nCorrect %s: %s%s" %
                         (_short_nl(c["nl"], 400), json.dumps(c["txn"], ensure_ascii=False), fld, g, tag))
            if len(demos) >= 10:
                break
        pref[fld] = "\n\nHere are worked examples:\n\n" + "\n---\n".join(demos) + "\n\n"
    return pref


def make_binary_prompt(case, cand_val, cand_desc):
    """후보 1개 격리 이진 판별: 전체 enum 안 보여줌(prior salience 차단)·이 후보 정의만 대령.
    도메인일반: 후보값/정의=ABox 스키마·판별=격리 sub-call([[05]]·[[10]] LLM=비교기만)."""
    return ("You are checking whether a customer's described situation matches ONE specific category.\n\n"
            "Category: '%s'\n"
            "Definition of this category: %s\n\n"
            "Customer's statements:\n%s\n\n"
            "Transaction facts: %s\n\n"
            "Considering ONLY this category's definition above, does the customer's situation match "
            "the category '%s'? Answer with exactly one word: yes or no." %
            (cand_val, cand_desc, case["nl"], json.dumps(case["txn"], ensure_ascii=False), cand_val))


def _yes(text):
    t = str(text).strip().lower().strip("'\".,")
    return t.startswith("yes") or t == "y" or (" yes" in (" " + t) and "no" not in t.split()[:1])


def run_compareloop(cl, model, cases, schema):
    """per-candidate COMPARE-or-ASK loop (결정론 컨트롤러·§0):
      후보마다 격리 이진 y/n → 유일매칭 select · else(0/다수) ASK. prior 디폴트 금지.
    grade: correct=유일매칭==gold · ask=매칭≠1 · wrong=유일매칭≠gold.
    collapse 진단: gold_attend(NL에 gold토큰 있음)인데 loop가 놓치나·ASK로 gold 회수되나."""
    by_field = defaultdict(lambda: [0, 0, 0, 0])   # [correct, ask, wrong, total]
    by_attend = defaultdict(lambda: [0, 0, 0, 0])
    sel_dist = defaultdict(Counter)                # 필드별 유일-select 값 분포(mode-collapse 진단)
    ask_recover = defaultdict(lambda: [0, 0])      # ASK 케이스 중 gold∈matches (회수가능)
    match_cnt = defaultdict(Counter)               # 필드별 매칭 개수 분포
    for i, c in enumerate(cases):
        opts = schema[c["field"]]
        matches = []
        for cand_val, cand_desc in opts:
            try:
                r = cl.chat.completions.create(model=model,
                        messages=[{"role": "user", "content": make_binary_prompt(c, cand_val, cand_desc)}],
                        temperature=0, max_tokens=5)
                if _yes(r.choices[0].message.content):
                    matches.append(cand_val)
            except Exception as e:
                print("ERR", type(e).__name__, e); return None
        gold = c["gold"].lower()
        fld = c["field"]
        akey = "attend(gold NL에)" if c["gold_attend"] else "non-attend(정책추론)"
        match_cnt[fld][len(matches)] += 1
        if len(matches) == 1:
            pred = matches[0].lower()
            sel_dist[fld][pred[:24]] += 1
            hit = (gold in pred or pred in gold)
            slot = 0 if hit else 2                       # correct or wrong
        else:                                            # 0 or 다수 → ASK
            slot = 1
            gold_in = any(gold in m.lower() or m.lower() in gold for m in matches)
            ask_recover[fld][0] += int(gold_in); ask_recover[fld][1] += 1
        by_field[fld][slot] += 1; by_field[fld][3] += 1
        by_attend[akey][slot] += 1; by_attend[akey][3] += 1
        if i < 8 or (i % 40 == 0):
            tag = ["✓select", "?ASK", "✗wrong"][slot]
            print("[%d/%d] %s attend=%s gold=%s matches=%s %s" %
                  (i, len(cases), fld, c["gold_attend"], c["gold"], matches or "[]", tag), flush=True)
    print("\n=== COMPARE-or-ASK loop 결과 (%s·n=%d) ===" % (model, len(cases)))
    print("  [필드별  correct / ASK / wrong / total]")
    for k, (co, ak, wr, tot) in by_field.items():
        t = max(tot, 1)
        print("    %-18s correct %d(%.0f%%) · ASK %d(%.0f%%) · wrong %d(%.0f%%)  [n=%d]" %
              (k, co, 100*co/t, ak, 100*ak/t, wr, 100*wr/t, tot))
    print("  [gold-attend별]")
    for k, (co, ak, wr, tot) in by_attend.items():
        t = max(tot, 1)
        print("    %-24s correct %.0f%% · ASK %.0f%% · wrong %.0f%%  [n=%d]" %
              (k, 100*co/t, 100*ak/t, 100*wr/t, tot))
    print("  [매칭개수 분포 (0=전부no→ASK · 1=유일select · ≥2=다수→ASK)]")
    for fld, mc in match_cnt.items():
        print("    %-18s %s" % (fld, dict(sorted(mc.items()))))
    print("  [유일-select 값 분포 Top4 (mode-collapse 비교: one-shot dispute_reason=fraud 98%%)]")
    for fld, sc in sel_dist.items():
        print("    %-18s %s" % (fld, dict(sc.most_common(4))))
    print("  [ASK 회수가능성: gold∈matches / ASK수]")
    for fld, (g, t) in ask_recover.items():
        print("    %-18s %d/%d = %.0f%% (ASK했어도 정답 후보엔 있음)" % (fld, g, t, 100*g/max(t, 1)))
    print("  판정: dispute_reason wrong%%가 one-shot 98%% fraud서 크게 떨어지면 → loop가 collapse 깸(SFT 불요·[[41]]).")
    print("        wrong 여전히 높으면 → 이진 판별도 prior에 오염 → SFT fallback([[42]]).")
    return by_field


def make_prompt(case, schema, strict=False, fewshot_prefix=None):
    opts = "\n".join("- '%s': %s" % (v, desc) for v, desc in schema[case["field"]])
    hdr = ("You classify a bank customer's situation into exactly ONE allowed category.\n")
    if strict:  # anti-prior 강화(리뷰/[[08]] robustness): 프롬프트-엔지니어링이 prior-override 고치나
        hdr += ("IMPORTANT: Decide STRICTLY from the definitions below and the customer's exact words. "
                "Do NOT default to 'fraud'/'unauthorized' unless the definition and the customer's words specifically match it. "
                "Read each definition and pick the one that fits the described situation.\n")
    fs = (fewshot_prefix.get(case["field"], "") if fewshot_prefix else "")
    return (hdr +
            "Allowed values for %s (choose exactly one, output the value verbatim):\n%s\n%s\n"
            "Now classify THIS case.\nCustomer's statements:\n%s\n\nTransaction facts: %s\n\n"
            "Output ONLY the single exact category value, nothing else." %
            (case["field"], opts, fs, case["nl"], json.dumps(case["txn"], ensure_ascii=False)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default="f3_cases.jsonl"); ap.add_argument("--cases", default="f3_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--fewshot", action="store_true", help="각 enum 클래스 예시+prior-conflict 반례 few-shot")
    ap.add_argument("--compareloop", action="store_true", help="§0 per-candidate COMPARE-or-ASK loop(후보별 격리 이진→select/ASK)")
    a = ap.parse_args()
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))
    # 스키마: 궤적 있으면 추출·없으면(리모트) 커밋된 f3_schema.json 로드
    _schpath = os.path.join(os.path.dirname(a.cases) or ".", "f3_schema.json")
    if files:
        schema = extract_schema(files)
    elif os.path.exists(_schpath):
        schema = {k: [tuple(x) for x in v] for k, v in json.load(open(_schpath, encoding="utf-8")).items()}
    elif os.path.exists("f3_schema.json"):
        schema = {k: [tuple(x) for x in v] for k, v in json.load(open("f3_schema.json", encoding="utf-8")).items()}
    else:
        schema = {}
    if a.build or not os.path.exists(a.cases):
        cases = build_cases(files, schema)
        json.dump(schema, open("f3_schema.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(a.out, "w", encoding="utf-8") as fh:
            for c in cases:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        wrong = [c for c in cases if c["label"] != "agent-correct"]
        print("=== case-set 구축 (스키마 %d필드·전 %d·오답 %d) ===" % (len(schema), len(cases), len(wrong)))
        print("  스키마 필드:", {k: len(v) for k, v in schema.items()})
        print("  오답 label 분포:", dict(Counter(c["label"] for c in wrong)))
        print("  → %s (run 준비완료·서버 시 --run)" % a.out)
        if not a.run: return
    if a.run:
        try:
            from openai import OpenAI
        except Exception:
            print("openai 미설치 — 리모트서 실행 권장"); return
        cl = OpenAI(base_url=a.base, api_key="x")
        allc = [json.loads(l) for l in open(a.cases, encoding="utf-8")]
        # 층화 샘플: (field × gold_attend)별 균형 (기본 상한 --max, 0=전체)
        strata = defaultdict(list)
        for c in allc:
            strata[(c["field"], c["gold_attend"])].append(c)
        cap = a.max or 10**9
        per = max(1, cap // max(len(strata), 1))
        cases = []; demo_pool = []
        for k, lst in strata.items():
            cases.extend(lst[:per])
            demo_pool.extend(lst[per:per + 40])       # test와 disjoint한 데모풀
        if a.compareloop:
            run_compareloop(cl, a.model, cases, schema)
            return
        fs_prefix = build_fewshot(demo_pool, schema) if a.fewshot else None
        if a.fewshot:
            print("  [few-shot] 데모풀 %d·필드별 예시:" % len(demo_pool),
                  {k: v.count("Correct ") for k, v in fs_prefix.items()})
        by_attend = defaultdict(lambda: [0, 0]); by_field = defaultdict(lambda: [0, 0])
        pred_dist = defaultdict(Counter)              # 필드별 예측 분포(mode-collapse 진단·[[08]])
        for i, c in enumerate(cases):
            try:
                r = cl.chat.completions.create(model=a.model, messages=[{"role": "user", "content": make_prompt(c, schema, strict=a.strict, fewshot_prefix=fs_prefix)}],
                                               temperature=0, max_tokens=30)
                pred = r.choices[0].message.content.strip().strip("'\"").lower()
            except Exception as e:
                print("ERR", type(e).__name__, e); break
            ok = c["gold"].lower() in pred or pred in c["gold"].lower()
            key = "attend(gold NL에)" if c["gold_attend"] else "non-attend(정책추론)"
            by_attend[key][0] += int(ok); by_attend[key][1] += 1
            by_field[c["field"]][0] += int(ok); by_field[c["field"]][1] += 1
            pred_dist[c["field"]][pred[:24]] += 1
            if i < 6 or (i % 40 == 0):
                print("[%d/%d] %s attend=%s gold=%s pred=%s %s" % (i, len(cases), c["field"], c["gold_attend"], c["gold"], pred[:26], "✓" if ok else "✗"), flush=True)
        mode = "few-shot" if a.fewshot else ("strict" if a.strict else "zero-shot")
        print("\n=== base 32B 스키마-분류 정확도 [%s] (%s·n=%d) ===" % (mode, a.model, len(cases)))
        print("  [gold-attend별]")
        for k, (ok, tot) in by_attend.items():
            print("    %-24s %d/%d = %.1f%%" % (k, ok, tot, 100 * ok / max(tot, 1)))
        print("  [필드별]")
        for k, (ok, tot) in by_field.items():
            print("    %-20s %d/%d = %.1f%%" % (k, ok, tot, 100 * ok / max(tot, 1)))
        print("  [예측 분포 Top3 (mode-collapse 진단)]")
        for fld, pc in pred_dist.items():
            print("    %-18s %s" % (fld, dict(pc.most_common(3))))
        print("  판정: attend 높음=제공스키마 활용OK(그 부분 prompt-closable)·attend 낮음=스키마 무시(prior-override·SFT정당·[[42]])")
        print("        non-attend 낮음=정책추론 잔여(F3코어). Track B 표적=attend-gap + non-attend 둘 다 학습.")

if __name__ == "__main__":
    main()
