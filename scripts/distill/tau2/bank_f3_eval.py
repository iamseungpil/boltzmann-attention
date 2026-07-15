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

def make_prompt(case, schema):
    opts = "\n".join("- '%s': %s" % (v, desc) for v, desc in schema[case["field"]])
    return ("You classify a bank customer's situation into exactly ONE allowed category.\n"
            "Allowed values for %s (choose exactly one, output the value verbatim):\n%s\n\n"
            "Customer's statements:\n%s\n\nTransaction facts: %s\n\n"
            "Output ONLY the single exact category value, nothing else." %
            (case["field"], opts, case["nl"], json.dumps(case["txn"], ensure_ascii=False)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default="f3_cases.jsonl"); ap.add_argument("--cases", default="f3_cases.jsonl")
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))
    schema = extract_schema(files)
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
        cases = []
        for k, lst in strata.items():
            cases.extend(lst[:per])
        by_attend = defaultdict(lambda: [0, 0]); by_field = defaultdict(lambda: [0, 0])
        for i, c in enumerate(cases):
            try:
                r = cl.chat.completions.create(model=a.model, messages=[{"role": "user", "content": make_prompt(c, schema)}],
                                               temperature=0, max_tokens=30)
                pred = r.choices[0].message.content.strip().strip("'\"").lower()
            except Exception as e:
                print("ERR", type(e).__name__, e); break
            ok = c["gold"].lower() in pred or pred in c["gold"].lower()
            key = "attend(gold NL에)" if c["gold_attend"] else "non-attend(정책추론)"
            by_attend[key][0] += int(ok); by_attend[key][1] += 1
            by_field[c["field"]][0] += int(ok); by_field[c["field"]][1] += 1
            if i < 6 or (i % 40 == 0):
                print("[%d/%d] %s attend=%s gold=%s pred=%s %s" % (i, len(cases), c["field"], c["gold_attend"], c["gold"], pred[:26], "✓" if ok else "✗"), flush=True)
        print("\n=== base 32B 스키마-분류 정확도 (%s·n=%d) ===" % (a.model, len(cases)))
        print("  [gold-attend별]")
        for k, (ok, tot) in by_attend.items():
            print("    %-24s %d/%d = %.1f%%" % (k, ok, tot, 100 * ok / max(tot, 1)))
        print("  [필드별]")
        for k, (ok, tot) in by_field.items():
            print("    %-20s %d/%d = %.1f%%" % (k, ok, tot, 100 * ok / max(tot, 1)))
        print("  판정: attend 높음=제공스키마 활용OK(그 부분 prompt-closable)·attend 낮음=스키마 무시(prior-override·SFT정당·[[42]])")
        print("        non-attend 낮음=정책추론 잔여(F3코어). Track B 표적=attend-gap + non-attend 둘 다 학습.")

if __name__ == "__main__":
    main()
