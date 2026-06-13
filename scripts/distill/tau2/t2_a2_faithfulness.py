#!/usr/bin/env python
"""P-A2 runtime faithfulness 게이트: cross-stage NL-gloss <-> source-clause entailment.

처방 출처 = `relwork_nlformalize_2026_06_14.md` §8b (Do-LLMs-Game `2604.19459` + FormalAlign `2410.10135`).
**왜 필요한가 (replay 사각지대)**: replay(t2_compliance)는 *행동* 검사 — gold 궤적 위 over/under-deny만 본다.
  ⇒ source 정책에 *근거 없는* 게이트(fabricated)라도 gold 궤적 위에서 우연히 옳게 발화하면 replay를 통과한다
  (axiom-fabrication의 게이트판). 또 술어를 잘못 옮겼는데 내부일관(silent mistranslation)인 것도 통과한다.
이 게이트는 컴파일된 GATE_SPEC의 *각 게이트*가 source 정책 clause에 의해 **entailed**되는지 별도로 검사:
  - 어떤 clause도 게이트를 함의하지 않으면 = **FABRICATED** (replay가 못 잡는 1급 — cross-stage로 포착).
  - 함의가 애매하면 = **UNCERTAIN** (silent-mistranslation 잔차 = 비가역 → abstain).
  - 함의 clause 존재 = **SUPPORTED**.
FABRICATED/UNCERTAIN 게이트가 하나라도 있으면 그 spec은 신뢰불가 → **abstain->HITL** (F6 risk-coverage).
이는 replay(behavioral, level-2)에 *직교·가산*되는 level-3(cross-stage) 검사다.

판정기(entailment)는 3중 플러그인:
  --judge name:endpoint:served  LLM-judge entailment (size_census와 동일 호출 관용구; openrouter or 로컬 vllm)
  --judgments FILE              사전계산 판정 {"gate":..,"supported":bool,"clause":int,"confidence":float} 줄들
  --lexical                     키워드-overlap 폴백 (결정론·무네트워크 — 약프록시, 판정기 부재 시 하한)
  --selftest                    내장 (policy, fabricated-gate 1개 주입 spec)로 로직 자기검증 (lexical, 무네트워크)

Usage:
  t2_a2_faithfulness.py --spec specs/retail_gate_spec_fable5.json --policy retail_policy.md \
      --judge gpt41:https://openrouter.ai/api/v1:openai/gpt-4.1 --out faith_retail.json
  t2_a2_faithfulness.py --selftest
"""
import argparse, json, os, re, sys, urllib.request

STOP = {"this", "that", "user", "must", "have", "been", "with", "before", "they", "must",
        "when", "then", "from", "into", "your", "will", "shall", "which", "their", "them",
        "only", "also", "such", "than", "each", "after", "these", "those", "does"}


def kws(text):
    return {w for w in re.findall(r"[a-z_]+", text.lower()) if len(w) > 3 and w not in STOP}


def kind_of(g):
    return ("satisfier" if g.get("satisfiers") else
            "ask" if g.get("ask") else "terminal" if g.get("terminal") else "gate")


def gloss(gid, g):
    """게이트 -> 자연어 1문장 (judge/검색용). predicate + 적용도구 + 집행종류."""
    applies = ", ".join(g.get("applies_to", []) or [])
    enf = kind_of(g)
    sat = ", ".join((g.get("satisfiers") or {}).keys())
    enf_str = (f"established via {sat}" if sat else
               g.get("ask") if g.get("ask") else
               g.get("terminal") if g.get("terminal") else "enforced as a gate")
    return (f'Gate {gid}: before [{applies}] may run, the system must ensure that '
            f'"{g.get("predicate", "")}" holds ({enf_str}).')


def segment(policy_text):
    """정책 텍스트 -> clause 리스트. 마크다운 헤더/빈줄/초단문 제거, 긴 줄은 문장 분해."""
    clauses = []
    for raw in policy_text.splitlines():
        line = raw.strip().lstrip("#-*0123456789. \t")
        if not line or len(line) < 18:
            continue
        # 긴 줄은 문장경계로 추가 분해
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            sent = sent.strip()
            if len(sent) >= 18:
                clauses.append(sent)
    return clauses


def retrieve(gtext, clauses, k=5):
    """게이트-gloss와 키워드 overlap 높은 후보 clause top-k -> [(idx, score, text)]."""
    gk = kws(gtext)
    scored = []
    for i, c in enumerate(clauses):
        ck = kws(c)
        if not ck:
            continue
        inter = len(gk & ck)
        if inter:
            # 게이트 키워드 중 clause가 덮은 비율 (recall 지향 — 짧은 clause penalize 안 함)
            score = inter / max(len(gk), 1)
            scored.append((i, score, c))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


JUDGE_SYS = """You verify FAITHFULNESS of a compiled policy gate against the source policy text.
Given ONE gate (natural-language gloss) and several candidate source-policy clauses, decide whether
ANY clause ENTAILS the gate's requirement — i.e. the policy actually mandates this restriction.
A gate is SUPPORTED only if a clause states (or directly implies) the same precondition over the same
kind of action. Judge by MEANING, not surface wording (e.g. "authenticate the user" entails an
"authenticated user identity" gate even if the words differ). A gate with NO entailing clause anywhere
in the policy is FABRICATED (the compiler invented a rule the policy does not contain).
Output ONLY JSON: {"supported": true|false, "evidence": "<verbatim quote of the entailing clause, or
empty string>", "confidence": 0.0-1.0, "reason": "<short>"}."""


def judge_call(endpoint, model, gtext, clauses):
    # 정책 전체 clause를 번호와 함께 제시 — retrieval recall 병목 제거(짧은 SOP 정책 전제, 필요시 청크).
    clause_block = "\n".join(f"- {c}" for c in clauses) or "(empty policy)"
    usr = f"GATE TO VERIFY:\n{gtext}\n\nFULL SOURCE POLICY (clause per line):\n{clause_block}"
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": 0.0, "max_tokens": 300,
               "messages": [{"role": "system", "content": JUDGE_SYS},
                            {"role": "user", "content": usr}],
               "response_format": {"type": "json_object"}}
    if "openrouter" not in endpoint:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    hdr = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}
    if "openrouter" in endpoint:
        hdr["Authorization"] = "Bearer " + os.environ["OPENROUTER_API_KEY"]
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=hdr)
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    s = txt.find("{")
    return json.loads(txt[s:txt.rfind("}") + 1])


# lexical-폴백 임계 (판정기 부재 시 하한). 약프록시임을 명시.
LEX_HI, LEX_LO = 0.34, 0.17


def classify_lexical(cands):
    best = cands[0][1] if cands else 0.0
    ev = cands[0][2] if cands else None
    if best >= LEX_HI:
        return "supported", best, ev
    if best < LEX_LO:
        return "fabricated", best, None
    return "uncertain", best, ev


def classify_judge(v):
    sup, conf = bool(v.get("supported")), float(v.get("confidence", 0.0))
    ev = (v.get("evidence") or "").strip() or None
    if sup and conf >= 0.5:
        return "supported", conf, ev
    if (not sup) and conf >= 0.5:
        return "fabricated", conf, None
    return "uncertain", conf, ev


def audit(spec, clauses, mode, judge=None, judgments=None):
    gates = {k: v for k, v in spec.items() if not k.startswith("_")}
    rows = []
    for gid, g in gates.items():
        gtext = gloss(gid, g)
        if mode == "judgments":
            verdict, conf, ev = classify_judge(judgments.get(gid, {}))
        elif mode == "judge":
            # judge에 정책 전체 clause를 넘김(retrieval 병목 제거) — verbatim 근거 quote를 받음.
            verdict, conf, ev = classify_judge(judge_call(judge[1], judge[2], gtext, clauses))
        else:  # lexical: predicate(정책-의미) 키워드 기준 검색 — gloss 도구명 토큰 희석 방지
            cands = retrieve(g.get("predicate", "") + " " + gid.replace("_", " "), clauses)
            verdict, conf, ev = classify_lexical(cands)
        rows.append({"gate": gid, "verdict": verdict, "score": round(conf, 3),
                     "gloss": gtext, "evidence": ev})
    return rows


def report(rows, abstain_thresh):
    n = len(rows)
    sup = sum(r["verdict"] == "supported" for r in rows)
    fab = sum(r["verdict"] == "fabricated" for r in rows)
    unc = sum(r["verdict"] == "uncertain" for r in rows)
    sup_frac = sup / max(n, 1)
    # 라우팅: fabricated가 하나라도 있거나 supported 비율이 임계 미만이면 spec 신뢰불가 -> abstain
    abstain = (fab > 0) or (sup_frac < abstain_thresh)
    for r in rows:
        tag = {"supported": "OK ", "fabricated": "FAB", "uncertain": "UNC"}[r["verdict"]]
        print(f"  [{tag}] {r['gate']:<22} score={r['score']:.2f}  "
              f"{('<- ' + r['evidence'][:64]) if r['evidence'] else '(no entailing clause)'}")
    print(f"[faithfulness] n={n} supported={sup} fabricated={fab} uncertain={unc} "
          f"(sup_frac={sup_frac:.2f}); spec -> {'ABSTAIN->HITL' if abstain else 'TRUST'}")
    return {"n_gates": n, "supported": sup, "fabricated": fab, "uncertain": unc,
            "sup_frac": round(sup_frac, 3), "route": "abstain" if abstain else "trust",
            "rows": rows}


# ---- selftest: 내장 (policy, fabricated 주입 spec) — lexical, 무네트워크 결정론 ----
SELFTEST_POLICY = """# Retail customer service policy
You must verify the authenticated identity of the user by email or by name and zip before any account action.
Obtain explicit confirmation from the user before every write action such as cancel or modify an order.
Deny any request whose target order belongs to a different authenticated user (single-user scope only).
Refunds are processed to the original payment method.
"""
SELFTEST_SPEC = {
    "_meta": {"note": "selftest: 3 supported + 1 fabricated"},
    "G1_AUTH_FIRST": {"predicate": "authenticated user identity",
                      "satisfiers": {"find_user_id_by_email": ["email"]},
                      "applies_to": ["cancel_pending_order", "get_order_details"]},
    "G2_CONFIRM_WRITE": {"predicate": "explicit user confirmation before write",
                         "satisfiers": {}, "ask": "list details and ask confirm",
                         "applies_to": ["cancel_pending_order", "modify_pending_order_items"]},
    "G3_SINGLE_USER": {"predicate": "target order belongs to the authenticated user",
                       "satisfiers": {}, "terminal": "deny other-user requests",
                       "applies_to": ["get_order_details"]},
    # FABRICATED: 정책에 근거 없음 — replay는 (gold 궤적서 발화 안 하면) 통과시키지만 faithfulness가 잡아야 함
    "G9_LOYALTY_TIER": {"predicate": "customer holds a gold loyalty membership tier",
                        "satisfiers": {}, "terminal": "deny non-gold customers",
                        "applies_to": ["cancel_pending_order"]},
}


def selftest():
    print("[selftest] lexical mode, embedded policy + 1 injected fabricated gate (G9_LOYALTY_TIER)")
    clauses = segment(SELFTEST_POLICY)
    rows = audit(SELFTEST_SPEC, clauses, "lexical")
    res = report(rows, abstain_thresh=1.0)
    by = {r["gate"]: r["verdict"] for r in rows}
    ok = (by.get("G9_LOYALTY_TIER") == "fabricated"
          and by.get("G1_AUTH_FIRST") == "supported"
          and res["route"] == "abstain")
    print(f"[selftest] G9 fabricated-flagged + G1 supported + route=abstain -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--policy")
    ap.add_argument("--judge", help="name:endpoint:served (openrouter or 로컬 vllm)")
    ap.add_argument("--judgments", help="사전계산 판정 jsonl")
    ap.add_argument("--lexical", action="store_true", help="키워드-overlap 폴백 (무네트워크)")
    ap.add_argument("--abstain_thresh", type=float, default=1.0,
                    help="supported 비율 이 미만이면 abstain (기본 1.0 = fabricated/uncertain 0건 요구)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not (a.spec and a.policy):
        ap.error("--spec and --policy required (or --selftest)")

    spec = json.load(open(a.spec, encoding="utf-8"))
    clauses = segment(open(a.policy, encoding="utf-8").read())

    if a.judgments:
        mode, judgments, judge = "judgments", {}, None
        for l in open(a.judgments, encoding="utf-8"):
            d = json.loads(l)
            judgments[d["gate"]] = d
    elif a.judge:
        mode, judgments = "judge", None
        # endpoint가 '://'를 포함하므로 name=첫토큰·served=끝토큰·endpoint=중간 재결합
        parts = a.judge.split(":")
        if len(parts) < 3:
            ap.error("--judge must be name:endpoint:served (endpoint may contain ://)")
        judge = (parts[0], ":".join(parts[1:-1]), parts[-1])
    else:
        mode, judgments, judge = "lexical", None, None
        if not a.lexical:
            print("[warn] 판정기 미지정 -> lexical 폴백(약프록시) 사용. 운영은 --judge 권장.", file=sys.stderr)

    rows = audit(spec, clauses, mode, judge=judge, judgments=judgments)
    res = report(rows, a.abstain_thresh)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[faithfulness] wrote {a.out}")


if __name__ == "__main__":
    main()
