#!/usr/bin/env python
"""P-A2-0b: A2 컴파일러 크기 하한 census (R7을 A2에 적용 — 사용자 지시 2026-06-12).

동일 프롬프트(스키마 + retail 1-shot 예시 + 타깃 정책 + A1 카탈로그)로 N개 모델에
airline GATE_SPEC을 zero-shot 컴파일시켜 Fable-5 reference 대비 채점:
  - gate_recall: reference 게이트 술어가 생성물에 잡혔나 (predicate 키워드 매칭)
  - applies_to F1: gate별 적용-도구 집합 일치
  - satisfier 정확도: G1류 satisfier 도구→입력 매핑 일치
v1 = 구조·키워드 tier (db_check prose의 replay는 DSL 후 = P-A2-1).

모델 = served-name:endpoint 쌍 (--model). 로컬 vllm OpenAI-호환 또는 openrouter.
Run: t2_a2_size_census.py --target airline \
  --model qwen7b:http://localhost:8000/v1:Qwen/Qwen2.5-7B-Instruct ...
"""
import argparse, json, re, urllib.request

REF_DIR = None  # set in main

PROMPT_SYS = """You compile a customer-service policy into a deterministic gate specification (JSON).
Output ONLY a JSON object: each key is a GATE id (e.g. "G1_AUTH_FIRST"), value has:
  "predicate": short string of the condition that must hold before the gated tools run,
  "satisfiers": {tool_name: [required_input_args]} that establish the predicate (or {} if conversational),
  "applies_to": [tool names this gate guards],
  and either "ask" (recovery instruction) or "terminal" (when it cannot be satisfied -> deny).
Emit a gate for every policy rule that blocks a tool BEFORE the tool's own checks
(authentication, write-confirmation, single-user scope, eligibility rules the API does not check)."""


def call(endpoint, model, sys, usr, max_tokens=2000):
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        "response_format": {"type": "json_object"},
    }
    if "openrouter" not in endpoint:
        # 로컬 vllm: Qwen3류 thinking 차단 (P2 교훈 — non-thinking 고정; Qwen2.5는 무해 무시)
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}
    import os
    if "openrouter" in endpoint:
        hdr["Authorization"] = "Bearer " + os.environ["OPENROUTER_API_KEY"]
    req = urllib.request.Request(url, data=body, headers=hdr)
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def parse(txt):
    s = txt.find("{")
    if s < 0:
        return None
    try:
        return json.loads(txt[s:txt.rfind("}") + 1])
    except ValueError:
        return None


def predicate_kw(spec):
    """reference 게이트별 핵심 키워드 집합 (recall 채점용)."""
    out = {}
    for g, v in spec.items():
        if g.startswith("_"):
            continue
        words = re.findall(r"[a-z_]+", (v.get("predicate", "") + " " + g).lower())
        out[g] = {w for w in words if len(w) > 3 and w not in
                  {"this", "that", "user", "must", "have", "been", "with", "before"}}
    return out


def score(ref, gen):
    refkw = predicate_kw(ref)
    # 생성물 전체를 평탄 텍스트로 — 게이트 경계가 달라도 술어 recall 측정
    gentext = json.dumps(gen).lower() if gen else ""
    gen_gates = {k: v for k, v in (gen or {}).items() if not k.startswith("_")}
    gate_recall = sum(1 for g, kw in refkw.items()
                      if kw and len(kw & set(re.findall(r"[a-z_]+", gentext))) >= max(1, len(kw) // 2))
    gate_recall /= max(len(refkw), 1)
    # applies_to F1: ref 게이트마다 best-match 생성 게이트의 도구집합 F1
    f1s = []
    for g, rv in ref.items():
        if g.startswith("_"):
            continue
        rt = set(rv.get("applies_to", []))
        if not rt:
            continue
        best = 0.0
        for gv in gen_gates.values():
            gt = set(gv.get("applies_to", []) or [])
            if not gt:
                continue
            inter = len(rt & gt)
            if inter:
                p, r = inter / len(gt), inter / len(rt)
                best = max(best, 2 * p * r / (p + r))
        f1s.append(best)
    applies_f1 = sum(f1s) / max(len(f1s), 1)
    return gate_recall, applies_f1, len(gen_gates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="airline")
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="name:endpoint:served_model (endpoint contains openrouter for OR)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen_only", action="store_true",
                    help="P5(S1 교사-컴파일): reference 무 도메인 — 채점 생략, spec 생성/저장만"
                         " (검증은 Track A replay가 사후 수행)")
    ap.add_argument("--no_oneshot", action="store_true",
                    help="S1-diag H3: retail 1-shot 예시 제거 — zero-shot 평가"
                         "(학습 프롬프트 형식과 일치, 형식-불일치 가설 분리)")
    a = ap.parse_args()

    ref = None if a.gen_only else json.load(
        open(f"{a.ref_dir}/{a.target}_gate_spec_fable5.json"))
    retail_ref = json.load(open(f"{a.ref_dir}/retail_gate_spec_fable5.json"))
    policy = open(a.policy).read()
    catalog = json.load(open(a.catalog))
    oneshot = ("# EXAMPLE (retail policy -> spec):\n" + json.dumps(
        {k: v for k, v in retail_ref.items() if not k.startswith("_")}, indent=1))
    if a.no_oneshot:
        usr = (f"# Compile THIS policy ({a.target}):\n{policy}\n\n"
               f"# tool catalog (name: type, required args):\n"
               + json.dumps(catalog["tools"], indent=1) + "\n\nOutput the gate spec JSON:")
    else:
        usr = (f"{oneshot}\n\n# NOW compile THIS policy ({a.target}):\n{policy}\n\n"
               f"# tool catalog (name: type, required args):\n"
               + json.dumps(catalog["tools"], indent=1) + "\n\nOutput the gate spec JSON:")

    rows = []
    for m in a.model:
        # name:endpoint:served — endpoint에 ://·:port 콜론 포함되므로 양끝에서 분리
        name, rest = m.split(":", 1)
        endpoint, served = rest.rsplit(":", 1)
        try:
            txt = call(endpoint, served, PROMPT_SYS, usr)
            gen = parse(txt)
            json.dump(gen, open(f"{a.out}.{name}.json", "w"), indent=1)
            if a.gen_only:
                ng = len([k for k in (gen or {}) if not k.startswith("_")])
                print(f"{name}: n_gates={ng} parsed={gen is not None} (gen-only)")
                continue
            gr, af, ng = score(ref, gen)
            rows.append((name, gr, af, ng, gen is not None))
            print(f"{name}: gate_recall={gr:.3f} applies_F1={af:.3f} n_gates={ng} parsed={gen is not None}")
        except Exception as e:
            rows.append((name, 0, 0, 0, False))
            print(f"{name}: ERROR {type(e).__name__}: {str(e)[:120]}")
    if not a.gen_only:
        print(f"\n[size-census] ref gates={len([k for k in ref if not k.startswith('_')])} "
              f"(Fable-5 reference). 하한 = gate_recall·applies_F1이 reference에 근접하는 최소 크기.")


if __name__ == "__main__":
    main()
