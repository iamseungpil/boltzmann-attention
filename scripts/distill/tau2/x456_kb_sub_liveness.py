# -*- coding: utf-8 -*-
r"""x456 — 수리된 **KB 축 서브콜**이 살아났나 (2026-08-21·격리·무료·C581 의 [[62]] 측정)

## 왜
C581: 날조 안전판의 술어가 `Record ID:` 계수(**DB 레코드 덤프 포맷 전용**)라서, getter 가 KB 검색인
선언은 **52/52 = 100%** 가 `source=0` → **항상 폐기**였다. 수리는 *"배열 근거 계약이 선언돼 있고
집행 중이면 계수기가 서지 않는다"* 다. 그런데 산 것은 **채널 생존**뿐이고, 서브의 답이 관문1
(`_ground_operands`)을 통과하는지는 **한 번도 관측된 적이 없다** — 늘 그 전에 버려졌기 때문이다.

여기서 재는 것은 딱 하나: **서브가 KB 를 읽어 낸 `components` 가 관문1 을 통과하는가.**
통과율이 0 이면 수리는 채널만 열었을 뿐 아무것도 사지 않은 것이고, 그 사실을 그대로 적는다.

## 팔 (한 변수만 다르다)
    A_repaired   현행 코드 · `T2_SG_GROUND=1`   → 계수기 미적용 · 관문1 이 심사
    B_prerepair  현행 코드 · `T2_SG_GROUND` 미설정 → **수리 전과 같은 경로**(계수기가 폐기)
    C_docs       `T2_SG_GROUND=1` + `T2_SG_DOCS=1` → 엔진이 A3 `isolate.docs` 범위를 잘라
                 전달·서브는 형식화만(검색 0·[[71]]·2026-08-21 추가). A 와의 차이 = 전달 방식 하나.
⇒ B 는 **부정통제**([[57]]): 같은 코드·같은 재료인데 B 만 전부 None 이면 갈린 원인은 계수기다.
⇒ C 의 판정선 = A 의 관문1 생존(기준 4/17). 생존이 안 오르면 선언은 샀지만 아무것도 못 산 것.

## 재료의 출처
`ref_params`(= `savings_account_type` · `customer_products`)는 **실제 궤적에서 에이전트 자신이 낸
인자**를 쓴다(`sim_results/*.results.json.gz` 의 `get_correct_savings_apy` tool_call). 우리가 짓지
않는다. gold 는 읽지 않는다([[23]]) — 이 스크립트는 `reward_info` 를 열지 않는다.
도구는 라이브와 **같은 환경**을 쓴다(`x448.Sandbox` 재사용·사본 금지 [[67]]).

## ⚠별건 기록 — 이 선언의 getter 는 `KB_search_bm25` 다 (→ 2026-08-21 해소)
[[71]] 은 *"전달은 선언된 id → `shell cat`, bm25·embedding 은 **baseline** 이지 우리 방식이 아니다"*
로 확정됐는데, `get_correct_savings_apy.isolate.getter_tools` 는 **`["KB_search_bm25"]`** 다.
A/B 팔은 **라이브를 있는 그대로 재는 것**이 목적이라 그 선언을 그대로 쓴다([[62]] 정보-맞춘 격리).
★그 색인이 저작·검산됐다(커밋 58c26b84 `isolate.docs`·클래스 38·문서 101·범위 152·앵커 152/152)
— C_docs 팔이 바로 그 *"선언된 id 정확 집기"* 팔이다(x448 의 `B_shell` ↔ `R_bm25` 대조 동형).
bm25(A) 는 이제 이 표에서 **baseline 자리**다.

## 채점 (닫힌 술어만)
    returned      서브가 dict 를 돌려줬나 (수리 전 = 항상 None)
    n_components  배열 길이
    gate1_kept    관문1 `_ground_operands` 가 **드롭하지 않은** component 수
    gate1_flags   드롭 사유 문면(엔진이 만든 것·도메인 판단 0)
⛔APY 값이 맞는지는 **여기서 판정하지 않는다** — gold 를 안 보기 때문이다([[69]] reward 가 기준).

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x456_kb_sub_liveness.py --port 8141
"""
import argparse
import collections
import glob
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_scaffold_get as SG            # noqa: E402  피측정 대상(정본)
import x448_index_vs_all_iso as IVA     # noqa: E402  Sandbox 재사용([[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
BASE = os.path.join(REP, "sim_results")
DECL = "get_correct_savings_apy"


def declaration():
    """선언은 A3 정본에서 읽는다 — 코드에 적지 않는다([[71]] 계약 2항)."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("scaffold_get_tools") or []):
        if s.get("name") == DECL:
            return s
    raise SystemExit("선언 %s 없음" % DECL)


def harvest_refs(iso, limit):
    """실제 궤적에서 **에이전트 자신이 낸** ref 인자를 모은다(중복 제거·gold 미참조)."""
    want = [k for k in (iso.get("ref_params") or [])]
    out, seen = [], set()
    for p in sorted(glob.glob(os.path.join(BASE, "*.results.json.gz")), reverse=True):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            for m in (s.get("messages") or []):
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") != DECL:
                        continue
                    ar = tc.get("arguments") or {}
                    ref = {k: ar[k] for k in want if ar.get(k)}
                    if len(ref) != len(want):
                        continue
                    key = json.dumps(ref, sort_keys=True, ensure_ascii=False)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"task": s.get("task_id"), "src": os.path.basename(p), "ref": ref})
                    if len(out) >= limit:
                        return out
    return out


class _Orch(object):
    """서브·관문1 이 요구하는 최소 표면.

    `agent.tools`/`agent.llm`/`agent.llm_args` 는 서브 생성이 쓰고,
    **`environment`** 는 관문1 이 쓴다 — `_corpus_texts` 가 `orch.environment.domain_name` 으로
    KB 코퍼스를 찾는다. 도메인 이름을 코드에 적지 않으려고 **라이브와 같은 env 객체**를 그대로
    물려준다([[25]] 같은 환경·[[05]] 리터럴 0).
    """

    def __init__(self, tool_names, model, base, env):
        import types
        # ★도구는 **env 의 실물 객체**여야 한다(2026-08-21 1차 실행이 여기서 죽었다):
        #   `la.generate` 가 `tool.openai_schema` 를 읽으므로 이름만 가진 대역은 0라운드에서
        #   `AttributeError` 로 전부 실패한다 — 그러면 *"서브가 답을 못 낸다"* 로 오독된다([[55]]).
        want = set(tool_names or [])
        tools = [t for t in (env.get_tools() or []) if getattr(t, "name", None) in want]
        missing = want - {getattr(t, "name", None) for t in tools}
        if missing:
            raise SystemExit("선언 getter 가 env 에 없다: %r" % sorted(missing))
        # ★모델 문자열·인자도 **라이브와 같아야** 한다(2026-08-21 2차 실행이 여기서 죽었다):
        #   `t2_run_gated.py:317-318` 축자 — `llm = f"openai/{agent_model}"` ·
        #   `llm_args = {"api_base": agent_base, "api_key": "dummy", "temperature": 0.0}`.
        #   접두사 없이 넘기면 litellm 이 provider 를 못 찾아 0라운드에서 전부 실패하고,
        #   그것이 *"서브가 답을 못 낸다"* 로 오독된다([[55]] 계기부터).
        self.agent = types.SimpleNamespace(
            tools=tools, llm="openai/%s" % model,
            llm_args={"api_base": base, "api_key": "dummy", "temperature": 0.0})
        self.environment = env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
                    help="served name — litellm 문자열은 라이브와 같게 openai/<name> 으로 만든다")
    ap.add_argument("--n", type=int, default=8, help="궤적에서 모을 ref 조합 수")
    ap.add_argument("--arms", default="A_repaired,B_prerepair,C_docs",
                    help="돌릴 팔(쉼표) — 픽커 문구 반복 시 C_docs 만 재는 용도")
    ap.add_argument("--out", default="x456_kb_sub_liveness.json")
    a = ap.parse_args()

    os.environ.setdefault("OPENAI_API_BASE", "http://localhost:%d/v1" % a.port)
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:%d/v1" % a.port)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["T2_SG_ISOLATE"] = "1"
    os.environ.pop("T2_SG_ISOFB", None)      # 서브내 되먹임은 이 검정의 변수가 아니다

    d = declaration()
    iso = d.get("isolate") or {}
    print("=" * 96)
    print("x456 · 선언 %s · getter=%s · operand_keys=%s · ref=%s"
          % (DECL, iso.get("getter_tools"), iso.get("operand_keys"), iso.get("ref_params")))
    print("=" * 96)

    sb = IVA.Sandbox()
    orch = _Orch(iso.get("getter_tools") or [], a.model,
                 "http://localhost:%d/v1" % a.port, sb.env)
    # 관문1 의 KB 코퍼스가 실제로 잡히는지 **먼저** 확인한다 — 비어 있으면 모든 인용이
    # "source not found" 로 떨어져 측정이 무효가 된다([[55]] 계기부터).
    _kb = SG._corpus_texts(orch, ["kb"])
    print("관문1 KB 코퍼스: 문서 %d편 · %d자" % (len(_kb), sum(len(t) for t in _kb)))
    if not _kb:
        raise SystemExit("KB 코퍼스 0편 — 관문1 이 무엇도 통과시킬 수 없다. 측정 중단.")

    def run_env(tcs):
        """서브의 getter 호출을 **라이브와 같은 환경**에 그대로 흘린다."""
        from tau2.data_model.message import ToolMessage
        out = []
        for t in tcs:
            try:
                txt = str(sb.env.use_tool(t.name, **(t.arguments or {})))
                err = False
            except Exception as e:
                txt, err = "ERROR: %r" % (e,), True
            out.append(ToolMessage(id=t.id, role="tool", requestor="assistant",
                                   content=txt, error=err))
        return out

    refs = harvest_refs(iso, a.n)
    print("궤적에서 모은 ref 조합 %d개 (에이전트 자신이 낸 인자·gold 미참조)" % len(refs))
    for r in refs:
        print("  %-10s %s" % (r["task"], json.dumps(r["ref"], ensure_ascii=False)[:120]))
    if not refs:
        raise SystemExit("ref 조합 0 — 궤적에 이 도구 호출이 없다")

    rows = []
    # ★C_docs (2026-08-21 추가·핸드오프 §1-3): 엔진이 A3 `isolate.docs` 범위를 잘라 전달하고
    #   서브는 형식화만 한다(검색 0·[[71]]). 판정선 = A_repaired 의 관문1 생존 4/17 —
    #   생존이 안 오르면 선언은 샀지만 아무것도 못 산 것이고 **그대로 적는다**.
    #   `docs_mat` = 발화 계측([[55]] 죽은 배선 방지): None 이면 docs 전달이 안 발화한 것이라
    #   그 행은 A 와 같은 경로를 잰 것이 된다 — 집계에서 분리한다.
    _want_arms = set(x.strip() for x in a.arms.split(",") if x.strip())
    for arm, ground_on, docs_on in (("A_repaired", True, False),
                                    ("B_prerepair", False, False),
                                    ("C_docs", True, True)):
        if arm not in _want_arms:
            continue
        if ground_on:
            os.environ["T2_SG_GROUND"] = "1"
        else:
            os.environ.pop("T2_SG_GROUND", None)
        if docs_on:
            os.environ["T2_SG_DOCS"] = "1"
        else:
            os.environ.pop("T2_SG_DOCS", None)
        print("\n── %s (T2_SG_GROUND=%s · T2_SG_DOCS=%s) ─────────────────────────────"
              % (arm, "1" if ground_on else "unset", "1" if docs_on else "unset"))
        for r in refs:
            ctx = dict(r["ref"])
            try:
                got = SG._sub_fetch_formalize(orch, d, iso, ctx, run_env)
            except Exception as e:
                got = None
                print("  %-10s EXC %r" % (r["task"], e))
            comps = (got or {}).get("components") or []
            kept, flags = None, []
            if got and ground_on:
                merged = dict(ctx)
                merged.update(got)
                try:
                    flags = SG._ground_operands(orch, d, merged) or []
                except Exception as e:
                    flags = ["EXC %r" % (e,)]
                kept = len(merged.get("components") or [])
            mat = getattr(orch, "_t2_docs_mat", None) if docs_on else None
            rows.append({"arm": arm, "task": r["task"], "ref": r["ref"],
                         "returned": bool(got), "n_components": len(comps),
                         "gate1_kept": kept, "gate1_flags": flags,
                         "docs_mat": mat, "components": comps})
            print("  %-10s returned=%-5s comps=%-2d gate1_kept=%-4s%s %s"
                  % (r["task"], bool(got), len(comps), kept,
                     (" docs=%s(%s자)" % (bool(mat), (mat or {}).get("chars", 0))
                      if docs_on else ""),
                     ("; ".join(str(x) for x in flags))[:70]))

    agg = collections.defaultdict(lambda: {"n": 0, "returned": 0, "comps": 0, "kept": 0,
                                           "docs_fired": 0})
    for x in rows:
        s = agg[x["arm"]]
        s["n"] += 1
        s["returned"] += 1 if x["returned"] else 0
        s["comps"] += x["n_components"]
        s["kept"] += (x["gate1_kept"] or 0)
        s["docs_fired"] += 1 if x.get("docs_mat") else 0

    payload = {"declaration": DECL, "getter_tools": iso.get("getter_tools"),
               "n_refs": len(refs), "rows": rows,
               "summary": {k: dict(v) for k, v in agg.items()}}
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("\n[산출물] → %s" % p)

    print("\n" + "=" * 96)
    for arm in ("A_repaired", "B_prerepair"):
        s = agg[arm]
        print("  %-12s  답 반환 %d/%d · component 합 %d · 관문1 생존 합 %s"
              % (arm, s["returned"], s["n"], s["comps"],
                 s["kept"] if arm == "A_repaired" else "(해당없음·답이 없다)"))
    print("\n판정: A 가 돌려주고 B 가 안 돌려주면 **갈린 원인은 계수기**(부정통제 성립).")
    print("      A 의 관문1 생존이 0 이면 채널만 열렸고 **아무것도 사지 않았다** — 그대로 적을 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
