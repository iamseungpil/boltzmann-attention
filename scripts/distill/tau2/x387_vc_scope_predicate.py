# -*- coding: utf-8 -*-
r"""x387 — **VC 발화 조건을 스키마로 정할 수 있는가**(무료·오프라인·LLM 0·GPU 0).

## 무엇을 묻나

VC(`VERDICT_CARRY`)는 **빼기 도구**인데 고를 것이 없는 자리에서도 발화해 무정보 판정을 낸다
(C535). 그리고 073 에서 그 발화가 pass 를 죽였다(같은 시드·노브 하나: ctl VC0 **1.0** ↔
vconly VC1 **0.0**).

x386/x386b 는 *"선택 태스크인가"* 를 **LLM 라벨**로 물었고 갈리지 않았다(표적有 CHOOSE 6/12).
다만 그 프로브는 질문이 얇았다(도구·후보를 안 줬다·컷 단위 채점) — *"모델이 못 한다"* 가 아니라
*"그렇게 물으면 안 갈린다"* 였다.

⇒ 여기서는 **묻지 않는다**. §1.5 Q1 대로 **decidable 한 술어**를 쓴다:

    발화 조건 = "그 결정점의 **후보 집합**을, 지금 형식화된 **표적 도구가 인자로 먹는가**"

  · 후보 집합 = A3 `doc_index[군]`(env 파일명 유래)
  · 표적 도구 = 라이브가 이미 매 턴 찍는 `[T2_ACTIONREQ] … formalized_target=…`
  · "먹는가" = 그 도구의 **인자 이름/값 공간**이 후보 집합과 닿는가(스키마 사실)

⚠엔진은 **답을 고르지 않는다** — 우리 레버를 켤지만 정한다(⛔0 이 금지하는 결정 이관 아님).

## 채점 (정답 = x377 과 같은 규칙 · gold 는 판정용 조회만·[[23]])

    표적有(그 군의 후보 이름이 gold 액션 인자에 실재) ↔ 술어가 **켬**
    표적無                                          ↔ 술어가 **끔**

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    일치율 ≥90%            → 이 술어로 게이트 설계(+배선 검정 +Δ 계측)
    80~90%                 → 어긋난 컷을 읽고([[08]]) 도구별 선언을 **최소로** 보탠다
    <80%                   → 스키마 술어 폐기 ⇒ A3 절차 선언으로

사용: python x387_vc_scope_predicate.py   (리모트·오프라인)
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGS = "/home/woori/scratch/logs"
TAGS = ["bank_t7313_treat_20260818h", "bank_t7312_treat_20260818g",
        "bank_t7310_treat_20260818e", "bank_t7314_treat_20260818j"]


def a2_load():
    out = {}
    for n in ("banking_knowledge.settings.json", "banking_knowledge.specific.json"):
        p = os.path.join(HERE, "a2", n)
        if os.path.exists(p):
            out.update(json.load(io.open(p, encoding="utf-8")))
    return out


def candidate_params(po):
    """**후보를 먹는 인자 이름**의 닫힌 집합 — A3 선언에서 기계적으로 모은다(저작 0).

    출처는 둘뿐이다:
      · `decide_candidates_text` 가 채우는 자리(= 후보 목록을 값으로 받는 인자)
      · `write_arg_enum`/`operand_keys` 계열이 후보 집합으로 검증하는 인자
    못 찾으면 **빈 집합**(그러면 술어는 언제나 '끔' → fail-safe 로 종전 거동과 갈린다).
    """
    keys = set()
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("candidate_params", "operand_keys", "class_params") and isinstance(v, list):
                    keys.update(str(x) for x in v)
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(po)
    return keys


def sims_of(tag):
    p = os.path.join(ROOT, tag, "results.json")
    if not os.path.exists(p):
        return {}
    doc = json.load(io.open(p, encoding="utf-8"))
    return {str(s.get("task_id")): s for s in (doc.get("simulations") or doc.get("results") or [])}


def verdict_cuts(tag):
    p = os.path.join(LOGS, "fb_%s.jsonl" % tag)
    out = []
    if not os.path.exists(p):
        return out
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("kind") != "verdict-lines":
            continue
        names = []
        for l in str(r.get("text") or "").splitlines():
            l = l.strip()
            if l.startswith("- ") and ":" in l:
                names.append(l[2:].split(":", 1)[0].strip())
        out.append({"tag": tag, "task": str(r.get("simtag", "")).split("#")[0],
                    "turn": r.get("turn"), "names": names})
    return out


def formalized_targets(tag, task, upto_turn):
    """그 컷 이전까지 라이브가 찍은 `formalized_target` 들(우리 계기 축자)."""
    p = os.path.join(LOGS, "%s.log" % tag)
    out = []
    if not os.path.exists(p):
        return out
    key = "sim=%s#" % task
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        if key not in ln or "formalized_target=" not in ln:
            continue
        m = re.search(r"formalized_target=([A-Za-z0-9_]+)", ln)
        if m:
            out.append(m.group(1))
    return out


def gold_blob(sim):
    buf = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or ck
        buf.append(str(a.get("name") or ""))
        for v in (a.get("arguments") or {}).values():
            buf.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
    return " || ".join(buf).lower()


def tool_params(name):
    """env 레지스트리에서 그 도구의 인자 이름 — 스키마 사실(하드코딩 0)."""
    import inspect
    from tau2.domains.banking_knowledge import tools as T
    for cls in ("KnowledgeTools", "KnowledgeUserTools"):
        c = getattr(T, cls, None)
        m = getattr(c, name, None) if c else None
        if callable(m):
            try:
                return [p for p in inspect.signature(m).parameters if p != "self"]
            except Exception:
                return []
    return []


def main():
    po = (a2_load().get("policy_ontology") or {})
    cparams = candidate_params(po)
    print("=" * 104)
    print("x387 · VC 발화 조건(스키마 술어) 분리력 · 후보-인자 이름 %d종: %s"
          % (len(cparams), ", ".join(sorted(cparams)) or "(없음)"))
    print("판정(사전 고정): 일치 ≥90%% → 게이트 설계 · 80~90%% → 어긋난 컷 읽고 최소 선언 보탬 · "
          "<80%% → 스키마 술어 폐기(A3 절차 선언으로)")
    print("=" * 104)

    rows, agg = [], collections.Counter()
    for tag in TAGS:
        sims = sims_of(tag)
        for c in verdict_cuts(tag):
            sim = sims.get(c["task"])
            if sim is None:
                continue
            gb = gold_blob(sim)
            hit = [n for n in c["names"] if n and n.split(" (")[0].lower() in gb]
            want = bool(hit)
            tg = formalized_targets(tag, c["task"], c["turn"])
            # 술어: 표적 도구들 중 **후보를 먹는 인자**를 가진 것이 하나라도 있는가
            fires, why = False, ""
            for t in dict.fromkeys(tg):
                ps = tool_params(t)
                inter = [p for p in ps if p in cparams]
                if inter:
                    fires, why = True, "%s(%s)" % (t, ",".join(inter))
                    break
            ok = int(fires == want)
            agg[("n",)] += 1
            agg[("ok",)] += ok
            agg[("want", want)] += 1
            agg[("fire", fires)] += 1
            rows.append({"task": c["task"], "tag": c["tag"].split("_")[1], "turn": c["turn"],
                         "want": want, "fires": fires, "why": why, "targets": list(dict.fromkeys(tg))[:3]})
            print("  %-9s %-6s t%-3s 표적=%-4s 술어=%-4s %s | %s"
                  % (c["task"], c["tag"].split("_")[1], c["turn"],
                     "있음" if want else "없음", "켬" if fires else "끔",
                     "✓" if ok else "✗", why or ("targets=" + ",".join(list(dict.fromkeys(tg))[:2]))))

    n, ok = agg[("n",)], agg[("ok",)]
    print("")
    print("## 집계  일치 %d/%d (%.0f%%) · 표적有 %d · 술어 켬 %d"
          % (ok, n, 100.0 * ok / max(1, n), agg[("want", True)], agg[("fire", True)]))
    r = ok / max(1, n)
    v = ("**게이트 설계 진행**" if r >= 0.9 else
         ("어긋난 컷을 읽고 도구별 선언 최소 보탬" if r >= 0.8 else
          "스키마 술어 폐기 ⇒ A3 절차 선언"))
    print("판정: %s" % v)
    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x387_vc_scope.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps({"rows": rows, "verdict": v,
                                                          "cparams": sorted(cparams)},
                                                         ensure_ascii=False, indent=1))
    print("원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
