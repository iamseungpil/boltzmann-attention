# -*- coding: utf-8 -*-
r"""x635 - **명단(roster) 배달이 검색 행동을 바꾸는가** (격리 · [[78]]).

## 왜 (사용자 지시 2026-08-30)
*"A2 A3 방식도 지금 alltools 의 shell 방식과 잘 조합될 수 있는 방식으로 재편성하라"* ·
*"모든 검색 후 검색 완결을 판단해서 더 할지 말지를 프롬프트로 리턴하는 거다"*.

재편성안(정본 §11): 엔진이 **본문을 배달하지 않고 경계(닫힌 명단)만** 댄다. 읽기는 모델의 shell.
그 명단이 실제로 **행동을 바꾸는지**를 배선 전에 격리로 잰다([[78]] - 격리 100% 일 때만 배선).

## 재료 = 회수된 라이브 문맥 (유료 런을 새로 사지 않는다)
`x617`(격리 base · Qwen3.8 · alltools · 우리 층 0)의 실패 궤적에서 **마지막 검색 직후**까지를 자른다.
표적은 `x634` 가 retrieval 우세로 가른 셋이고, 결손의 양 끝을 담는다:
    016  검색 15 · 이름으로 연 문서 0 · 실질 write 1
    033  검색  2 · 15 메시지에 종료          <- **과소**
    063  검색 57 · 문서 83 · 끝에 사람 이관   <- **과다**

## 팔 - 바뀌는 것은 **한 칸**뿐
    A_asis    문맥 그대로                                  <- 재현 게이트
    B_L0      + **정적 전역 명단**(11군 68계열 · 2,016 B) - 시작 지도
    C_L1      + **직전 shell 결과의 군** 명단만            - 국소 경계
    N_len     같은 길이의 무관 문장                        <- 길이 통제([[57]])

★L0/L1 을 가른 근거 (`x636` 실측 · 유료 0): 한 검색 결과가 걸치는 **군 수**가
    bm25 3.22 (군==1 **5%**) · dense 3.18 (**2%**) · shell(문서 열람분) **1.61** (**74%**)
  => bm25/dense 뒤에는 *"이 축"* 이라 말할 축이 **없다**. 경계 진술은 **shell 뒤에만** 뜻이 있다.
  => 033·016 은 shell 을 **0회** 썼으므로 L1 이 원리상 발화 못 하고 **L0 만** 닿는다.
     063 은 shell 41회이므로 L1 이 매번 경계를 상기시킨다.
명단은 **엔진 빌더가 만든다** - 이 파일은 프롬프트를 쓰지 않는다([[78]]).
군은 **이미 배달된 문서 id 가 속한 군 전부**다(닫힌 원소 검사 · argmax 0 · 해석 0).

## 채점 - 닫힌 술어 · gold 미접촉([[23]])
엔진은 무엇이 정답인지 모른다. 다음 수의 **행동**만 센다:
    named   지목한 계열 수            (063 에 **-** 가 성공 = 좁혀짐)
    new     아직 안 읽은 계열을 새로 지목 (033 에 **+** 가 성공 = 더 봄)
    srch    검색을 더 하겠다고 하는가
=> 033 과 063 은 **반대 방향**이 성공이다. 한 점수로 합치지 않는다([[70]]).

사용: PYTHONPATH=. python x635_roster_iso.py --port 8141 [--n 4] [--wiring-only]
"""
import argparse
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_search as SRCH                                        # noqa: E402
import t2_gate_patch as G                                       # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

RES = "/home/woori/iso_tau3/tau2-bench/data/simulations/bank_x617_iso_q38_bank20_20260830/results.json"
DOCS = "/home/woori/iso_tau3/tau2-bench/data/tau2/domains/banking_knowledge/documents"
TARGETS = ["task_033", "task_016", "task_063"]
KB = {"KB_search_bm25", "KB_search_dense", "shell"}
DOCPAT = re.compile(r"doc_[a-z0-9_()\-]{6,90}", re.I)
MODEL = "Qwen/Qwen3.8-27B-FP8"


def norm(s):
    s = str(s or "").lower().replace(".md", "").replace("(", "_").replace(")", "_")
    return re.sub(r"_+", "_", s).strip("_")


def owner_map(a2):
    """doc id -> (군, 계열). A3 선언만 읽는다 (698/698 · 해석 0)."""
    idx = (a2.get("policy_ontology") or {}).get("doc_index") or {}
    own = {}
    for g, subs in idx.items():
        if not isinstance(subs, dict):
            continue
        for s, lst in subs.items():
            for dd in (lst or []):
                own[norm(dd)] = (g, s)
    return idx, own


def cut(sim):
    """마지막 검색 호출을 담은 어시스턴트 턴까지 자른다 (그 직후가 결정점)."""
    msgs = sim.get("messages") or []
    last = 0
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []) or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            if n in KB:
                last = i
    return msgs[:last + 3]


def touched(msgs, own):
    """이미 배달된 문서가 속한 (군, 계열) - 닫힌 원소 검사."""
    out = set()
    for m in msgs:
        body = str(m.get("content") or "")
        for tc in (m.get("tool_calls") or []) or []:
            a = tc.get("arguments")
            body += " " + (a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
        for h in DOCPAT.findall(body):
            k = norm(h)
            if k in own:
                out.add(own[k])
    return out


def to_openai(msgs):
    out = []
    for m in msgs:
        role = str(m.get("role") or "")
        c = str(m.get("content") or "")
        if role == "system":
            out.append({"role": "system", "content": c})
        elif role == "user":
            out.append({"role": "user", "content": c})
        elif role == "assistant":
            d = {"role": "assistant", "content": c or None}
            tcs = []
            for tc in (m.get("tool_calls") or []) or []:
                a = tc.get("arguments")
                nm = tc.get("name") or (tc.get("function") or {}).get("name") or "unknown"
                tcs.append({"id": tc.get("id") or "x", "type": "function",
                            "function": {"name": nm,
                                         "arguments": a if isinstance(a, str)
                                         else json.dumps(a or {}, ensure_ascii=False)}})
            if tcs:
                d["tool_calls"] = tcs
            out.append(d)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id") or "x", "content": c[:6000]})
    return out


def static_roster(idx):
    """L0 - 도메인 전체 명단(계열명 + 문서 수 + 닫힘 진술). 런타임 상태 0 · 2 KB."""
    out = []
    for g in sorted(idx):
        subs = idx.get(g)
        if not isinstance(subs, dict):
            continue
        keys = [k for k in sorted(subs) if k != "_general_"]
        body = ", ".join("%s (%d)" % (G._slug_disp(k), len(subs[k] or [])) for k in keys)
        gen = len(subs.get("_general_") or [])
        line = "- %s: %s" % (G._slug_disp(g), body or "(no sub-sets)")
        if gen:
            line += " + General (%d)" % gen
        out.append(line)
    hdr = ("The knowledge base is organised into the following document sets. "
           "This list is complete: every document belongs to exactly one set below.")
    nl = chr(10)
    return hdr + nl + nl + nl.join(out)


def all_search_groups(msgs, own):
    """L1' - **모든 검색 결과**(bm25/dense/shell)가 실린 군·계열 누적. 도구 무관 = 구멍 0.

    ★왜 이 형태인가 (사용자 지시 2026-08-30): *"수동적으로 shell 이 불릴 때만 하면 **빠져 나갈
      구멍**이 생긴다. 100% 를 원한다."* — 033·016 은 shell 을 0회 썼으므로 shell-전용 발화는
      그 둘을 통째로 비켜 간다. 어느 도구든 결과에 doc id 가 실리므로 이 경로는 **항상** 발화한다.
    ⚠대가(미리 밝힘·[[70]]): `x636` 실측으로 bm25/dense 결과는 평균 **3.22 군**에 걸친다
      (군==1 이 5%/2%). 그래서 목록이 넓어진다 — 좁혀 주는 게 아니라 *경계가 있다* 를 알린다.
    ⚠`x637` 실측: KB_search 는 **61% 가 원문 전량**(길이비 중앙 1.00)이다. 따라서 *"발췌만 봤다"* 는
      술어는 성립하지 않는다 — 결손은 깊이가 아니라 **집합 커버리지**다.
    """
    out = set()
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []) or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            if n not in KB:
                continue
            resp = ""
            for mj in msgs[i + 1:]:
                if mj.get("role") == "tool" and mj.get("id") == tc.get("id"):
                    resp = str(mj.get("content") or "")
                    break
            hits = {own[norm(h)] for h in DOCPAT.findall(resp) if norm(h) in own}
            if hits and len(hits) < 60:            # INDEX.md 전량 덤프 배제
                out |= hits
    return out


def last_shell_groups(msgs, own):
    """L1(좁은 판) - **직전 shell 결과**가 실린 군 (INDEX.md 덤프는 제외). 닫힌 원소 검사.

    ⚠이 판은 **구멍이 있다**(shell 을 안 쓰면 발화 0). 비교용으로만 남긴다 - 정본은 `all_search_groups`.
    """
    best = None
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []) or []:
            n = tc.get("name") or (tc.get("function") or {}).get("name")
            if n != "shell":
                continue
            resp = ""
            for mj in msgs[i + 1:]:
                if mj.get("role") == "tool" and mj.get("id") == tc.get("id"):
                    resp = str(mj.get("content") or "")
                    break
            hits = {own[norm(h)] for h in DOCPAT.findall(resp) if norm(h) in own}
            if hits and len(hits) < 60:            # INDEX.md 전량 덤프 배제
                best = hits
    return best or set()


def build_cases(a2, idx, own, corpus, sims):
    cases = []
    for t in TARGETS:
        s = sims.get(t)
        if not s:
            print("SKIP %s (실패 sim 없음)" % t)
            continue
        pre = cut(s)
        tch = touched(pre, own)
        seen = {s2 for _, s2 in tch}
        l0 = static_roster(idx)

        def render(hits):
            gs = sorted({g for g, _ in hits})
            parts = []
            for g in gs:
                disp = {k: G._slug_disp(k) for k in (idx.get(g) or {})}
                txt, _d = SRCH.roster_for(a2, g, corpus=corpus, seen=seen,
                                          disp=disp, with_ids=False)
                if txt:
                    parts.append(txt)
            return gs, (chr(10) + chr(10)).join(parts)

        g_all, l1all = render(all_search_groups(pre, own))     # 정본 - 구멍 0
        g_sh, l1sh = render(last_shell_groups(pre, own))       # 좁은 판 - 비교용
        cases.append(dict(task=t, msgs=pre, seen=sorted(seen), l0=l0,
                          l1all=l1all, l1sh=l1sh, g_all=g_all, g_sh=g_sh))
        print("[%s] 문맥 %d · 만진 계열 %d · L0 %d B · L1all 군 %d/%d B · L1shell 군 %s/%d B"
              % (t, len(pre), len(seen), len(l0), len(g_all), len(l1all),
                 g_sh or "없음", len(l1sh)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args()

    a2 = load_domain_a2("banking_knowledge")
    idx, own = owner_map(a2)
    d = json.load(io.open(RES, encoding="utf-8"))
    sims = {str(s.get("task_id")): s for s in (d.get("simulations") or [])
            if (s.get("reward_info") or {}).get("reward") == 0.0}

    corpus = {}
    for f in os.listdir(DOCS):
        if not f.endswith(".json"):
            continue
        j = json.load(io.open(os.path.join(DOCS, f), encoding="utf-8"))
        corpus[j.get("id")] = j.get("content") or ""
    print("코퍼스 %d 문서 · A3 색인 소유 %d 문서" % (len(corpus), len(own)))

    cases = build_cases(a2, idx, own, corpus, sims)
    if a.wiring_only:
        for c in cases:
            print("\n===== %s L0 전역명단 (%d B) =====\n%s"
                  % (c["task"], len(c["l0"]), c["l0"][:900]))
            print("\n----- %s L1 직전 shell 군 %s (%d B) -----\n%s"
                  % (c["task"], c["l1_groups"], len(c["l1"]), c["l1"][:900] or "(없음)"))
        return

    import urllib.request
    url = "http://localhost:%d/v1/chat/completions" % a.port

    def ask(msgs):
        body = json.dumps({"model": MODEL, "messages": msgs,
                           "temperature": a.temp, "max_tokens": 700}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read().decode())
        return j["choices"][0]["message"].get("content") or ""

    disp_all = {}
    for g, subs in idx.items():
        if isinstance(subs, dict):
            for k in subs:
                if k != "_general_":
                    disp_all[G._slug_disp(k).strip().lower()] = k

    print("\n%-9s %-9s %-4s %-8s %-8s %s" % ("task", "arm", "n", "계열지목", "새계열", "검색계속"))
    print("-" * 58)
    for c in cases:
        base = to_openai(c["msgs"])
        filler = "Please continue helping the customer. " * max(1, len(c["l0"]) // 38)
        arms = [("A_asis", None), ("B_L0", c["l0"]), ("N_len", filler)]
        if c["l1all"]:
            arms.insert(2, ("C_L1all", c["l1all"]))       # 정본 - 모든 검색 뒤 (구멍 0)
        if c["l1sh"]:
            arms.insert(3, ("D_L1shell", c["l1sh"]))      # 좁은 판 - 비교용
        for arm, extra in arms:
            named_t, new_t, srch_t = [], [], []
            for _ in range(a.n):
                msgs = list(base) + ([{"role": "user", "content": extra}] if extra else [])
                try:
                    out = ask(msgs)
                except Exception as e:
                    print("  %s %s ERR %s" % (c["task"], arm, e))
                    continue
                low = out.lower()
                named = {k for dn, k in disp_all.items() if dn and dn in low}
                named_t.append(len(named))
                new_t.append(len(named - set(c["seen"])))
                srch_t.append(1 if any(w in low for w in
                                       ("kb_search", "grep", "cat ", "ls ", "search")) else 0)
            if named_t:
                print("%-9s %-9s %-4d %-8.1f %-8.1f %.0f%%"
                      % (c["task"], arm, len(named_t),
                         sum(named_t) / len(named_t), sum(new_t) / len(new_t),
                         100.0 * sum(srch_t) / len(srch_t)))


main()
