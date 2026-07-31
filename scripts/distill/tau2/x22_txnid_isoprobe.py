# -*- coding: utf-8 -*-
"""X22 — `transaction_id` wrong-pick 정보-맞춘 격리 프로브 (2026-07-31).

설계 정본 = `TXNID_ISOLATION_PROBE_DESIGN_2026_07_31.md`. 근거 = C256(64건·전체 실패의 32%).

**[[18]] 의무**: F3/⋈ 경계로 분류하려는 모든 사례는 먼저 A_minimal vs B_fullctx 격리 프로브를
돌린다. 궤적이 말할 수 있는 것은 "날조 아님·gold를 봤음"까지이고, **부하인지 경계인지는 이 프로브
없이 판정 금지**다(C124 전례: 같은 형태가 전사-슬립+자기-정박으로 판명).

  A_minimal  = 그 결정에 필요한 정보만 (사용자 요구 + 후보 txn 레코드 원문)
  B_fullctx  = 그 시점의 실제 궤적 문맥 전량
  A정답·B오답 → **부하**(scaffold로 닫힌다)   /   A오답·B오답 → **경계 후보**

★오염 방지(설계 §3) — 이 파일이 지키는 것:
  1. 후보 집합은 **모델이 실제로 본 도구 출력**에서만 만든다. gold를 보고 고르지 않는다([[03b]]).
  2. 프롬프트에 **gold id를 넣지 않는다**. 후보 순서는 출력 그대로(정렬·필터 금지).
  3. A와 B의 **후보 집합이 동일**해야 한다(A에서만 줄이면 난이도가 달라진다).
  4. 라이브와 **동일 모델·동일 온도**.
★부수 arm(설계 §4): H1 전사-슬립(후보 한 줄씩 분리) · H2 다건(하나가 아니라 **전부** 고르라).

용법:
  py -3 x22_txnid_isoprobe.py --cases <results.json> --out cases.jsonl     # ①사례 추출(무료)
  py -3 x22_txnid_isoprobe.py --run cases.jsonl --base http://…/v1 --out r.jsonl   # ②프로브(GPU)
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x12_action_fail_exact as X12  # noqa: E402

TXN = re.compile(r"txn_[0-9a-f]+")
ARMS = ("A_minimal", "B_fullctx", "A_split", "A_all")


def extract(results_path):
    """실패 사례 → 프로브 입력. **gold는 채점용으로만 보관하고 프롬프트에 넣지 않는다.**"""
    with open(results_path, encoding="utf-8") as f:
        d = json.load(f)
    cases = []
    drop_not_yet_seen = 0
    for s in (d.get("simulations") or []):
        msgs = s.get("messages") or []
        plist = X12.preds(msgs)
        # ★결정 시점 절단 (2026-07-31 수정) — 초판은 궤적 *전체*의 도구 출력을 B_fullctx로 줬다.
        #   그건 두 가지로 틀렸다: ①에이전트가 결정 시 **갖지 않았던 이후 정보**까지 주므로
        #   [[18]]의 "정보-맞춘 격리"가 깨진다(부하가 과소·경계가 과대 추정된다) ②9/16 사례가
        #   문맥 한계(44,672 토큰)를 넘어 프로브가 그냥 에러로 죽는다.
        #   ⇒ 문맥·후보 모두 **모델이 처음 txn을 고른 메시지 이전**까지만 쓴다.
        dec = len(msgs)
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            if any("transaction_id" in json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
                   for tc in (m.get("tool_calls") or [])):
                dec = i
                break
        # 모델이 실제로 본 도구 출력만 (role=tool) — 순서 보존·결정 시점까지
        tool_out = [str(m.get("content") or "")
                    for m in msgs[:dec] if m.get("role") == "tool"]
        user_txt = next((str(m.get("content") or "") for m in msgs
                         if m.get("role") == "user"), "")
        cand, seen = [], set()
        for c in tool_out:
            for t in TXN.findall(c):
                if t not in seen:
                    seen.add(t)
                    cand.append(t)
        for chk in ((s.get("reward_info") or {}).get("action_checks") or []):
            if chk.get("action_match"):
                continue
            a = chk.get("action") or {}
            g = {"name": a.get("name"), "arguments": a.get("arguments") or {},
                 "compare_args": a.get("compare_args")}
            cls, det = X12.classify(g, plist)
            if cls != "NESTED_VALUE":
                continue
            if not any(k == "transaction_id" for k, _ in (det.get("inner") or [])):
                continue
            gj = X12._pj(g["arguments"].get("arguments")) or {}
            gid = gj.get("transaction_id")
            if not gid:
                continue
            if gid not in cand:
                # 결정 시점까지의 출력에 gold가 없다 = 그 순간 **고를 수 없었다**.
                # 이건 wrong-pick(⋈)이 아니라 **발굴/reach 실패**다 — 다른 축이므로 제외하고 센다.
                drop_not_yet_seen += 1
                continue
            cases.append({
                "task": s.get("task_id"), "trial": s.get("trial"),
                "gold": gid,                      # ★채점 전용 — 프롬프트 금지
                "candidates": cand,               # 출력 순서 그대로
                "user_text": user_txt[:1500],
                "tool_outputs": tool_out,         # B_fullctx용(결정 시점까지)
            })
    if drop_not_yet_seen:
        print("  ⚠결정 시점에 gold가 아직 안 보인 사례 %d건 제외 = wrong-pick 아니라 발굴 실패 축"
              % drop_not_yet_seen)
    return cases


def prompt_for(case, arm):
    """★gold 미포함. A와 B의 후보 집합은 동일하다."""
    cands = case["candidates"]
    ask = ("Which transaction_id should be used? Answer with exactly one id and nothing else."
           if arm != "A_all" else
           "Which transaction_ids should be used? Answer with a JSON list of ids and nothing else.")
    if arm == "B_fullctx":
        ctx = "\n".join(case["tool_outputs"])
    elif arm == "A_split":
        ctx = "\n".join("- %s" % c for c in cands)          # H1: 한 줄씩 분리
    else:
        ctx = ", ".join(cands)
    return ("Customer request:\n%s\n\nTransaction records available to you:\n%s\n\n%s"
            % (case["user_text"], ctx, ask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.environ.get("T2_PROBE_BASE", ""))
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--arms", default="A_minimal,B_fullctx")
    ap.add_argument("--limit", type=int, default=0)
    # B_fullctx가 서빙 문맥(44,672 토큰)을 넘으면 프로브가 에러로 죽어 쌍이 안 생긴다.
    # 넘는 사례는 **제외하고 수를 보고**한다(조용히 자르면 B의 정보량이 달라져 판정이 오염된다).
    ap.add_argument("--maxctx", type=int, default=0, help="B_fullctx 최대 문자수(0=무제한)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.cases:
        cs = extract(args.cases)
        if args.maxctx:
            n0 = len(cs)
            cs = [c for c in cs if sum(len(t) for t in c["tool_outputs"]) <= args.maxctx]
            print("  ⚠B_fullctx가 문맥 한계를 넘어 제외: %d/%d" % (n0 - len(cs), n0))
        if args.limit:
            # ★설계 §7: 한 태스크 특성으로 기울지 않도록 **태스크별 라운드로빈**으로 뽑는다
            by = defaultdict(list)
            for c in cs:
                by[c["task"]].append(c)
            picked, i = [], 0
            while len(picked) < args.limit and any(by.values()):
                for t in sorted(by):
                    if by[t] and len(picked) < args.limit:
                        picked.append(by[t].pop(0))
                i += 1
                if i > 1000:
                    break
            cs = picked
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            for c in cs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print("사례 %d건 → %s" % (len(cs), args.out))
        print("  태스크 분포:", Counter(c["task"] for c in cs).most_common())
        print("  후보 수 중앙값:", sorted(len(c["candidates"]) for c in cs)[len(cs) // 2] if cs else 0)
        return

    if not args.run:
        ap.error("--cases 또는 --run 중 하나가 필요하다")
    if not args.base:
        ap.error("--base(vLLM endpoint)가 필요하다 — 이 단계는 GPU를 쓴다")

    import litellm
    cases = [json.loads(l) for l in open(args.run, encoding="utf-8")]
    arms = [a for a in args.arms.split(",") if a in ARMS]
    rows = []
    for c in cases:
        for arm in arms:
            p = prompt_for(c, arm)
            # ★오염 가드 (2026-07-31 교정) — 초판은 `gold not in p`를 걸었는데 **설계와 모순**이라
            #   첫 사례에서 즉사했다(그래서 지난 세션에 프로브가 못 돌았다): 정보-맞춤이 성립하려면
            #   gold는 후보 안에 *있어야* 한다. 금지할 것은 gold의 *존재*가 아니라 **정답 표시**다.
            #   ⇒ ①gold ∈ 후보 ②프롬프트가 `gold` 필드를 읽지 않았음(그림자 case로 동일성 확인)
            #   ③후보 순서 = 추출 순서(정렬·필터 금지) ④A와 B의 후보 집합 동일.
            assert c["gold"] in c["candidates"], "★gold가 후보에 없다 — 정보-맞춤 위반"
            shadow = {k: v for k, v in c.items() if k != "gold"}
            assert prompt_for(shadow, arm) == p, "★프롬프트가 gold 필드를 참조했다 — 중단"
            if arm in ("A_minimal", "A_all"):
                seq = TXN.findall(p)
                assert [x for x in seq if x in set(c["candidates"])][:len(c["candidates"])] \
                    == c["candidates"], "★후보 순서가 바뀌었다(정렬 금지) — 중단"
            try:
                r = litellm.completion(model="openai/" + args.model, api_base=args.base,
                                       api_key="x", temperature=0.0,
                                       messages=[{"role": "user", "content": p}])
                out = (r.choices[0].message.content or "").strip()
            except Exception as e:
                out = "ERROR: %r" % (e,)
            got = TXN.findall(out)
            ok = (c["gold"] in got) if arm == "A_all" else (bool(got) and got[0] == c["gold"])
            rows.append({"task": c["task"], "arm": arm, "gold": c["gold"],
                         "answer": out[:200], "correct": bool(ok),
                         "n_cand": len(c["candidates"])})
            print("  %-10s %-10s %s" % (c["task"], arm, "OK" if ok else "X"))
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== 판정 ([[18]])")
    by = defaultdict(dict)
    for r in rows:
        by[(r["task"], r["gold"])][r["arm"]] = r["correct"]
    v = Counter()
    for k, d_ in by.items():
        a, b = d_.get("A_minimal"), d_.get("B_fullctx")
        if a is None or b is None:
            continue
        v[("부하(A정답·B오답)" if (a and not b) else
           "경계 후보(둘 다 오답)" if (not a and not b) else
           "재현 실패(둘 다 정답)" if (a and b) else "역전(A오답·B정답)")] += 1
    for k, n in v.most_common():
        print("   %-24s %d" % (k, n))
    print("\n⚠판정은 **쌍이 완성된 사례만**. 재현 실패가 많으면 프로브 설계 결함이다(설계 §2).")


if __name__ == "__main__":
    main()
