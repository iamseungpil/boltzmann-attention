# -*- coding: utf-8 -*-
r"""x349 — **정보-맞춘** 요구 격리(C506 재검정). t7305 가 왜 격리를 못 옮겼는지 가른다.

## 왜 (t7305 판정 후·원장 C508)

라이브(t7305 treat)는 요구 주입 배선이 16/16 발화했는데 savings 는 **0/8** 그대로였고,
결정 서브의 답은 gold(`Silver Plus`)가 아니라 **`Green Account (savings)` 7/8** 이었다.
포렌식으로 확정한 사실(궤적 축자·`task_055#s554706`):

  · savings 결정 = **turn 4** · 그 시점 손님 발화는 **msg 1·3 뿐**
  · msg 3 은 **체킹(여행) 요구**이고 그 안에 벤치가 심은 미끼가 있다 —
    축자 *"And I'm kind of eco-conscious — any “green” options?"* · *"premium accounts"*
  · **savings 요구($5~7천 vs Gold 의 $10,000 최소)는 msg 16** = 결정 **뒤**에 나온다

⇒ x343(C506)이 B_REQ 로 준 것은 **결정 시점에 존재하지 않던 미래 정보**였다. 등대 §1.4
   *"정보-맞춘 격리 replay 만이 부하를 잰다"* 위반이다. 이 프로브가 그것을 되돌린다.

## 셀 (서브 입력만 바꾼다 · 문서·후보줄·now 는 전 셀 동일)

    A_REF       문서 + 후보줄                                  ← 라이브 재현 기준선
    B_LIVEQ     + **그 turn 에 실재한 발화로 뽑은 인용 전부**      ← ★정보-맞춤(라이브가 실제로 실은 것)
    C_FUTUREQ   + **msg 16 축자**(결정 뒤에 나온 savings 요구)     ← C506 재현(미래 정보)
    D_NEG       + 다른 태스크의 요구                              ← 부정통제
    E_MINUSECO  B_LIVEQ 에서 손님이 쓴 단어 `green` 이 든 인용을 뺀 것(그런 인용이 있을 때만)
                ← **진단 전용**(미끼 귀속). 레버 후보 아님 — 무엇을 뺄지는 사람이 정했다.

본문 구성은 **라이브 축자**(A2 `doc_decide_prompt`): 요청이 머리 · 재료가 몸 · 질문이 꼬리.

인용은 라이브와 **같은 절차**로 만든다: A2 `requirement_prompt` 로 LLM 이 내고, 각 인용의
**원문 존재만** `in` 으로 확인한다(엔진 추출 0·[[59]]·C45 동형). ⚠라이브는 인용 축자를
어디에도 안 남긴다(사이드카 0건·감사 구멍) — 이 프로브가 그 축자를 처음으로 인쇄한다.

## 판정 (사전 고정)

    B ≈ A (둘 다 오답)  ∧ C ≥18  → **C506 은 미래 정보로 잰 것**. 라이브 결손의 이름은
                                   *"요구 미전달"* 이 아니라 **결정이 요구보다 먼저**다(순서).
    B ≥18                        → 정보는 이미 충분했다 ⇒ 결손은 전달 형식(라이브 배선을 본다)
    D ≥18                        → 계기 무효
    E ≥ B+5 (checking)           → 미끼가 원인 ⇒ [[63]] 빼기 결손

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x349_infomatched_requirement_iso.py
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import t2_probe as P                                              # noqa: E402
import t2_search as TS                                            # noqa: E402

TAG = "bank_t7305_treat_20260817a"          # 요구 주입이 실제로 발화한 런
AUX = "bank_t7305_treataux_20260817a"
TASK, SEED = "task_055", "554706"
NEG_TASK, NEG_SEED, NEG_MSG = "task_024", "1567", 1
NOW = "2025-11-14"                          # 이 런의 시계(로그 축자)
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
ASK = "\n\nWhich ONE option should the agent recommend? Answer with the name only."

# 축별 (그룹, 결정 turn, **그 turn 에 실재한** 손님 발화 index, 미래 요구 msg, 지표)
#   ⚠1차 실행에서 checking 에 `[1, 3]` 을 줬다 = **컷 오류**(계기 결함·자가 검출).
#     라이브 로그 축자: checking 결정은 `turn=2`(인용 3개) — 그 시점 손님 발화는 **msg 1 뿐**이고
#     msg 3(여행·수수료·미끼)은 아직 안 나왔다. savings 는 `turn=4`(인용 6개) = msg 1·3.
AXES = [
    ("savings_accounts", 4, [1, 3], 16,
     {"SILVERPLUS": "Silver Plus", "GREEN": "Green Account", "GOLD": "Gold Account"}),
    ("checking_accounts", 2, [1], 16,
     {"PURPLE": "Purple", "EVERGREEN": "Evergreen", "BLUE": "Blue Account",
      "GREENFEE": "Green Fee-Free"}),
]


def a2_load():
    out = {}
    for name in ("banking_knowledge.settings.json", "banking_knowledge.specific.json"):
        p = os.path.join(A2DIR, name)
        if os.path.exists(p):
            out.update(json.load(io.open(p, encoding="utf-8")))
    return out


def disp(slug):
    """슬러그 → 표시명. 문자열 치환만(정규식 0)."""
    return " ".join(w.capitalize() if w[:1].islower() else w
                    for w in str(slug).replace("_", " ").split())


def msgs_of(tag, task, seed):
    for s in F.sims(tag):
        if F.task_id(s) == task and str(s.get("seed")) == str(seed):
            return s.get("messages") or []
    return []


def quotes_live(spec, user_text):
    """라이브와 **같은 절차**로 인용을 만든다 — A2 프롬프트로 LLM 이 내고 존재만 확인.

    엔진(`t2_search.sub_requirements`)은 에이전트 객체를 받지만 프로브에는 그 객체가 없다.
    프롬프트·JSON 경계 잡기·존재확인은 **같은 코드 규약**을 그대로 따른다(정규식 0)."""
    tpl = (spec or {}).get("requirement_prompt")
    if not tpl:
        return []
    raw = str((chat(tpl.format(messages=user_text[:8000]), None, 0.0, 500) or {}).get("content") or "")
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        rows = json.loads(raw[i:j + 1])
    except Exception:
        return []
    return [q for q in rows if isinstance(q, str) and q and q in user_text]


def block(quotes):
    """라이브 부착 형태 그대로(`t2_gate_patch` 2775-2777 축자 동형)."""
    return "Customer's stated request:\n" + "\n".join("- " + q for q in quotes)


def main():
    a2 = a2_load()
    po = a2.get("policy_ontology") or {}
    if not po.get("doc_index"):
        print("A2 doc_index 없음 — 중단(계기 결함)")
        return 1
    corpus = {}
    for fn in sorted(os.listdir(DOCS)):
        d = json.load(io.open(os.path.join(DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")

    ms = msgs_of(TAG, TASK, SEED)
    neg_ms = msgs_of(AUX, NEG_TASK, NEG_SEED)
    if not ms or not neg_ms:
        print("궤적을 못 찾음 — 중단(계기 결함)")
        return 1

    for group, turn, idxs, fut_idx, marks in AXES:
        material, info = TS.material_for(a2, group, now=NOW, corpus=corpus)
        if not material or not (info.get("dropped") or ()):
            print("[%s] 재료/만료제거 실패 — 중단(계기 결함)" % group)
            return 1
        cands = ", ".join(disp(x) for x in sorted((po.get("doc_index") or {}).get(group) or ()))
        cand_line = str(po.get("decide_candidates_text") or
                        "The full official names on file are: {candidates}."
                        ).format(candidates=cands)

        # ★라이브가 그 turn 에 실제로 갖고 있던 손님 발화 = 인덱스로 골라 통째로(추출 0)
        user_text = "\n\n".join(" ".join(str(ms[i].get("content") or "").split()) for i in idxs)
        fut = " ".join(str(ms[fut_idx].get("content") or "").split())
        neg = " ".join(str(neg_ms[NEG_MSG].get("content") or "").split())
        qs = quotes_live(po, user_text)
        if not qs:
            print("[%s] 인용 0개 — 중단(formalize 실패=계기 결함)" % group)
            return 1

        print("\n" + "#" * 100)
        print("## %s · 결정 turn=%d · 그 시점 발화 msg %s · 재료 %d자(문서 %d·뺀 것 %d)"
              % (group, turn, idxs, len(material), info.get("kept", 0),
                 len(info.get("dropped") or ())))
        print("## 라이브 절차로 뽑힌 인용 %d개 (라이브는 이걸 안 남긴다 — 여기가 첫 인쇄):" % len(qs))
        for q in qs:
            print("   - %s" % q[:180])
        print("## 미래 요구(msg %d·결정 뒤에 나온 말): %s" % (fut_idx, fut[:200]))

        # ★구성은 **라이브 축자**로(1차 실행의 두 번째 계기 결함): 라이브는 A2
        #   `doc_decide_prompt` = `{ask}` **머리** → `{material}` 몸 → 질문 꼬리 순인데,
        #   1차 프로브는 재료를 머리에 놓았다. 순서는 답을 바꾼다(C417⒠ *요청이 머리에*).
        tpl = str(po.get("doc_decide_prompt") or "")
        if "{ask}" not in tpl or "{material}" not in tpl:
            print("[%s] doc_decide_prompt 없음 — 중단(계기 결함)" % group)
            return 1
        live = lambda ask: tpl.format(ask=ask, material=material)      # noqa: E731

        arms = [
            ("A_REF", live(cand_line)),
            ("B_LIVEQ", live(block(qs) + "\n\n" + cand_line)),
            ("C_FUTUREQ", live(block([fut]) + "\n\n" + cand_line)),
            ("D_NEG", live(block([neg]) + "\n\n" + cand_line)),
        ]
        # 진단 전용: 손님이 쓴 단어 `green` 이 든 인용을 뺀다(사람이 정한 뺄셈·레버 아님)
        keep = [q for q in qs if "green" not in q.lower()]
        if keep and len(keep) < len(qs):
            arms.append(("E_MINUSECO", live(block(keep) + "\n\n" + cand_line)))

        # base·ask 를 비운다 — 본문 전체가 팔마다 라이브 템플릿으로 완성돼 있다.
        site = {"tag": TAG, "task": TASK, "cut": turn, "sim": SEED, "base": ""}
        P.run("x349-" + group, site, arms, marks,
              "B≈A(둘 다 오답) ∧ C≥18 → C506 은 **미래 정보** ⇒ 결손 이름은 '요구 미전달'이 아니라 "
              "**결정이 요구보다 먼저**(순서) · B≥18 → 정보는 충분했다(전달 형식 문제) · "
              "D≥18 → 계기 무효 · E≥B+5 → 미끼가 원인([[63]] 빼기)",
              "", None, 8, 3, det=True)   # ask 는 템플릿 안에 이미 있다
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
