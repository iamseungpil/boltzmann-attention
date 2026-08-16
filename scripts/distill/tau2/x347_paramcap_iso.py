# -*- coding: utf-8 -*-
r"""x347 — `param_cap_check`(CLI 증액 상한)를 **격리로 대체할 수 있는가**. 규칙 ① 검정.

## 왜 (2026-08-17·사용자 결정 규칙)

    ①격리로 되면 엔진에 남기지 않는다 → ②안 되는 것만 남기되 형식화는 LLM → ③패턴매칭 금지

`_param_cap_deny`(t2_gate_patch:1016)는 레코드 텍스트를 `t2_resolve.parse_records` **정규식**으로
뜯어 상한(등급별 비율 × 한도)을 비교한다 — 금지된 형태다. 그런데 **격리는 한 번도 안 재봤다**.
이 자리는 x288/x291(흩어진 사실 산술·격리 0/8)과 난이도가 다르다: **한 레코드 안에 두 값이
다 있는 단일 곱셈**이다.

## 셀 3 (재료 = 라이브 도구 출력 축자 · t7295 task_050 msg 17)

    A_REF     레코드만                     ← 정책을 알고 있는가(기대: 모름)
    B_POLICY  + 카드 정책 문서(정본 경로)    ← 문서를 주면 상한을 내는가
    D_NEG     레코드만 + **다른 등급**을 물음 ← ★부정통제: 아무 숫자나 내면 무효

## 판정 (사전 고정 · det 모드 = 온도 0 ×2 동일이면 n=1)

    B ≥ 정답 ∧ D_NEG 오답  → **격리로 된다** ⇒ `param_cap_check` **삭제**(정규식 동시 소멸)
    A·B 둘 다 오답         → 격리 불가 ⇒ 검증기 유지 + 입력만 formalize 이설(규칙 ②)
    D_NEG 도 정답          → 숫자를 문맥 없이 낸다 = 통제 무효

정답: Gold Rewards Card 는 한도의 **50%** ⇒ $5,000 × 0.5 = **$2,500**.
부정통제: 같은 레코드로 **Bronze**(25%)를 물으면 $1,250 이 나와야 하고, 2500 을 그대로 내면 무효.

⛔정규식 0 — 채점은 부분문자열 존재만. 재료는 메시지 인덱스로 골라 통째로 쓴다.

실행:  T2_PROBE_URL=http://localhost:8140/v1/chat/completions \
       /home/woori/venvs/seka_env/bin/python x347_paramcap_iso.py
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

import t2_forensic as F                                            # noqa: E402
import t2_probe as P                                               # noqa: E402
import t2_search as TS                                             # noqa: E402

TAG, TASK, SEED = "bank_t7295_a_20260815n", "task_050", "1567"
REC_MSG = 17
NOW = "2025-11-14"
A2DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

# ★1차 실행의 내 지표 결함: 오답 `$32,500` 이 `"2,500"` 을 **부분문자열로 포함**해 적중으로
#   세어졌다. 부분문자열 채점에는 자릿수 경계가 없다. 고침 = **충돌 탐지 마크를 함께 싣고**
#   판정을 "정답 적중 ∧ 충돌 미적중" 으로 읽는다(정규식 0·[[59]]).
MARKS = {"C2500": "2500", "C2_500": "2,500",          # 정답 = 2,500
         "X32500": "32500", "X32_500": "32,500",      # ← 이게 켜지면 위 적중은 무효
         "LIM5000": "5000"}                           # 한도 복창 탐지
NEG_MARKS = {"B1250": "1250", "B1_250": "1,250",      # 부정통제 정답 = 1,250(25%)
             "C2500": "2500", "C2_500": "2,500",      # 이게 켜지면 통제 실패
             "X32500": "32500", "X32_500": "32,500"}
ASK = ("\n\nWhat is the maximum increase amount the agent may submit for THIS card, "
       "per the tier cap? Answer with digits only - no currency symbol, no words.")
NEG_ASK = ("\n\nSuppose this same customer instead held the lowest tier card. What is the "
           "maximum increase amount then? Answer with digits only - no currency symbol, "
           "no words.")


def a2_load():
    out = {}
    for name in ("banking_knowledge.settings.json", "banking_knowledge.specific.json"):
        p = os.path.join(A2DIR, name)
        if os.path.exists(p):
            out.update(json.load(io.open(p, encoding="utf-8")))
    return out


def main():
    sim = next((s for s in F.sims(TAG)
                if F.task_id(s) == TASK and str(s.get("seed")) == SEED), None)
    if sim is None:
        print("대상 sim 없음 — 중단")
        return 1
    rec = str((sim.get("messages") or [])[REC_MSG].get("content") or "")
    if "credit_limit" not in rec or "Gold Rewards Card" not in rec:
        print("레코드 메시지가 아니다 — 중단(계기 결함)")
        return 1
    if "2500" in rec or "2,500" in rec:
        print("정답이 재료에 이미 있다 — 통제 무효·중단")
        return 1

    corpus = {}
    for fn in sorted(os.listdir(DOCS)):
        d = json.load(io.open(os.path.join(DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    material, info = TS.material_for(a2_load(), "credit_cards", now=NOW, corpus=corpus)
    if not material:
        print("정책 재료 생성 실패 — 중단")
        return 1

    print("x347 · %s/%s(seed %s) · 레코드 %d자 · 정책 재료 %d자(문서 %d)"
          % (TAG, TASK, SEED, len(rec), len(material), info.get("kept", 0)))
    print("레코드 축자: %s\n" % " ".join(rec.split())[:200])

    site = {"tag": TAG, "task": TASK, "cut": REC_MSG, "sim": sim, "base": rec}
    P.run("x347", site,
          [("A_REF", ""), ("B_POLICY", "Policy documents on record (verbatim):\n" + material)],
          MARKS,
          "정답 = (C2500|C2_500) 적중 ∧ (X32500|X32_500) 미적중. 그 정답이 A 또는 B 에서 나오고 "
          "D_NEG 가 1250 을 내면 → **격리로 된다** ⇒ param_cap_check 삭제 · 정답이 어디서도 "
          "안 나오면 → 검증기 유지 + 입력만 formalize 이설 · D_NEG 가 2500 이면 통제 무효",
          ASK, None, 8, 3, det=True)
    print("\n── D_NEG(최저 등급을 물음·2500 이 나오면 통제 무효) ──")
    P.run("x347-neg", site,
          [("A_REF", "Policy documents on record (verbatim):\n" + material)],
          NEG_MARKS, "(위 판정의 통제)", NEG_ASK, None, 8, 3, det=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
