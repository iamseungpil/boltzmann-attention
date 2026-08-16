# -*- coding: utf-8 -*-
r"""x343 — 격리 서브에 **요구를 넣으면 스스로 배제하는가**. 엔진 최소화의 게이트 실험.

설계서 = `ENGINE_MINIMIZATION_2026_08_16.md`(§1b 결정 규칙·§2 이 실험).

## 왜 (오늘 확정)

    메인 모델 · 요구 O · 문서 X (x342 A_REF)      → Green **24/24 오답**
    격리 서브 · 문서 O · 후보 O · **요구 X**(라이브) → savings Gold **8/8** · checking Blue **8/8 오답**
    격리 서브 · 문서 O · 후보 O · **요구 O**        → **미시험** ← 이 프로브

라이브는 서브의 `ask` 를 **후보 목록으로 치환**한다. 그 치환은 *"대화 잔여물이 해롭다"*(x269)
에서 왔는데 **잔여물과 요구를 함께** 버렸다. 문서상 gold 는 유일해다(요구 3개를 다 만족하는
클래스가 하나) 이고 모델의 오답은 **각각 한 조건에서만** 탈락한다 ⇒ 정보 부족이 아니라
**배제 실패**([[63]]).

## 셀 4 (서브 입력만 바꾼다 · 문서는 전 셀 동일)

    A_REF      문서 + 후보줄                      ← 라이브 재현(기대 오답)
    B_REQ      문서 + 후보줄 + **요구 메시지 축자**  ← H_iso 검정
    C_NOCAND   문서 + 요구(후보줄 없이)             ← 후보줄 기여 분리
    D_NEG      문서 + 후보줄 + **다른 태스크의 요구** ← ★부정통제(오르면 무효)

## 판정 (사전 고정 · 잡음 ±4 ⇒ 차 ≥5 만 인용)

    B ≥18 ∧ D ≤6   → **H_iso 참** ⇒ 엔진에 남길 것이 없다(전달·운반만)
    B ≈ A (둘 다 낮음) → H_iso 거짓 ⇒ 능력 경계 ⇒ 그때만 엔진(형식화는 LLM)
    C ≥ B          → 후보줄 불필요
    D ≥18          → 계기 무효

⛔**정규식 0**([[59]] 2026-08-16 강화판): 요구는 궤적 메시지를 **인덱스로 골라 통째로 복사**
  하고(추출 0), 문서는 정본 `t2_search.material_for`(A2 선언 키 조회), 후보는 `doc_index`
  키에 문자열 치환만. 이 파일에 `re` import 자체가 없다.

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x343_sub_requirement_iso.py [k] [nb]
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

TAG = "bank_t7304_treat_20260816j"
AUX = "bank_t7304_treataux_20260816j"
TASK, SEED = "task_055", "626729"
REQ_MSG = 87                     # 손님이 savings 요구를 진술한 메시지(포렌식 확정)
NEG_TASK, NEG_SEED, NEG_MSG = "task_024", "1567", 1   # 다른 태스크의 요구(부정통제)
GROUP = "savings_accounts"
NOW = "2025-11-14"          # 이 런의 시계(로그 축자) — 만료 제거의 기준
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")

MARKS = {"SILVERPLUS": "Silver Plus", "GREEN": "Green Account", "GOLD": "Gold Account"}
ASK = ("\n\nWhich ONE option should the agent recommend? Answer with the name only.")


def a2_load():
    """A2 병합(settings + specific) — 선언 읽기뿐."""
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


def msg_text(tag, task, seed, idx):
    """궤적 메시지를 **인덱스로 골라 통째로** 돌려준다 — 추출·패턴 0."""
    for s in F.sims(tag):
        if F.task_id(s) == task and str(s.get("seed")) == str(seed):
            ms = s.get("messages") or []
            if 0 <= idx < len(ms):
                return str(ms[idx].get("content") or "")
    return ""


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    a2 = a2_load()
    po = a2.get("policy_ontology") or {}
    if not po.get("doc_index"):
        print("A2 policy_ontology.doc_index 없음 — 중단")
        return 1

    corpus = {}
    for fn in sorted(os.listdir(DOCS)):
        d = json.load(io.open(os.path.join(DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    # ★`now` 필수(1차 실행에서 내가 빠뜨렸다·즉시 중단): 안 넘기면 만료 제거가 안 걸려
    #   `뺀 것 0` 이 되고, 그 조건은 x248 이 이미 **savings 0/8** 로 측정한 자리다. 그러면 B 가
    #   실패해도 *"요구가 무효"* 인지 *"만료 문서가 오염"* 인지 못 가른다([[55]] 계기 먼저).
    #   값은 이 런의 시계(로그 축자 `now=2025-11-14`)이고 도메인 판단이 아니다.
    material, info = TS.material_for(a2, GROUP, now=NOW, corpus=corpus)
    if not material:
        print("재료 생성 실패 — 중단(계기 결함)")
        return 1
    if not (info.get("dropped") or ()):
        print("⚠만료 제거 0건 — now=%s 가 안 먹었다. 중단(x248: 만료 미제거 시 savings 0/8)" % NOW)
        return 1

    cands = ", ".join(disp(x) for x in sorted((po.get("doc_index") or {}).get(GROUP) or ()))
    cand_line = str(po.get("decide_candidates_text") or
                    "The full official names on file are: {candidates}."
                    ).format(candidates=cands)
    req = msg_text(TAG, TASK, SEED, REQ_MSG)
    neg = msg_text(AUX, NEG_TASK, NEG_SEED, NEG_MSG)
    if not req or not neg:
        print("요구/부정통제 메시지를 못 찾음 — 중단(계기 결함) req=%d neg=%d"
              % (len(req), len(neg)))
        return 1

    print("x343rep · %s/%s(seed %s) · group=%s · 재료 %d자(문서 %d·뺀 것 %d)"
          % (TAG, TASK, SEED, GROUP, len(material), info.get("kept", 0),
             len(info.get("dropped") or ())))
    print("후보줄: %s" % cand_line[:150])
    print("요구(msg %d·축자 통째): %s" % (REQ_MSG, " ".join(req.split())[:260]))
    print("부정통제(%s msg %d): %s\n" % (NEG_TASK, NEG_MSG, " ".join(neg.split())[:160]))

    site = {"tag": TAG, "task": TASK, "cut": REQ_MSG, "sim": None,
            "base": "Policy documents on record (verbatim):\n" + material}

    P.run("x343rep", site, [
        ("A_REF", cand_line),
        ("B_REQ", "Customer's stated request:\n" + req + "\n\n" + cand_line),
        ("C_NOCAND", "Customer's stated request:\n" + req),
        ("D_NEG", "Customer's stated request:\n" + neg + "\n\n" + cand_line),
    ], MARKS,
        "B≥18 ∧ D≤6 → **H_iso 참** ⇒ 엔진에 남길 것 없음(전달·운반만) · B≈A(둘 다 낮음) → "
        "H_iso 거짓 ⇒ 능력 경계(그때만 엔진·형식화는 LLM) · C≥B → 후보줄 불필요 · D≥18 → 계기 무효",
        ASK, None, k, nb, det=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
