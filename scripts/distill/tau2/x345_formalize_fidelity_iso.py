# -*- coding: utf-8 -*-
r"""x345 — **정규식 → formalize 이설의 전제**: 격리 서브가 레코드 값을 충실히 형식화하는가.

## 왜 (2026-08-17·[[59]] 강화판 이행)

살아 있는 C 계열 정규식 두 곳(`_ref_verify_deny` 1191 · `_param_cap_deny → parse_records` 1016)은
**도구 출력 산문에서 `field: value` 를 정규식으로 뽑는다** — 사용자 지시로 금지된 형태다.
두 곳의 *판정*(상점 대조·상한 비교)은 C129 가 이미 격리를 시험해 **LLM 재선택이 해롭다**고
닫았으므로(규칙 ②) 엔진에 남는다. 남는 물음은 하나다:

    **입력을 정규식 대신 격리 서브의 formalize 로 받아도 값이 충실한가?**

되면 이설은 *"판정은 그대로, 입력만 서브"* 로 끝난다. 안 되면 이설이 성적을 깎으므로
그 사실을 먼저 알고 다른 길을 찾아야 한다.

## 셀 3 (재료 = **라이브 도구 출력 축자** · 정답은 그 안에 있다)

    A_JSON   형식화 프롬프트만                 ← 그냥 되는가
    B_KEYS   + 뽑을 **키 목록**을 명시          ← A2 `row_keys` 형태(선언·판단 0)
    D_NEG    **없는 필드**를 함께 요구          ← ★부정통제: 날조하면 이설 불가

## 판정 (사전 고정 · 잡음 ±4 ⇒ 차 ≥5 만 인용)

    A 또는 B ≥18  ∧ D_NEG 날조 ≤6   → **이설 가능**(입력을 서브로·정규식 제거)
    A·B 둘 다 낮음                  → 이설 불가 ⇒ 정규식 유지 사유를 원장에 남기고 재설계
    D_NEG 날조 ≥18                  → 서브가 없는 값을 지어낸다 ⇒ 이설하면 **조용히 틀린다**

⛔정규식 0: 채점은 **부분문자열 존재**(`t2_probe._count` 의 `in`)로만 한다. 이 파일에 `re` 없음.
⚠`max_tokens=240` — 1차 실행이 기본 60 이라 **JSON 이 잘려** 채점값이 사라졌다(내 계기 결함).
⚠재료는 t7295 `task_050` 의 실제 도구 출력(레코드)을 **메시지 인덱스로 골라 통째로** 쓴다.

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x345_formalize_fidelity_iso.py [k] [nb]
"""
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

TAG, TASK, SEED = "bank_t7295_a_20260815n", "task_050", "1567"
REC_MSG = 17          # 도구 출력: credit_card_accounts 레코드(카드 등급·한도가 여기 있다)

# 채점 = 이 값들이 답에 **그대로** 있는가(전부 재료 안에 축자로 존재)
MARKS = {"LIMIT": "5000", "CARD": "Gold Rewards Card", "ACCT": "cc_584f9c5d00_gold"}
# 부정통제 지표 — ★1차 실행의 내 오류 교정: 지표를 **키 이름**(`annual_fee`)으로 잡았더니,
#   내가 그 키를 달라고 한 탓에 `"annual_fee": null`(정직한 답)까지 적중으로 세어졌다(24/24).
#   그건 날조 검사가 아니다. 고침 = 없는 필드 **하나만** 묻고 `null` 을 내는지 본다.
NEG_MARKS = {"NULLED": "null"}

BASE_ASK = ("\n\nReturn the record above as a JSON object. Copy every value VERBATIM from the "
            "text - do not compute, convert, or infer anything. Reply with JSON only.")
KEYS_ASK = ("\n\nReturn a JSON object with exactly these keys: account_id, card_type, "
            "credit_limit. Copy every value VERBATIM from the text above. Reply with JSON only.")
NEG_ASK = ("\n\nReturn a JSON object with exactly one key: annual_fee. Copy the value VERBATIM "
           "from the text above; if it is not present in the text, set it to null. "
           "Reply with JSON only.")


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next((s for s in F.sims(TAG)
                if F.task_id(s) == TASK and str(s.get("seed")) == SEED), None)
    if sim is None:
        print("대상 sim 없음 — 중단")
        return 1
    ms = sim.get("messages") or []
    rec = str(ms[REC_MSG].get("content") or "")
    if "credit_limit" not in rec:
        print("레코드 메시지가 아니다(msg %d) — 중단(계기 결함)" % REC_MSG)
        return 1
    for want in MARKS.values():
        if want not in rec:
            print("채점값 %r 가 재료에 없다 — 중단(계기 결함)" % want)
            return 1
    if "annual_fee" in rec:
        print("부정통제 필드가 재료에 실재한다 — 통제 무효·중단")
        return 1

    print("x345 · %s/%s(seed %s) · 재료 = msg %d 도구 출력 %d자" % (TAG, TASK, SEED, REC_MSG, len(rec)))
    print("재료 축자: %s\n" % " ".join(rec.split())[:240])

    site = {"tag": TAG, "task": TASK, "cut": REC_MSG, "sim": sim, "base": rec}
    P.run("x345", site, [("A_REF", ""), ("B_KEYS", "")], MARKS,
          "A 또는 B ≥18 ∧ D_NEG 날조 ≤6 → **이설 가능**(입력을 서브로) · 둘 다 낮음 → 이설 불가 · "
          "D_NEG 날조 ≥18 → 서브가 지어낸다 ⇒ 이설하면 조용히 틀린다",
          BASE_ASK, None, k, nb, maxtok=240, det=True)
    print("\n── B_KEYS(키 명시) ──")
    P.run("x345-keys", site, [("A_REF", "")], MARKS, "(위와 같은 판정)", KEYS_ASK, None, k, nb,
          maxtok=240)
    print("\n── D_NEG(없는 필드 요구·날조 검사) ──")
    P.run("x345-neg", site, [("A_REF", "")], NEG_MARKS,
          "NULLED 높음 → 정직(없는 값을 null 로) · 낮음 → **날조** = 이설 불가", NEG_ASK, None, k, nb,
          maxtok=240)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
