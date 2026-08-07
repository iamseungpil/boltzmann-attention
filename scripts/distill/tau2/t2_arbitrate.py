# -*- coding: utf-8 -*-
"""C3 중재 — **근거 등급이 높은 쪽이, 자기 선언 범위 안에서만, 명령한다. 진 쪽은 침묵이 아니라 치환된다.**

계약 정의 = `GENERAL_CONTRACTS_DESIGN_2026_08_06.md` §2-C3 ·
이론 = `CONFLICT_ARBITRATION_THEORY_2026_08_06.md`.

이 계약이 덮는 것은 충돌 census 12건 중 **T2(순서 지배)·T3(출처 오분류) 5건**이다. 나머지 셋은
다른 기구가 맡는다 — T1은 C4(단일 술어 함수), T4 슬롯은 아래 `merge`, T5는 아래 `may_suppress`.
**T6(창 부재)는 어느 계약도 덮지 않는다**(실패 198 sim 중 109). 그것을 여기에 밀어 넣지 말 것:
T6는 *무엇을 판정하는가*가 아니라 *언제 말하는가*이고, 직교 축이다.

세 규칙만 갖는다:
  grade(kind)      근거 등급 E1..E5
  merge(reqs)      같은 표적을 덮는 요건을 **한 문장**으로 (명령 하나 · 사실 합집합)
  may_suppress()   다른 레버를 침묵시킬 자격이 있는가
"""

from t2_dominance import requirements_for, merged_text, dominating_gate   # noqa: F401

__all__ = ["grade", "GRADES", "merge", "requirements_for", "may_suppress"]

# ★등급은 **출처의 종류**에 붙지 주장의 내용에 붙지 않는다. 숫자가 작을수록 강하다.
GRADES = {
    "execution_ledger": 1,   # E1 성공한 도구 호출의 결과
    "policy_verbatim": 2,    # E2 선언·정책 문서 축자
    "env_output": 3,         # E3 환경이 돌려준 것
    "retrieved_prose": 4,    # E4 회수한 산문
    "model_formalize": 5,    # E5 모델의 형식화 · 손님 주장
}


def grade(kind):
    """모르는 종류는 **가장 약하게** 둔다 — 모르는 것이 이기면 규칙이 규칙이 아니다."""
    return GRADES.get(str(kind or ""), 5)


def merge(reqs, a2, target):
    """같은 표적을 덮는 요건 전부를 한 문장으로.

    왜 하나를 고르지 않는가: 라이브에서 치환 24회가 **전부 같은 게이트**였고 뒤에 선 요건은
    **0회**였다. 순수 사전식 순서는 하위를 굶긴다(Tercan & Prabhu ECAI 2024). 그리고 굶은 요건은
    선언돼 있어도 존재하지 않는 것과 같다 — 101에서 원장 조회가 정확히 그랬다.
    ⇒ **명령은 하나, 사실은 합집합.** 이것이 T4(발화 슬롯 경합)도 함께 없앤다: 먼저 응답한 쪽이
    다른 쪽을 영영 막는 구조 자체가 사라진다.
    """
    return merged_text(a2, reqs, target) if reqs else ""


def may_suppress(lever_id, a2, targets_of_others=()):
    """이 레버가 다른 레버를 침묵시킬 자격이 있는가.

    T5(기제 억제)의 처방: 억제형 레버는 자기가 **무엇을 지우는지 선언**해야 하고, 지워지는 대상이
    다른 레버의 발화 조건이면 금지하거나 명시 면제한다. C13(반복 억제)이 050/051에서 **이행을
    만들던 반복 자체**를 지운 것이 이 규칙이 없어서였다.

    > **불변식 I2**: 억제 레버의 대상 집합 ∩ 다른 레버의 발화 조건 = ∅ (또는 명시 면제).

    자격 = A2 `suppression_warrants[lever_id]`에 **정책 축자 또는 사전등록 계량**이 선언돼 있을 것.
    못 대면 차단 자격 없음 → 표면화만. 미선언 도메인은 자격 없음으로 본다(안전측).
    """
    w = ((a2 or {}).get("suppression_warrants") or {}).get(str(lever_id or ""))
    if not w:
        return False, "자격 미선언 — 표면화만 허용(차단 금지)"
    overlap = sorted(set(w.get("erases") or ()) & set(targets_of_others or ()))
    if overlap:
        return False, "I2 위반: 지우는 대상이 다른 레버의 발화 조건 %s" % overlap
    return True, str(w.get("warrant") or "")
