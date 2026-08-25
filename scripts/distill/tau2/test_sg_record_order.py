# -*- coding: utf-8 -*-
"""T2_SG_RECORD_ORDER 래칫 (2026-08-25) — 재배열이 **순열**인가, 그리고 조용히 죽지 않는가.

왜: 이 레버가 파는 것은 오직 순서여야 한다. 한 줄이라도 잃으면 그것은 순서 문제를 고치다
전사 결손을 **제조하는** 일이고, 그 손실은 서브 출력에서 *모델 탓*으로 보인다([[25]]).
그리고 오늘 아침 `T2_SG_PROMPT_V2` 가 격리 통과 후 라이브에서 죽은 전례가 있으므로([[24]])
조립부에 마커가 붙어 있는지도 여기서 지킨다.

이 검정이 지키는 것:
  ① 산출은 입력의 **순열** — Record ID 집합과 개수가 같다(내용 손실 0)
  ② 실제로 재배열한다 — 알려진 입력에서 순서가 바뀐다(무동작 아님)
  ③ 형식이 아니면 **그대로 돌려준다**(fail-open)
  ④ 술어에 **도메인 낱말이 없다** — 함수 본문에 도메인 어휘 0
  ⑤ 조립부에 마커가 있고 **OFF 에서도 관측 한 줄**을 남긴다(반증 경로)
  ⑥ 기본 OFF
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DUMP = """Transactions for account chk_x1
Found 5 record(s) in 'transactions':

1. Record ID: b_w3
   type: atm_withdrawal
   date: 2025-11-03
   amount: -100.00
2. Record ID: b_f1
   type: atm_fee
   date: 2025-11-01
   amount: -2.50
3. Record ID: b_w1
   type: atm_withdrawal
   date: 2025-11-01
   amount: -60.00
4. Record ID: b_p2
   type: purchase
   date: 2025-11-02
   amount: -9.99
5. Record ID: b_w2
   type: atm_withdrawal
   date: 2025-11-02
   amount: -40.00
"""


def ids(t):
    return re.findall(r"Record ID:\s*(\S+)", t)


def main():
    import t2_scaffold_get as S

    out = S._reorder_records(DUMP)

    # ① 순열
    assert sorted(ids(out)) == sorted(ids(DUMP)), (
        "재배열이 순열이 아니다 — 잃거나 만들었다: %r -> %r" % (ids(DUMP), ids(out)))
    assert len(ids(out)) == len(ids(DUMP)) == 5

    # ② 실제로 바뀐다 · 타입 묶음 + 묶음 안 날짜 오름차순
    assert ids(out) != ids(DUMP), "무동작이다 — 재배열이 일어나지 않았다"
    assert ids(out) == ["b_w1", "b_w2", "b_w3", "b_f1", "b_p2"], ids(out)

    # 각 블록의 본문이 축자로 살아 있다(번호만 다시 매긴다)
    for tok in ("-100.00", "-2.50", "-60.00", "-9.99", "-40.00"):
        assert tok in out, "블록 본문이 사라졌다: %r" % tok
    assert "Found 5 record(s)" in out, "머리말이 사라졌다"

    # ③ fail-open
    assert S._reorder_records("그냥 산문") == "그냥 산문"
    assert S._reorder_records("") == ""
    # 축(type)이 없는 블록이 하나라도 있으면 손대지 않는다
    broken = DUMP.replace("   type: purchase\n", "")
    assert S._reorder_records(broken) == broken, "축이 없는데 재배열했다"
    # ★덤프가 둘 이상이면 손대지 않는다 — 호출부가 여러 도구 출력을 이어 붙이므로,
    #   재배열하면 **서로 다른 계좌의 행이 한 묶음으로 섞인다**(순서를 고치려다 결손 제조).
    two = DUMP + "\n\n" + DUMP.replace("chk_x1", "chk_x2").replace("b_", "c_")
    assert S._reorder_records(two) == two, "두 원장을 이어 붙였는데 재배열했다 — 계좌가 섞인다"

    src = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
    i = src.index("def _reorder_records(")
    body = src[i:src.index("\ndef ", i + 1)]
    # 독스트링(근거·실측표가 도메인 낱말을 정당하게 담는다)과 주석을 뺀 **실행 코드만** 본다.
    q = body.index('"""')
    body_code = body[body.index('"""', q + 3) + 3:]
    code = "\n".join(l for l in body_code.splitlines() if not l.strip().startswith("#"))
    # ④ 도메인 낱말 0
    for bad in ("atm_", "withdrawal", "fee", "transaction", "dispute", "card"):
        assert bad not in code, "재배열 술어에 도메인 낱말이 들어왔다: %r" % bad

    # ⑤ 조립부 마커 · OFF 관측
    assert '[T2_SG_RECORD_ORDER] 관측(OFF)' in src, "OFF 관측 줄이 없다 — 死배선을 못 본다"
    assert '[T2_SG_RECORD_ORDER] %s: 덤프 재배열' in src, "적용 마커가 없다"

    # ⑥ 기본 OFF
    assert "T2_SG_RECORD_ORDER=0" in io.open(os.path.join(HERE, "go_stack.sh"),
                                             encoding="utf-8").read(), "기본 OFF 가 아니다"
    print("OK T2_SG_RECORD_ORDER: 순열 5/5 · 재배열 발생 · fail-open 3종 · 도메인 낱말 0 · "
          "마커 2종(적용/관측) · 기본 OFF")


if __name__ == "__main__":
    main()
