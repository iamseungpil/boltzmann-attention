# -*- coding: utf-8 -*-
r"""test_sg_dedup - `_dedup_by_id` 계약 (t7378 `task_074#s361454` 수리의 오프라인 래칫).

실물: 같은 계좌·같은 원장인데 전사 서브가 첫 호출 16행 -> 재호출 **19행**을 냈고 늘어난 3행이
`btxn_ar_lb_03f_err`·`05f_err`·`08f_err` 의 중복이었다. 비교기가 그대로 더해 총액 **30.00** 을
냈다(옳은 값 14.50) - 그리고 `[coverage] 19 of 19 (0 could not be verified)` 라 성공처럼 보였다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from t2_scaffold_get import _dedup_by_id                                  # noqa: E402

OK = 0
N = 0


def check(name, cond):
    global OK, N
    N += 1
    if cond:
        OK += 1
        print("  ok   %s" % name)
    else:
        print("  FAIL %s" % name)


def row(tid, fee, net="non_rho"):
    return {"transaction_id": tid, "fee_amount": fee, "network": net}


# 1) 실물 형태: 16 고유 + 3 완전중복 = 19 -> 16
base = [row("btxn_ar_lb_%02d" % i, 2.5) for i in range(1, 17)]
dups = [dict(base[2]), dict(base[4]), dict(base[7])]
keep, ndrop, conf = _dedup_by_id(base + dups, "transaction_id")
check("완전중복 3행 제거 (19 -> 16)", len(keep) == 16 and ndrop == 3)
check("충돌 없음", conf == [])
check("순서 보존 (첫 등장 유지)",
      [r["transaction_id"] for r in keep] == [r["transaction_id"] for r in base])

# 2) 같은 id 인데 내용이 다르면 지우지 않는다 - 고르는 일은 우리 몫이 아니다
a = row("btxn_x", 2.5)
b = row("btxn_x", 4.0)
keep, ndrop, conf = _dedup_by_id([a, b], "transaction_id")
check("내용충돌은 보존", len(keep) == 2 and ndrop == 0)
check("충돌 id 를 돌려준다", conf == ["btxn_x"])

# 3) 완전중복과 내용충돌이 섞여 있으면 각각 제 갈 길로
keep, ndrop, conf = _dedup_by_id([a, dict(a), b], "transaction_id")
check("혼합: 중복 1 제거 · 충돌 1 보존", ndrop == 1 and len(keep) == 2 and conf == ["btxn_x"])

# 4) id_field 가 없는 행은 손대지 않는다
noid = [{"fee_amount": 1.0}, {"fee_amount": 1.0}]
keep, ndrop, conf = _dedup_by_id(noid, "transaction_id")
check("id 없는 행은 통과", len(keep) == 2 and ndrop == 0 and conf == [])

# 5) 다른 id_field 이름으로도 동작한다 (도메인 낱말 0)
r1 = {"btxn": "a", "v": 1}
keep, ndrop, conf = _dedup_by_id([r1, dict(r1)], "btxn")
check("선언된 어떤 필드로도 동작", len(keep) == 1 and ndrop == 1)

# 6) 경계
check("빈 입력", _dedup_by_id([], "transaction_id") == ([], 0, []))
check("None 입력", _dedup_by_id(None, "transaction_id") == ([], 0, []))
keep, ndrop, conf = _dedup_by_id([row("only", 1.0)], "transaction_id")
check("단일 행 무변", len(keep) == 1 and ndrop == 0)

# 7) 값의 타입이 달라도 표기가 같으면 같은 행이다 (전사 서브가 1 과 1.0 을 섞어 낼 수 있다)
keep, ndrop, conf = _dedup_by_id([{"id": "z", "v": 1}, {"id": "z", "v": "1"}], "id")
check("str 표기 동일 -> 완전중복으로 본다", ndrop == 1 and conf == [])

print("")
print("test_sg_dedup: %d/%d" % (OK, N))
sys.exit(0 if OK == N else 1)
