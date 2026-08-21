# -*- coding: utf-8 -*-
r"""x461 — 선행-read 레버 **생존 감사** (2026-08-21·무료·오프라인·LLM 0·핸드오프 §1-4)

## 왜 ([[55]] 순서: 모델 탓 전에 우리 배관)
1단계 census(x460)가 **read gold 미수행 22건**을 확정했다(074×10·079×4·094×4·085×3·073×1 —
`get_bank_account_transactions` — 그리고 050×2 `get_payment_history_6183`). 사용자 지시:
*"새 레버 만들지 말 것 — 선행-read 레버는 이미 있으니 왜 안 먹는지 `t2_liveness` 로 먼저 감사한다."*

## 무엇을 재나 (닫힌 술어만 — 태그 grep + 선언 읽기·도메인 판단 0)
    ① 영속 로그(t7328 40 sim)에서 태스크별 [T2_PIN_READ]/[T2_PIN_READ_STEPS]/[T2_DEMANDED_STEP]
      줄 수 — ABSENT/발화 분포([[67]] `t2_liveness` 의 세 상태와 같은 축)
    ② A3 선언에서 pin 이 보는 선행-read 위상(`t2_precedence.declarations`) — read 별 피의존 수
⛔gold·reward_info 를 열지 않는다([[23]]) — 미수행 read 목록은 x460 산출물(이미 확정)에서 상수로
  받지 않고, **선언에 등장하는 read 이름 전부**를 표로 낸다(어느 read 가 몇 곳에 걸렸는지).

## 이 감사가 확정하는 것 (2026-08-21 1차 실행·세션 실측)
    T2_PIN_READ    발화 51줄 = 6 태스크(055·100·072·016·050·098)뿐 — **074·079·094·085·073 은 0**
    T2_DEMANDED_STEP  read-미수행 6 태스크 전부 ABSENT · 발화처(100·016·063·098)의 표적은
                      verify_identity·get_referrals·get_all_user_accounts·check_card_application_fit 4종뿐
    구조 원인 = 선언 위상: get_bank_account_transactions 피의존 1(dep=get_interest_correction 뿐)
               · get_payment_history 피의존 1(dep=check_cli_eligibility 뿐)
               ⇒ 이 태스크들이 실제로 시도하는 write 에는 read 요구가 안 걸려 있어
                 요건 큐가 수요를 못 내고, 핀은 시도조차 되지 않는다.
    ⇒ 처방 후보는 **A3 선언 보강**(각 write 의 requires_reads — 단 정책 축자 출처 필수·[[23]])이지
      새 레버가 아니다. 저작은 별도 단계([[72]] 1회 완결 저작).

사용: py x461_readgold_lever_liveness.py   (cwd=scripts/distill/tau2 · 로그는 sim_results 로컬)
"""
import collections
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_precedence as prec            # noqa: E402  선행 선언의 유일한 입구([[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
LOGS = [os.path.join(REP, "sim_results", f) for f in
        ("bank_t7328_halfA_20260819r.log.gz", "bank_t7328_halfB_20260819r2.log.gz")]
SIM = re.compile(r"\[sim=(task_\d+)[^\]]*\]")
TAGS = ("T2_PIN_READ_STEPS", "T2_PIN_READ", "T2_DEMANDED_STEP", "T2_READ_ROUTINE")


def main():
    per = collections.defaultdict(collections.Counter)   # task -> tag -> n
    dem_targets = collections.Counter()                  # DEMANDED_STEP 이 지목한 satisfier
    cur = None
    for fn in LOGS:
        with gzip.open(fn, "rt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = SIM.search(ln)
                if m:
                    cur = m.group(1)
                for tg in TAGS:
                    if "[%s]" % tg in ln:
                        per[cur][tg] += 1
                        if tg == "T2_DEMANDED_STEP" and "→" in ln:
                            dem_targets[ln.split("→")[-1].strip()[:50]] += 1
                        break                            # PIN_READ_STEPS 가 PIN_READ 에 중복 매칭 방지

    with io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                 encoding="utf-8") as f:
        a2 = json.load(f)
    decls = prec.declarations(a2, (prec.SRC_REQUIRE_BEFORE, prec.SRC_REQUIRES_READS))
    refcount = collections.Counter()
    dep_of = collections.defaultdict(list)
    for dep, reads in decls:
        for r in reads:
            refcount[r] += 1
            dep_of[r].append(dep)

    tasks = sorted(per)
    print("=" * 96)
    print("x461 · 선행-read 레버 생존 감사 (t7328 40 sim 영속 로그 · 태그 grep)")
    print("=" * 96)
    for t in tasks:
        print("  %-10s %s" % (t, dict(per[t])))
    silent = [t for t in tasks if not per[t]]
    print("\nDEMANDED_STEP 표적 분포: %s" % dict(dem_targets))
    print("\npin 이 보는 선언(read 피의존):")
    for r, n in refcount.most_common():
        print("  %2d  %-40s deps=%s" % (n, r, dep_of[r]))

    out = {"logs": [os.path.basename(x) for x in LOGS],
           "per_task": {t: dict(per[t]) for t in tasks},
           "demanded_targets": dict(dem_targets),
           "read_refcount": {r: {"n": n, "deps": dep_of[r]} for r, n in refcount.items()}}
    p = os.path.join(REP, "x461_readgold_lever_liveness.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[산출물] → %s" % p)


if __name__ == "__main__":
    main()
