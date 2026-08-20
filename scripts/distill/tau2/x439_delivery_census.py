# -*- coding: utf-8 -*-
r"""x439 — **적격 집합을 손님에게 넘겼나** (전수 포렌식 · 2026-08-20 밤)

## 왜 (사용자 지시 *"그래라"* — 잔여 ⓓ)
`N97_TASK_ROOT_CAUSE_2026_08_06` 이 [M] 로 확정한 자리: 003 의 세 sim 이 **같은** `check_card_application_fit`
출력을 받았는데, **통과한 trial 은 적격 3장을 비교 속성과 함께 열거**해 손님이 gold(`Silver`)를 골랐고,
실패한 trial 들은 **한 장을 골라 밀었다**(Platinum). 즉 같은 입력에서 **전달만 하면** 정답이 나왔다.

여기서는 레버를 만들기 전에([[62]]) **그 현상이 얼마나 자주 나는지**를 먼저 센다 — 유료 런 0·LLM 0.

## 무엇을 세나 (전부 **존재 확인**뿐 · 파싱 0)
후보 이름은 **우리 A2 카드표의 닫힌 집합**(13개)에서 가져온다 — 도구 출력 본문을 뜯지 않는다([[59]]).
각 자리마다:
    ⒜ 도구가 후보 목록을 배달한 메시지(우리 이름이 2개 이상 등장)
    ⒝ 그 **다음 어시스턴트 발화**에 우리 이름이 **몇 개** 등장하나 (0 / 1 / ≥2)
    ⒞ 그 발화가 **쓰기 도구를 호출**했나(`apply_for_credit_card` 등)
    ⒟ 그 sim 의 `reward`
낱말 경계 검사는 `t2_probe._word_in` 정본을 쓴다(`EVERGREEN ⊃ GREEN` 오적중 방지·C515).

## 읽는 법
`1개만 등장 + 쓰기 호출` = **밀기**(push) · `2개 이상 등장` = **넘기기**(hand-over).
⚠이것은 **관측**이지 처방이 아니다. 밀기가 항상 나쁜 것도 아니다 — 손님이 *"당신이 골라라"* 라고
  말한 자리도 있다. 그래서 태스크별로 따로 낸다([[70]]).

사용: py -3 x439_delivery_census.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402
import t2_probe as PR  # noqa: E402
import x431_spec_selects as X  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
TAGS = ("bank_t7326_halfA_20260819q", "bank_t7326_halfB_20260819q",
        "bank_t7328_halfA_20260819r", "bank_t7328_halfB_20260819r")
WRITES = ("apply_for_credit_card", "open_bank_account", "order_debit_card", "apply_checking_account")


def names():
    """후보 이름의 **닫힌 집합** = 우리 A2 카드표의 키(13). 도메인 산문에서 뽑지 않는다."""
    return sorted(X.card_table()[0].keys())


def hits(text, ns):
    return [n for n in ns if PR._word_in(text, n)]


def all_tags():
    """로컬에 있는 결과 파일 전부 — 표본을 11 자리에서 수백 sim 으로 넓힌다([[09]] 무료)."""
    import glob
    base = os.path.abspath(os.path.join(REP, "sim_results"))
    return sorted({os.path.basename(p).replace(".results.json.gz", "")
                   for p in glob.glob(os.path.join(base, "*.results.json.gz"))})


def main():
    ns = names()
    rows = []
    tags = all_tags() if "--all" in sys.argv else TAGS
    for tag in tags:
        try:
            sims = F.sims(tag, ".results.json.gz")
        except Exception:
            continue
        for s in sims:
            task, trial = F.task_id(s), s.get("trial")
            reward = (s.get("reward_info") or {}).get("reward")
            msgs = s.get("messages") or []
            for i, m in enumerate(msgs):
                if (m.get("role") or "") not in ("tool", "user", "assistant"):
                    continue
                c = str(m.get("content") or "")
                # ★**우리 도구의 출력만** 센다 — `[checks] applied:` 는 `t2_compute.py:493` 이 붙이는
                #   우리 표식이다(도메인 산문 파싱 0·[[59]]). 이 필터가 없으면 KB 검색 결과가 카드 이름을
                #   여럿 담고 있어 자리 수가 3배로 부푼다(1차판 140 자리 중 대부분이 그것이었다).
                if "[checks] applied:" not in c or len(hits(c, ns)) < 2:
                    continue
                # 그 다음 **말이 있는** 어시스턴트 발화(툴콜만 있는 빈 메시지는 건너뛴다)
                nxt, ni = None, None
                for j in range(i + 1, min(i + 8, len(msgs))):
                    mj = msgs[j]
                    if (mj.get("role") or "") != "assistant":
                        continue
                    if str(mj.get("content") or "").strip() or (mj.get("tool_calls") or []):
                        nxt, ni = mj, j
                        break
                if nxt is None:
                    continue
                say = str(nxt.get("content") or "")
                named = hits(say, ns)
                calls = [((tc.get("function") or {}).get("name") or tc.get("name"))
                         for tc in (nxt.get("tool_calls") or [])]
                rows.append({"tag": tag, "task": task, "trial": trial, "msg": i, "next": ni,
                             "n_delivered": len(hits(c, ns)), "n_named": len(named),
                             "named": named, "wrote": [x for x in calls if x in WRITES],
                             "calls": calls, "reward": reward})
    if "--all" in sys.argv:
        # ★교차표 — 전달 유형이 성적을 예측하나(태스크별·[[08]] 집계 직행 금지)
        import collections
        ct = collections.Counter()
        for r in rows:
            k = "넘기기" if r["n_named"] >= 2 else "밀기" if r["n_named"] == 1 else "무언급"
            ct[(r["task"], k, r["reward"] == 1.0)] += 1
        print("=" * 100)
        print("x439 --all · 자리 %d · 태그 %d" % (len(rows), len(tags)))
        print("%-10s %-8s %6s %6s %8s" % ("task", "전달", "pass", "fail", "pass율"))
        for task in sorted({r["task"] for r in rows}):
            for k in ("넘기기", "밀기", "무언급"):
                p_, f_ = ct[(task, k, True)], ct[(task, k, False)]
                if p_ + f_:
                    print("%-10s %-8s %6d %6d %8.2f" % (task, k, p_, f_, p_ / float(p_ + f_)))
        tot = collections.Counter()
        for (task, k, ok), v in ct.items():
            tot[(k, ok)] += v
        print("%-10s %-8s %6s %6s %8s" % ("─" * 8, "─" * 6, "─" * 5, "─" * 5, "─" * 6))
        for k in ("넘기기", "밀기", "무언급"):
            p_, f_ = tot[(k, True)], tot[(k, False)]
            if p_ + f_:
                print("%-10s %-8s %6d %6d %8.2f" % ("전체", k, p_, f_, p_ / float(p_ + f_)))
        q = os.path.abspath(os.path.join(REP, "x439_delivery_census_all.json"))
        with io.open(q, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("→ %s" % q)
        return 0
    print("=" * 100)
    print("x439 · 적격 집합 전달 전수 · 자리 %d" % len(rows))
    print("=" * 100)
    push = hand = quiet = 0
    for r in rows:
        kind = ("넘기기" if r["n_named"] >= 2 else "밀기" if r["n_named"] == 1 else "무언급")
        push += kind == "밀기"
        hand += kind == "넘기기"
        quiet += kind == "무언급"
        print("  %-9s t%s msg%-3d 배달 %d → 다음 발화 이름 %d [%s] %s reward=%s"
              % (r["task"], r["trial"], r["msg"], r["n_delivered"], r["n_named"], kind,
                 ("· write " + ",".join(r["wrote"])) if r["wrote"] else "", r["reward"]))
    print(chr(10) + "  ★ 넘기기 %d · 밀기 %d · 무언급 %d (자리 %d)" % (hand, push, quiet, len(rows)))
    by = {}
    for r in rows:
        k = (r["task"], "넘기기" if r["n_named"] >= 2 else "밀기" if r["n_named"] == 1 else "무언급")
        by[k] = by.get(k, 0) + 1
    print("  태스크별: %s" % json.dumps({"%s/%s" % k: v for k, v in sorted(by.items())}, ensure_ascii=False))
    p = os.path.abspath(os.path.join(REP, "x439_delivery_census.json"))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
