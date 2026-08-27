# -*- coding: utf-8 -*-
r"""x566 — 016 의 gap 1 앞 한 칸: 진단 서브에게 **행으로 · 정의에 맞게** 물으면 갈리는가.

## 왜 여기인가 (2026-08-27 · gap 우선)

016 의 gap 은 **1**(`MISSING submit_transaction`)이고 hard-0 에서 제일 얕다. 그 한 칸은 손님이
찍는데, user-sim 은 에이전트가 ①그 추천을 지목하고 ②구체 금액을 말해야 찍는다. ②는 라이브가
이미 한다(t7363 msg[44] *"$750 within 60 days"*) — 다만 **Bronze 에 붙여서** 한다. 그러니 남은
칸은 ① 하나다.

## 선언된 물음이 그 자리를 만든다

    `diagnose_prompt`: *"One of these records did not pay out. Reply with that record's
                        **account type** exactly as written above …"*

둘이 겹쳐 있다:
  ⑴ **단위가 이름**이다. 016 원장은 네 이름이 전부 상태를 2~3종 이고 있어(x554·x559 실측)
     이름으로는 행이 안 정해진다 — 그래서 `T2_DIAG_UNAMBIGUOUS` 가 **침묵**한다(t7364 실측).
  ⑵ *"did not pay out"* 은 우리가 같은 블록에 실어 보내는 정의와 **겹친다**:
     `COMPLETE — … met the criteria to get the referral bonus`.
     그 정의 아래에서 *"못 받은 것"* 은 COMPLETE 로도 읽힌다 — x559 에서 다섯 팔 40호출이
     **전부** `Bronze (COMPLETE)` 였던 이유다. 모델이 틀린 게 아니라 물음이 겹쳤다.

## 팔 — 바뀌는 것은 **선언의 물음 한 칸**뿐 ([[78]])

문맥은 정본 `t2_ledger.onto_context` 가 짓는 그 블록이다(대화 0자·선언대로).

    A_asis    선언된 물음 그대로                      ← 재현 게이트
    B_row     같은 물음, **행으로** (`<type> | <date>`)
    C_crit    행으로 + 정의를 그대로 되묻는다(*"아직 기준을 못 채운 기록"*)
    N_len     길이만 맞춘 무관 문장([[57]])

## 채점 — 닫힌 술어 · gold 무참조([[23]])

답이 지목한 **행**과 그 행의 상태를 원장에서 읽어 그대로 인쇄한다. 어느 상태가 정답인지
엔진이 정하지 않는다 — 표를 보고 사람이 판정한다.

사용: PYTHONPATH=. py -3 x566_diag_prompt_row_iso.py --port 8140
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI                                       # noqa: E402
import t2_ledger as LG                                              # noqa: E402
import x554_diag_mispick_iso as X554                                # noqa: E402
import x559_016_row_pick_iso as X559                                # noqa: E402

NL = chr(10)

ROW_TAIL = ("Reply with that record's account type AND its date, exactly as written above, "
            "in the form `<account type> | <date>`, and one short sentence saying what the "
            "definitions above make of it. Nothing else.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tags", default="bank_t7363_hard0_20260827,bank_t7356_grpB3_20260826")
    ap.add_argument("--sim", default="task_016#s626729")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    a2 = GI.load_domain_a2("banking_knowledge") or {}
    spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get("diagnose_prompt")), None)
    if not spec:
        print("선언에 원장 스펙이 없다", file=sys.stderr)
        return 2
    keys = list(spec.get("row_keys") or ())
    a3r = ((a2.get("policy_ontology") or {}).get("rows")) or ()
    declared = str(spec.get("diagnose_prompt") or "")

    ctxs = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        rows = X554.rows_from_traj(tag, a.sim, keys)
        if not rows:
            print("[skip] %s 원장 없음" % tag)
            continue
        ctxs.append((tag, rows))
    if not ctxs:
        print("잴 문맥이 없다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    g, dt, st = spec.get("group_field"), spec.get("date_field"), spec.get("status_field")

    def ask_of(kind, block):
        if kind == "A_asis":
            return declared.replace("{block}", block)
        head = declared.split("{block}")[0] + block + NL + NL
        if kind == "B_row":
            return head + "One of these records did not pay out. " + ROW_TAIL
        if kind == "C_crit":
            return (head + "Using only the status definitions above, which of these records has "
                    "not yet met the criteria for the referral bonus? " + ROW_TAIL)
        return (head + "These records were retrieved earlier and have not changed since; they "
                "are current and complete for this customer. One of these records did not pay "
                "out. " + declared.split("{block}")[-1].strip())

    print("# x566 — 진단 물음: 이름 ↔ 행")
    for tag, rows in ctxs:
        blk = LG.onto_context(rows, spec, a3r)
        dist = collections.Counter(str(r.get(st) or "") for r in rows)
        print("  %-26s 행 %d · 상태 분포 %s · 블록 %d자" % (tag[:26], len(rows), dict(dist), len(blk)))
    if a.wiring_only:
        tag, rows = ctxs[0]
        for k in ("A_asis", "B_row", "C_crit"):
            q = ask_of(k, LG.onto_context(rows, spec, a3r))
            print("--- %s 꼬리 ---" % k)
            print("   " + " ".join(q.split())[-260:])
        return 0

    print()
    print("%-26s %-8s %-5s %-34s %s" % ("tag", "팔", "temp", "지목한 행", "상태"))
    print("-" * 100)
    tally = collections.defaultdict(collections.Counter)
    for tag, rows in ctxs:
        blk = LG.onto_context(rows, spec, a3r)
        for nm in ("A_asis", "B_row", "C_crit", "N_len"):
            q = ask_of(nm, blk)
            for tp, cnt in ((0.0, 1), (a.temp, a.n)):
                for _ in range(cnt):
                    try:
                        rep = " ".join(str(X559.gen(a.port, q, 96, tp)).split())
                    except Exception as e:
                        print("%-26s %-8s %-5s 호출 실패: %r" % (tag[:26], nm, tp, e))
                        continue
                    pick, status = X559.row_of(rep, rows, spec)
                    tally[(tag, nm)][status or "판정 불가"] += 1
                    print("%-26s %-8s %-5s %-34s %s"
                          % (tag[:26], nm, tp, str(pick)[:34], status or "판정 불가"))
    print()
    print("## 지목한 행의 **상태 분포** (엔진은 어느 것이 정답인지 모른다)")
    for tag, rows in ctxs:
        for nm in ("A_asis", "B_row", "C_crit", "N_len"):
            print("   %-26s %-8s %s" % (tag[:26], nm, dict(tally[(tag, nm)])))
    print()
    print("⚠A_asis 가 이미 IN_PROGRESS 를 내면 물음은 원인이 아니다([[62]] 2b).")
    print("⚠N_len 이 C_crit 과 같으면 그 이득은 **길이**다([[57]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
