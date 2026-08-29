# -*- coding: utf-8 -*-
r"""x593 - t7389 두 팔의 **채널 판정** (모델 0 · 무료 · 계수만).

## 왜 이 파일이 따로 있나 (2026-08-29)

`run_t7389.sh` §④ 의 계수기 **두 개가 틀렸다**. 스모크에서 잡혔다:

  · `[components]` 를 **stderr 로그**에서 셌다 -> 0. 그런데 그 문면은 stderr 마커가 아니라
    **도구 반환문**이라 사이드카에는 4건 있었다. `TASK_094.md` §8 이 이미 박제한 함정이다
    (*"READ-FIRST 는 stderr 마커가 아니라 도구 반환문에 있다. 로그 grep 0 을 미발화로 읽으면
    오진한다"*·[[55]]) — 같은 자리에 두 번째로 빠졌다.
  · `거래read` 를 로그의 도구 **이름 등장 횟수**로 셌다 -> `[T2_SG_REQREADS] ... missing reads
    ['get_bank_account_transactions']` 라는 **거절문**까지 read 로 센다. 요구와 실행이 섞인다.

런이 도는 중이라 스크립트를 고칠 수 없었다(bash 는 실행 중 파일을 이어 읽는다). 그래서 판정은
여기서 한다. 런이 끝난 뒤 `run_t7389.sh` §④ 도 이 술어로 맞춘다.

## 술어 (닫힘 · 추측 0)

  거래read        `role=assistant` 의 tool_call 중 **실효 이름**이 그 getter 인 것 (호출만)
  파생-검산       stderr 마커 (여기는 stderr 가 맞는 자리)
  -> None         스칼라 abstain 횟수
  [components]    **role=tool 본문**에서 센다 (반환문이 사는 자리)
  재호출           components 문면이 실린 tool 메시지 **뒤에** 같은 도구 호출이 또 오나
  오퍼랜드         `get_interest_correction` 호출마다 (expected_apy, actual_apy, 결과)

⛔집계로 원인을 말하지 않는다([[08]]) — 모든 행이 `sim # msg` 로 짚힌다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F

READ = "get_bank_account_transactions"
APY = "get_correct_savings_apy"
COR = "get_interest_correction"


def eff(tc):
    a = F.argsof(tc) or {}
    return F.inner_name(a) or F.nameof(tc) or ""


def scan(sim):
    msgs = sim.get("messages") or []
    reads, notes, recall, ops = [], [], 0, []
    last_note = None
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool" and "[components]" in str(m.get("content") or ""):
            notes.append(i)
            last_note = i
        for tc in (m.get("tool_calls") or []):
            nm = eff(tc)
            if READ in nm:
                reads.append(i)
            if nm == APY and last_note is not None and i > last_note:
                recall += 1
                last_note = None
            if F.nameof(tc) == COR:
                a = F.argsof(tc) or {}
                ops.append((i, a.get("expected_apy"), a.get("actual_apy"), a.get("principal")))
    return {"reads": reads, "notes": notes, "recall": recall, "ops": ops}


def main(argv=None):
    args = (argv or sys.argv[1:]) or ["bank_t7389_control_20260829", "bank_t7389_treat_20260829"]
    for tag in args:
        try:
            sims = F.sims(tag)
        except Exception as ex:
            print("(못 읽음) %s : %r" % (tag, ex))
            continue
        log = ""
        try:
            log = F.log_text(tag) or ""
        except Exception:
            pass
        print("=" * 112)
        print("# %s   (sim %d · pass %d)"
              % (tag, len(sims),
                 sum(1 for s in sims if (s.get("reward_info") or {}).get("reward") == 1.0)))
        print("   stderr:  파생-검산 통과 %d · 불성립 %d · `-> None` %d · REQREADS denied %d"
              % (log.count("파생-검산 통과"), log.count("파생-검산 불성립"),
                 log.count("%s -> None" % COR),
                 log.count("[T2_SG_REQREADS] %s denied" % COR)))
        for s in sims:
            r = scan(s)
            print("   %-20s reward=%-5s 거래read=%-2d [components]=%-2d 뒤이은재호출=%-2d %s"
                  % (F.simtag(s), (s.get("reward_info") or {}).get("reward"),
                     len(r["reads"]), len(r["notes"]), r["recall"], F.term_reason(s)))
            if r["reads"]:
                print("        거래read msg %s" % r["reads"][:6])
            for (i, e, a, p) in r["ops"]:
                print("        msg[%-3s] expected_apy=%-8s actual_apy=%-8s principal=%s"
                      % (i, e, a, p))
            d = F.mutation_diff(s, F.mutating_tools(), tag=tag) or {}
            for k in ("missing", "wrongarg", "extra"):
                for x in (d.get(k) or ()):
                    print("        %-9s %s" % (k, str(x.get("key"))[:130]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
