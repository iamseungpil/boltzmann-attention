# -*- coding: utf-8 -*-
r"""x309 — 배치 레버의 **마지막 고리**: 서브가 만든 호출을 메인에 전달하면 메인이 실행하는가.

x308 이 처방에 licence 를 줬다(같은 사실·같은 도구·배치만 다름):
  A_MAIN 0/8 · **B_SUB 7/8** · C_SUBQ 8/8(JSON) · D_NOBASIS 0/8(근거 없으면 안 지어냄)
그런데 출시 형태는 *"엔진이 대신 실행"* 이 아니라 **[[65]] 답만 메인에 올린다** 여야 한다
([[05]] Q3: scaffold 가 도메인 행동을 수행하면 금지). 그 형태는 **아직 안 쟀다** — 전달된 제안을
메인이 집어서 실제로 호출하는지가 미측정 고리다. 안 재고 구현하면 [[62]] 위반이라 이 프로브가 선결.

셀 4 (n=8·서브 산출은 **라이브로 생성**해 그대로 전달 — 손으로 값을 쓰지 않는다):
  A_MAIN    x308 A_MAIN 축자                                   ← 기준선(0/8 재현)
  B_DELIV   A_MAIN + **서브 JSON 산출 축자 전달**              ← 출시 후보 형태
  C_DELIV1  A_MAIN + 서브 산출 중 **첫 호출 하나만** 전달       ← 목록 크기가 인자인지(x287b 교훈)
  D_WRONG   A_MAIN + **무관 도구**로 바꾼 동형 제안 전달        ← 순응 부정통제([[57]]·x304 동형)

⚠서브 산출이 비면(=근거 부족) 그 시행은 전달 팔에서 제외하고 인쇄한다 — 빈 제안을 전달하는 것은
출시 형태가 아니다(폴백 경로).

판정(사전 고정):
  B_DELIV ≥6 ∧ A_MAIN ≤2 ∧ D_WRONG ≤2 → **전달 배치 출시**(서브 산출 → 메인 리마인더)
  B_DELIV ≤2 ∧ C_DELIV1 ≥6 → 목록 크기가 인자 → 한 건씩 전달로 설계
  B_DELIV ≤2 ∧ C_DELIV1 ≤2 → 전달로는 안 열림 → 출시 형태 재설계(초안 대체 경로 검토·재측정)
  D_WRONG ≥3 → 문면-순응(제안이면 뭐든 따름) = 프로브 무효·안전 문제 선결

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x309_sub_delivery_iso.py [N]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x302_kindmap_ownership_probe as K                          # noqa: E402
import x306_phase_fold_probe as F                                 # noqa: E402
import x308_write_isolate_iso as S                                # noqa: E402

TARGET = K.TARGET
WRONG = "get_debit_dispute_status_7483"      # 레지스트리 실재·이 요청과 무관(x304 통제와 동일 계열)
DELIV_T = ("Note: [ISOLATED-FORMALIZATION] this decision was formalized in isolation from the "
           "audit results already in this conversation, and it produced the following call(s). "
           "Execute them now with your own tools:\n%s")


def sub_calls(sim, cut, note, temp):
    """x308 C_SUBQ 축자 프레이밍으로 **라이브 서브 1회** — 산출 JSON 을 그대로 돌려준다."""
    fees, asks = S.fee_lines(sim, cut), S.ask_lines(sim, cut)
    body = "\n".join([S.SUB_HEAD, ""] + asks + [""] + fees + ["", note, S.SUB_JSON])
    r = chat(body, None, temp, 1500)
    t = str(r.get("content") or "")
    m = re.search(r'\{.*"calls".*\}', t, re.S)
    if not m:
        return []
    try:
        calls = (json.loads(m.group(0)) or {}).get("calls") or []
    except Exception:
        return []
    return [c for c in calls if isinstance(c, dict) and TARGET in json.dumps(c, ensure_ascii=False)]


def render(calls):
    return "\n".join("  - %s" % json.dumps(c, ensure_ascii=False) for c in calls)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in X.load(K.TAG) if s["task_id"] == K.TASK
               and s.get("reward_info") is not None)
    cut = F.cut_of(sim)
    note = K.NOTE_T % ", ".join(K.kind_matches())
    tools = U.tools_of(sim)
    a_main = F.early_ctx(sim, cut) + "\n[system] " + note
    print("x309 cut=%d · n=%d · URL=%s\n" % (cut, n, os.environ.get("T2_PROBE_URL", "8140⚠")))
    skipped = collections.Counter()
    for label in ("A_MAIN", "B_DELIV", "C_DELIV1", "D_WRONG"):
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            temp = 0.0 if i == 0 else 0.7
            body = a_main
            if label != "A_MAIN":
                calls = sub_calls(sim, cut, note, temp)
                if not calls:
                    skipped[label] += 1
                    print("    [%s %02d] 서브 산출 없음 — 제외(폴백 경로)" % (label, i), flush=True)
                    continue
                if label == "C_DELIV1":
                    calls = calls[:1]
                if label == "D_WRONG":
                    calls = [dict(c, tool=WRONG) for c in calls]
                body = a_main + "\n[system] " + DELIV_T % render(calls)
            try:
                r = chat(body, tools, temp, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            blob = " ".join(str(t) for t in (r.get("tool_calls") or []))
            k = F.classify(r)
            ok = (WRONG in blob) if label == "D_WRONG" else (k == "target")
            hit += ok
            cnt[k] += 1
            print("    [%s %02d] %s%s" % (label, i, "HIT" if ok else "-",
                                          "" if ok else " (%s)" % k), flush=True)
        m = n - skipped[label]
        print("%-9s %d/%d%s · %s\n" % (label, hit, m,
                                       "" if not skipped[label] else " (제외 %d)" % skipped[label],
                                       dict(cnt)))
    print("※ 판정(사전 고정): B_DELIV ≥6 ∧ A_MAIN ≤2 ∧ D_WRONG ≤2 → 전달 배치 출시 · "
          "B_DELIV ≤2 ∧ C_DELIV1 ≥6 → 한 건씩 전달 · 둘 다 ≤2 → 전달 형태 재설계 · "
          "D_WRONG ≥3 → 무효(문면-순응).")


if __name__ == "__main__":
    main()
