# -*- coding: utf-8 -*-
r"""x333 — **0단계**: "해야 하는데 안 한다"가 **학습벤치에도 있나** (사용자 지시 2026-08-15).

## 왜 이걸 먼저 재나

τ² banking 에서 knowing–doing 을 확정했다(C489: 이름 18/24 ↔ 실행 2/24 · 부정통제 0/24).
처방 후보가 **학습**(SFT 설치 + DPO 벌점·[[42]]·[[13]])인데, [[05]]/[[11]] 상 **τ² 로 학습하면
전이 주장이 무효**다. 학습은 **학습벤치·synth 에서만** 한다.

⇒ 선결 질문: **그 결손이 학습벤치에 존재하나?** 없으면 도메인-일반으로 가르칠 재료가 없고,
그러면 학습 경로 자체가 막힌다(그리고 τ² 학습으로 우회하면 [[05]] 위반).

## 무엇을 세나 (기계적·판단 0)

SOPBench 평가 필드가 이 현상을 **그대로** 담고 있다:

    action_should_succeed      = 그 자리에서 종단 행동을 **했어야 한다**
    action_successfully_called = 실제로 **불렀다**

  ⇒ **ACT-GAP** = `should=True ∧ called=False` = *"해야 하는데 안 했다"*
     그 안에서 다시 가른다:
       REPORT   마지막에 도구 없이 **텍스트만** 냈다      ← τ² 에서 본 그 형태
       WRONG    다른 도구는 불렀는데 종단 행동만 빠졌다
       ERROR    도구 호출 오류로 못 갔다(`no_tool_call_error=False`)
  ⇒ 대조군: `should=False` 에서 `called=True` = **과행동**(반대 방향·[[57]] 상쇄 감시)

⚠이 파일들은 벤치가 배포한 **여러 모델의 출력**이다(우리 런 아님) — 그래서 *"현상이 이 벤치에
존재하는가"* 를 모델-독립으로 볼 수 있다. 모델별로도 병기한다.
⚠판정은 벤치의 필드로만 한다. 어느 것이 옳은 행동인지 우리가 정하지 않는다([[23]]).

사용: seka python x333_actgap_census.py [output_dir]
"""
import collections
import glob
import io
import json
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/woori/scratch/SOPBench/output"


def model_of(path):
    b = os.path.basename(path)
    return b.split("-mode_")[0].replace("ast_", "")[:34]


def mode_of(path):
    b = os.path.basename(path)
    return (b.split("-mode_")[1].split("-")[0]) if "-mode_" in b else "?"


def final_is_text_only(item):
    """마지막 어시스턴트 발화가 **도구 없이 텍스트만**인가. 못 읽으면 None(모른다)."""
    try:
        inter = (item.get("interactions") or [{}])[0].get("interaction")
    except Exception:
        return None
    if not isinstance(inter, list) or not inter:
        return None
    # ⚠SOPBench 는 `role` 이 아니라 **`sender`** 를 쓴다(2026-08-15 수리 — 첫 판이 이걸 놓쳐
    #   88.6% 를 '못 읽음' 으로 버렸다). 도구 반환 메시지는 `tool_call_id` 로 구분된다.
    for m in reversed(inter):
        if not isinstance(m, dict):
            continue
        if m.get("tool_call_id") is not None:      # 도구 반환 — 발화자가 아니다
            continue
        who = str(m.get("sender") or m.get("role") or "").lower()
        if who and who not in ("assistant", "agent", "ai"):
            continue
        has_tc = bool(m.get("tool_calls")) or bool(m.get("function_call"))
        has_txt = bool(str(m.get("content") or "").strip())
        return (not has_tc) and has_txt
    return None


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*.json")))
    if not files:
        print("출력 파일 없음: %s" % ROOT); return 1
    tot = collections.Counter()
    by_model = collections.defaultdict(collections.Counter)
    by_domain = collections.defaultdict(collections.Counter)
    kinds = collections.Counter()
    unread = 0
    for p in files:
        if "-mode_fc" not in p:          # 도구 호출 모드만 — 텍스트 예측 모드는 이 현상을 표현 못 한다
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        mdl, dom = model_of(p), os.path.basename(os.path.dirname(p))
        for item in (d if isinstance(d, list) else []):
            for ev in (item.get("evaluations") or []):
                should = bool(ev.get("action_should_succeed"))
                called = bool(ev.get("action_successfully_called"))
                key = None
                if should:
                    tot["should=True"] += 1
                    by_model[mdl]["should"] += 1
                    by_domain[dom]["should"] += 1
                    if not called:
                        tot["ACT-GAP"] += 1
                        by_model[mdl]["gap"] += 1
                        by_domain[dom]["gap"] += 1
                        if ev.get("no_tool_call_error") is False:
                            key = "ERROR"
                        else:
                            t = final_is_text_only(item)
                            if t is None:
                                key = "?(궤적 못 읽음)"
                            elif t:
                                key = "REPORT(텍스트만)"
                            else:
                                key = "WRONG(다른 도구)"
                        kinds[key] += 1
                else:
                    tot["should=False"] += 1
                    if called:
                        tot["과행동(should=False∧called)"] += 1
    print("=== SOPBench fc-mode 전수 (%s)" % ROOT)
    for k in ("should=True", "ACT-GAP", "should=False", "과행동(should=False∧called)"):
        print("   %-32s %d" % (k, tot[k]))
    if tot["should=True"]:
        print("   ★ACT-GAP 비율 = %.1f%%" % (100.0 * tot["ACT-GAP"] / tot["should=True"]))
    print("\n=== ACT-GAP 의 내역")
    g = sum(kinds.values()) or 1
    for k, v in kinds.most_common():
        print("   %-22s %4d (%.1f%%)" % (k, v, 100.0 * v / g))
    print("\n=== 모델별 (should → gap)")
    for m in sorted(by_model, key=lambda x: -by_model[x]["should"]):
        s, gp = by_model[m]["should"], by_model[m]["gap"]
        if s:
            print("   %-36s %4d → %4d (%.0f%%)" % (m, s, gp, 100.0 * gp / s))
    print("\n=== 도메인별")
    for dmn in sorted(by_domain):
        s, gp = by_domain[dmn]["should"], by_domain[dmn]["gap"]
        if s:
            print("   %-16s %4d → %4d (%.0f%%)" % (dmn, s, gp, 100.0 * gp / s))
    print("\n판정: ACT-GAP 이 여러 모델·도메인에 걸쳐 존재하고 그 안에 REPORT 형태가 있으면 "
          "**학습벤치에서 가르칠 재료가 있다**. REPORT 가 0 이면 τ² 특이 현상이므로 "
          "도메인-일반 학습 경로는 막힌다([[05]]: τ² 학습 금지).")


if __name__ == "__main__":
    sys.exit(main() or 0)
