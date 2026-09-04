# -*- coding: utf-8 -*-
"""P9 [[70]] 부호표 격리 — 재생성 호출을 쓰기 게이트에 **재진입**시키면 gold 칸도 함께 죽는가.

왜 이 프로브인가. P9 의 exit(«off 에서 금지 write 커밋 ∧ on 에서 deny»)는 이미 라이브 기록으로
충족돼 있다 — 027 은 같은 write(update e403)를 정상경로에서 **5회 live-DENY**(t55·57·63·65·71)
하고 `searchexhaust` 재생성 1발(t73)로 커밋했고(sim 내부 대조), 029 의 금지 write 5건은 오프라인
재실행 **5/5 DENY** 다. 남아 있던 미측정은 **부호표 반대편** 하나였다:
x737 축자 *"048 t63(gold `pay_credit_card`)·t123(gold unlock) … 예측: pay 는 `write_evidence_specs`
밖이라 무영향 · [미측정]"*.

선언 조회로 절반이 닫혔다(무료):
    update_transaction_rewards      WEV 사정권 **안**  ⇒ 재진입하면 막힌다 (금지 write · 사는 쪽)
    unlock_discoverable_agent_tool  전 쓰기 게이트 선언 **밖** ⇒ 재진입이 건드릴 수 없다 (0 비용)
    pay_credit_card_from_checking   WEV **밖** — 그러나 `write_arg_grounding` **안**  ⇒ x737 예측의 부분 반증

그래서 이 프로브가 재는 것은 정확히 하나다: **회수된 048 궤적의 gold `pay` 호출들이
`write_arg_grounding` 을 통과하는가.** 통과하면 부호표 반대편은 0이고, 막히면 재진입 배선은
gold 칸을 판다. 엔진 술어를 그대로 부른다(프롬프트 저작 0 · 판단 0 · [[78]]).
"""
import glob
import gzip
import io
import json
import os
import sys

TAU2 = r"C:\workspace\ba-frft\scripts\distill\tau2"
SR = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
sys.path.insert(0, TAU2)
os.chdir(TAU2)

from t2_gate_patch import _write_arg_ground_deny, _write_evidence_deny  # noqa: E402

A2 = json.load(io.open("a2/banking_knowledge.gate.json", encoding="utf-8"))
WAG = A2["write_arg_grounding"]
WEV = A2["write_evidence_specs"]


class TC(object):
    """엔진이 기대하는 tool_call 모양만 흉내낸다(값 생성 0)."""

    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = d.get("arguments") or {}
        self.id = d.get("id")


class MSG(object):
    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])]
        self.error = bool(d.get("error"))


def inner_tool(tc):
    """디스패처형 호출의 실제 도구 이름."""
    a = tc.arguments or {}
    return str(a.get("agent_tool_name") or tc.name or "")


def scan(task_id, want_prefixes):
    out = []
    for f in sorted(glob.glob(os.path.join(SR, "*.results.json.gz"))):
        try:
            d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        except Exception:
            continue
        for s in d.get("simulations", []):
            if s.get("task_id") != task_id:
                continue
            raw = s.get("messages") or []
            msgs = [MSG(m) for m in raw]
            for i, m in enumerate(msgs):
                for tc in m.tool_calls:
                    nm = inner_tool(tc)
                    if not any(nm.startswith(p) for p in want_prefixes):
                        continue
                    window = msgs[:i]          # 그 호출 시점까지가 근거 창
                    out.append((os.path.basename(f), i, nm, tc, window,
                                (s.get("reward_info") or {}).get("reward")))
    return out


def prefixes(specs):
    """사정권은 **선언에서 읽는다** — 도구 이름을 프로브에 타이핑하지 않는다([[71]]②)."""
    out = set()
    for sp in specs:
        p = (sp.get("applies_when") or {}).get("prefix")
        if p:
            out.add(p)
    return out


def main():
    WAG_P, WEV_P = prefixes(WAG), prefixes(WEV)
    print("=" * 96)
    print("P9 부호표 격리 — 재진입이 gold 칸을 파는가")
    print("=" * 96)
    print("write_arg_grounding 사정권 %d종: %s" % (len(WAG_P), sorted(WAG_P)))
    print("write_evidence_specs 사정권 %d종: %s" % (len(WEV_P), sorted(WEV_P)))
    gold_pay = sorted(WAG_P - WEV_P)          # WEV 밖인데 WAG 안 = x737 예측이 놓친 칸
    print("⇒ WEV 밖 ∧ WAG 안 (= x737 «무영향» 예측의 반례 후보): %s" % gold_pay)
    print()
    rows = []
    for p in gold_pay:
        rows += scan("task_048", (p,))
    if not rows:
        print("⛔재료 0건 — 회수분에 048 의 해당 호출이 없다. 주장 금지([[77]]).")
        return
    denied = passed = err = 0
    for fn, idx, nm, tc, window, rw in rows:
        try:
            d1 = _write_arg_ground_deny(window, tc, WAG)
        except Exception as e:
            err += 1
            print("  ERR  %-46s msg%-4d %r" % (fn[:46], idx, e))
            continue
        try:
            d2 = _write_evidence_deny(None, tc, WEV)
        except Exception:
            d2 = "(WEV: orch 필요 — 건너뜀)"
        v = "DENY" if d1 else "pass"
        if d1:
            denied += 1
        else:
            passed += 1
        print("  %-4s %-46s msg%-4d reward=%-5s %s" % (v, fn[:46], idx, rw,
                                                       (str(d1)[:90] if d1 else "")))
    print()
    print("회수분 pay 호출 %d건 · 통과 %d · DENY %d · 오류 %d" % (len(rows), passed, denied, err))
    print("⚠«gold 호출» 이 아니라 «회수된 호출» 이다 — 이 프로브는 gold 대조를 하지 않는다.")
    print()
    print("--- 부정통제: 금지 write(update_transaction_rewards)는 같은 술어에서 어떻게 되나 ---")
    rows2 = scan("task_027", ("update_transaction_rewards",)) + \
        scan("task_029", ("update_transaction_rewards",))
    if not rows2:
        print("  재료 0건 (027·029 의 update 호출이 회수분에 없다)")
    else:
        dn = sum(1 for r in rows2 if _write_arg_ground_deny(r[4], r[3], WAG))
        print("  update 호출 %d건 중 write_arg_grounding DENY %d" % (len(rows2), dn))
        print("  ⚠이 게이트는 **출처 검사**다 — 금지 write 를 막는 것은 WEV 이지 이쪽이 아니다.")


if __name__ == "__main__":
    main()
