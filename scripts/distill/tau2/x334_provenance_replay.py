# -*- coding: utf-8 -*-
r"""x334 — provenance 게이트를 켜면 **무엇을 막고 무엇을 잘못 막나** (오프라인 재생·GPU 0).

## 왜 켜기 전에 이걸 하나

게이트는 지금 **꺼져 있다**(`T2_PROVENANCE` 미설정 · t7295/t7296 `PROVENANCE_R1B` 0건) 그리고
켜도 **banking 은 못 봤다** — `_provenance_deny` 가 discoverable 래퍼의 **중첩 인자**를 안 폈다.
오늘 그 사각을 고쳤다(`test_provenance_nested`). 이제 켤 수 있지만, C45 의 *over-block 0* 은
**중첩을 안 보던 시절 수치**라 그대로 쓸 수 없다 ⇒ [[57]] 상쇄를 **먼저** 잰다.

## 방법 (판단 0·재생만)

영속된 궤적을 순서대로 걸으며, **각 도구 호출 직전까지의 문맥**(그 시점까지의 손님 발화 +
도구 출력)으로 `_provenance_deny` 를 그대로 돌린다. 실제 엔진과 같은 함수·같은 힌트다.

    BLOCK-GOLD   막았는데 그 호출이 **gold 액션**이었다      ← ★over-block(해악·이 수가 핵심)
    BLOCK-OTHER  막았고 gold 액션이 아니었다                  ← 잠재 이득(과행동·날조 차단)
    PASS         안 막았다

⚠**시점 정합**: 문맥은 반드시 **그 호출 이전**만 쓴다. 도구가 우리 인자를 되울리므로
  전체 궤적으로 재면 날조가 스스로를 증명한다(2026-08-15 실물: `tx111111` 이 write **이후**
  출력에만 있었다).
⚠gold 판정은 하네스의 `action_checks` 에서 **도구 이름 + 의미 비교**로 한다(`action_match` 는
  표기로 무너지므로 쓰지 않는다·C486).

## 읽는 법 (사전 고정)

    BLOCK-GOLD == 0                  → over-block 0 재확인 ⇒ **켤 근거**
    BLOCK-GOLD ≤ 2 이고 BLOCK-OTHER 가 그보다 훨씬 크다 → 조건부 채택(어느 태스크인지 병기)
    BLOCK-GOLD 가 크다               → **켜지 않는다**. 힌트/술어를 좁히고 다시 잰다

사용: PYTHONPATH=…/tau2-bench/src python x334_provenance_replay.py [tag ...]
"""
import collections
import io
import json
import os
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F                                            # noqa: E402
import t2_gate_patch as G                                          # noqa: E402


class TC(object):
    """엔진이 받는 것과 같은 모양의 최소 객체."""
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def norm(v):
    if isinstance(v, str):
        try:
            j = json.loads(v)
            if isinstance(j, (dict, list)):
                return norm(j)
        except Exception:
            pass
        return v.strip().lower()
    if isinstance(v, dict):
        return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def gold_keys(sim):
    """gold 액션을 (대상도구, 정규화 인자) 로."""
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = (ck.get("action") or {}).get("arguments") or {}
        t = a.get("agent_tool_name") or a.get("user_tool_name") or (ck.get("action") or {}).get("name")
        inner = a.get("arguments", a)
        out.append((str(t), norm(inner)))
    return out


def main(tags):
    kinds = collections.Counter()
    bygold = collections.Counter()
    detail = []
    for tag in tags:
        try:
            sims = F.scored(tag)
        except Exception as e:
            print("로드 실패 %s: %r" % (tag, e)); continue
        for s in sims:
            gk = gold_keys(s)
            ctx = ""
            for m in (s.get("messages") or []):
                # ★호출 검사 → **그 다음에** 이 메시지를 문맥에 넣는다(시점 정합)
                for tc in (m.get("tool_calls") or []):
                    if m.get("role") != "assistant":
                        continue
                    args = F.argsof(tc)
                    obj = TC(F.nameof(tc), args)
                    try:
                        pd = G._provenance_deny(obj, ctx.lower())
                    except Exception:
                        pd = None
                    if not pd:
                        kinds["PASS"] += 1
                        continue
                    tgt = F.inner_name(args) or F.nameof(tc)
                    inner = args.get("arguments", args)
                    is_gold = any(t == tgt and g == norm(inner) for t, g in gk)
                    k = "BLOCK-GOLD" if is_gold else "BLOCK-OTHER"
                    kinds[k] += 1
                    if is_gold:
                        bygold[F.task_id(s)] += 1
                    detail.append((k, F.task_id(s), tgt, str(pd[1])[:90]))
                c = str(m.get("content") or "")
                if m.get("role") in ("tool", "user") and c:
                    ctx += c
    tot = sum(kinds.values()) or 1
    print("=== provenance 오프라인 재생 · %s" % ", ".join(tags))
    for k in ("PASS", "BLOCK-OTHER", "BLOCK-GOLD"):
        print("   %-12s %5d (%.1f%%)" % (k, kinds[k], 100.0 * kinds[k] / tot))
    print("\n★over-block(BLOCK-GOLD) 태스크별:", dict(bygold) or "없음")
    print("\n막힌 것 표본(최대 20):")
    for d in detail[:20]:
        print("   %-12s %-10s %-42s %s" % d)
    print("\n판정(사전 고정): BLOCK-GOLD==0 → 켤 근거 · ≤2 이고 OTHER 가 훨씬 크면 조건부 · "
          "크면 켜지 않고 술어를 좁힌다")


if __name__ == "__main__":
    main(sys.argv[1:] or ["bank_t7295_a_20260815n", "bank_t7295_b_20260815n"])
