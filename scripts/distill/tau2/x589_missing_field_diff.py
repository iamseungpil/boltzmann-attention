# -*- coding: utf-8 -*-
r"""x589 - 빠진 gold 변이를 **같은 키의 실제 호출과 필드 단위로 맞대어** 원인을 이름 붙인다.

## 왜 (2026-08-28 밤)

`x588` 이 040 을 `CALLED_THEN_ERROR` 로 판정했는데 **틀렸다**: msg[47]·[49]·[55] 는 초기 인자
형식 오류였고 실제 8 건은 msg[50]~[70] 에서 **전부 성공**했다. 도구 단위로 오류를 찾으면 앞선
실패 한 건이 뒤의 성공 전부를 오염시킨다. ⇒ **gold 변이 하나마다 같은 키의 호출을 찾아
필드 단위로 비교**해야 원인이 이름을 얻는다.

## 어떻게 (닫힌 술어 · 추측 0)

  · 키 = `transaction_id` > `account_id` > `card_last_4_digits` 순으로 존재하는 첫 필드.
  · gold 변이마다 **같은 키 값**을 가진 성공 호출을 찾는다(오류 반환은 제외).
  · 찾으면 **다른 필드만** 나열한다 = 그 스텝의 정확한 원인.
  · 못 찾으면 `NO_CALL_WITH_THIS_KEY` (키 자체를 안 건드림).
"""
import json
import sys

sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")
import t2_forensic as F

KEYS = ("transaction_id", "account_id", "card_last_4_digits", "agent_tool_name")


def inner_args(a):
    if not isinstance(a, dict):
        return {}
    x = a.get("arguments")
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    if isinstance(x, dict):
        return x
    return a


def keyof(d):
    for k in KEYS:
        if k in d:
            return k, str(d[k])
    return None, None


def succeeded_calls(msgs):
    out = []
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            nm = str(tc.get("name") or "")
            if "unlock" in nm:
                continue
            a = tc.get("arguments")
            a = a if isinstance(a, dict) else (json.loads(a) if isinstance(a, str) and a.strip().startswith("{") else {})
            eff = a.get("agent_tool_name") or nm
            args = inner_args(a)
            ok = None
            for j in range(i + 1, len(msgs)):
                mj = msgs[j]
                if mj.get("role") != "tool" or mj.get("id") != tc.get("id"):
                    continue
                ok = not str(mj.get("content") or "").lstrip().startswith("Error:")
                break
            if ok:
                out.append((i, eff, args))
    return out


def main(argv=None):
    tag = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else "bank_t7376_treat_20260828"
    want = (argv or sys.argv[1:])[1:] or None
    for s in F.sims(tag):
        if (s.get("reward_info") or {}).get("reward") == 1.0:
            continue
        key = F.simtag(s)
        if want and not any(w in key for w in want):
            continue
        msgs = s.get("messages") or []
        calls = succeeded_calls(msgs)
        d = F.mutation_diff(s, F.mutating_tools(), tag=None) or {}
        miss = [e for e in (d.get("missing") or ()) if isinstance(e, dict)]
        if not miss:
            continue
        print("=" * 108)
        print("%s   빠진 변이 %d" % (key, len(miss)))
        for e in miss:
            ga = inner_args(e.get("args") or {})
            gname = (e.get("args") or {}).get("agent_tool_name") or e.get("name")
            kk, kv = keyof(ga)
            hit = [c for c in calls if c[1] == gname and str(inner_args({"arguments": c[2]}).get(kk, c[2].get(kk))) == kv] if kk else []
            if not hit:
                print("   %-38s key %s=%s -> NO_CALL_WITH_THIS_KEY" % (str(gname)[:38], kk, kv))
                continue
            i, _, aa = hit[-1]
            diff = {k: (ga.get(k), aa.get(k)) for k in set(ga) | set(aa)
                    if str(ga.get(k)) != str(aa.get(k))}
            print("   %-38s key %s=%s -> 호출 msg[%d] · 다른 필드 %d" % (str(gname)[:38], kk, kv, i, len(diff)))
            for k, (g, a) in sorted(diff.items()):
                print("        %-26s gold=%-28s 제출=%s" % (k, str(g)[:28], str(a)[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
