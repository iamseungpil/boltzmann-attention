# -*- coding: utf-8 -*-
r"""x305 — 캡 리셋을 짓기 전에: **옳은 이름이 후보에 있으면 formalize 가 그걸 고르는가**.

배경(x303/x304·087 `bank_t7286_a_20260814h`):
  x303  A_MIN 0/8 · B_FULL 0/8            → 능력 후보
  x304  B_STEP2(옳은 이름 동봉) **6/8**   → 문면 생존·잔여 = **이름이 그 자리에 닿는 것**
  라이브 STEP2 6회 발화가 전부 이름 노출(msg31) **이전**의 무관 이름(fraud_alert·debit_cards·
  freeze·unfreeze)이었고, 옳은 이름이 도착한 turn30~32 는 `resolve_cap(정체 3회)` 침묵.

수리 후보는 "회수-후보에 **새 이름**이 추가되면 정체 카운터 리셋"(=[[57]] 인자-변화 재시도)이다.
그런데 **리셋해도 formalize 가 계속 무관 이름을 고르면 이득이 0**이다. 그래서 레버 이전에
선택 자체를 잰다([[62]]).

측정 대상 = 라이브 함수 **그대로**([[03b]] 별도구현 금지): `t2_resolve.formalize_intent_tool`
을 import 해서 호출하고, **전송만** 프로브 `chat()` 으로 스텁한다(프롬프트 구성·파싱·집합
소속 판정 전부 라이브 코드).

셀 (n=8·후보 집합은 라이브 규칙 `_retrieved_unlockables` + 레지스트리 교집합으로 기계 산출):
  PRE    컷 = 이름 노출 **이전**(라이브 STEP2 가 실제 발화하던 구간) — 라이브 오선택 재현 통제
  POST   컷 = 이름 노출 **이후**(x303/x304 와 동일 컷 34)  ← 본 측정

판정(사전 고정):
  POST ≥6/8 TARGET → 선택은 된다 ⇒ 잔여는 순전히 **우리 캡 침묵** → 리셋 수리 정당(빼기).
  POST ≤2/8        → 선택이 병목 ⇒ **캡 리셋은 무의미** — 후보 랭킹 축으로 이동(측정 후).
  PRE 에서 TARGET 이 후보 집합에 **없음**을 함께 인쇄(타이밍 서사의 사실 확인).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x305_name_selection_iso.py [N]
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x303_account_reach_iso as Z                                # noqa: E402
from x297_registry_hygiene_probe import REGISTRY                  # noqa: E402
import t2_resolve as R                                            # noqa: E402

TARGET = Z.TARGET
UNLOCK = "unlock_discoverable_agent_tool"


class _M(object):
    """라이브 함수가 보는 최소 메시지 뷰 (dict 궤적 → 속성 접근)."""
    def __init__(self, m):
        self.role = m.get("role")
        self.content = m.get("content")
        self.tool_calls = [_TC(tc) for tc in (m.get("tool_calls") or [])]


class _TC(object):
    def __init__(self, tc):
        f = tc.get("function") or {}
        self.name = f.get("name") or tc.get("name")
        a = tc.get("arguments", f.get("arguments"))
        self.arguments = a if isinstance(a, dict) else {"_": str(a)}


class _UserMessage(object):
    def __init__(self, content="", role="user"):
        self.role, self.content = role, content


class _LA(object):
    """la.generate 스텁 — 프롬프트는 라이브가 만든 것을 그대로 받아 프로브 서버에 보낸다."""
    def __init__(self, temp):
        self.temp = temp
        self.last = None

    def generate(self, model=None, tools=None, messages=None, call_name=None, **kw):
        self.last = str(getattr(messages[0], "content", ""))
        r = chat(self.last, None, self.temp, 200)
        return type("S", (), {"content": r.get("content") or ""})()


class _Agent(object):
    llm = "probe"
    llm_args = {}


def cands_at(msgs, cut):
    """라이브 규칙 그대로 후보 집합 산출 (회수 텍스트 ∩ 레지스트리 − unlock 기시도)."""
    view = [_M(m) for m in msgs[:cut]]
    got = R._retrieved_unlockables(view, set(REGISTRY), UNLOCK)
    return [n for n in got if n in set(REGISTRY)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in X.load(Z.TAG) if s["task_id"] == Z.TASK
               and s.get("reward_info") is not None)
    msgs = sim["messages"]
    post = Z.cut_of(sim)
    # PRE = 이름이 회수 텍스트에 처음 등장하기 **전** (라이브 STEP2 발화 구간)
    pre = next(i for i, m in enumerate(msgs)
               if m.get("role") == "tool" and TARGET in str(m.get("content") or ""))
    print("x305 · PRE cut=%d · POST cut=%d · n=%d · URL=%s" % (
        pre, post, n, os.environ.get("T2_PROBE_URL", "8140(기본⚠)")))
    for label, cut in (("PRE", pre), ("POST", post)):
        cands = cands_at(msgs, cut)
        has = TARGET in cands
        print("\n%-5s 후보 %d개 · TARGET 포함=%s · %s" % (
            label, len(cands), has, ", ".join(cands[:8]) + (" …" if len(cands) > 8 else "")))
        if not cands:
            print("      (후보 0 — formalize 호출 자체가 없다)")
            continue
        cnt = collections.Counter()
        view = [_M(m) for m in msgs[:cut]]
        for i in range(n):
            la = _LA(0.0 if i == 0 else 0.7)
            got = R.formalize_intent_tool(_Agent(), la, _UserMessage, view, set(cands))
            cnt[got or "none"] += 1
        print("      TARGET %d/%d · %s" % (cnt.get(TARGET, 0), n, dict(cnt.most_common(6))))
    print("\n※ 판정(사전 고정): POST ≥6/8 → 선택 가능 ⇒ 잔여=우리 캡 침묵 → 리셋 수리 정당."
          " POST ≤2/8 → 선택이 병목 ⇒ 캡 리셋 무의미(후보 랭킹 축으로 이동).")


if __name__ == "__main__":
    main()
