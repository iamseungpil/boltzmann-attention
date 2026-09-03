# -*- coding: utf-8 -*-
"""`T2_ACT_DEMAND` 전달 수리 T2·T3·T4 단위검정 (2026-09-03).

정본: `reports/facet_rft_2026/CLAIM_DEMAND_ISO_VS_LIVE_AUDIT_2026_08_22.md` §5.2.
무엇을 지키나 — **문면은 안 건드리고 전달만 고쳤다**는 것:

  ① `_eff_called` 가 디스패처를 안쪽 이름으로 푼다(겉이름으로 세면 미호출 집합이 항상 가득 찬다)
  ② `_last_user_idx` 가 마지막 손님 발화를 가리킨다(지문의 한 축)
  ③ **T4** 마지막 손님 발화 이후 행동도구를 이미 쳤으면 촉구 자격 없음
  ④ **T4** 새 손님 발화가 오면 자격이 되살아난다(초반 무목표 촉구만 걸러야지 영구 침묵이면 안 된다)
  ⑤ **T3** 지문이 대화 진행으로 **변한다** — 상수 가드는 sim 당 1회 뒤 영영 침묵이었다(감사 행 3)
  ⑥ **T3** 같은 지문은 1회만
  ⑦ **T2** 촉구 슬롯이 `_t2_cp2_pending` 과 **다른 이름**이다(109 중 64 덮임의 원인)
  ⑧ ⛔**T1 미실시** — 문면은 축자 그대로여야 한다(x470 이 D_name 을 자격시키지 못했다·1/24)
  ⑨ 레버 OFF 면 소비 지점이 **무발화**(ctl 바이트 불변)

⛔이 파일은 도메인 어휘를 쓰지 않는다 — 도구 이름은 전부 가짜다([[05]]/[[59]]).
"""
import os
import sys

try:                                    # Windows 콘솔 cp949 — ⛔·★ 가 UnicodeEncodeError 를 낸다
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_gate_patch as G

DEMAND = "Carry out the next step of this request now."


class TC(object):
    def __init__(self, name, args=None, requestor="assistant"):
        self.name, self.arguments, self.requestor = name, (args or {}), requestor


class M(object):
    def __init__(self, role, tool_calls=None):
        self.role, self.tool_calls = role, (tool_calls or [])


ACTS = {"alpha_act", "beta_act", "gamma_act"}


def base():
    """손님 → 에이전트가 read 하나 → 손님 → 에이전트가 alpha_act 를 디스패처로 실행."""
    return [M("user"), M("assistant", [TC("read_thing")]), M("tool"),
            M("user"),
            M("assistant", [TC("call_discoverable_agent_tool",
                               {"agent_tool_name": "alpha_act_7834"})]),
            M("tool"), M("assistant")]


def ok(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    return bool(cond)


def main():
    r = []
    m = base()

    # ① 디스패처 unwrap
    r.append(ok("① _eff_called 가 call_ 디스패처를 안쪽 이름으로 푼다",
                G._eff_called(m) == {"read_thing", "alpha_act"}))
    # ② 마지막 손님 인덱스
    r.append(ok("② _last_user_idx = 마지막 user 위치", G._last_user_idx(m) == 3))
    # ③ T4 — 이 손님 발화 이후 이미 쳤다 → 자격 없음
    r.append(ok("③ T4: 손님 발화 이후 행동도구 발화 있음 → 촉구 자격 없음",
                G._acts_since_last_user(m, ACTS) == {"alpha_act"}))
    # ④ T4 — 새 손님 발화가 오면 되살아난다
    m2 = m + [M("user"), M("assistant")]
    r.append(ok("④ T4: 새 손님 발화 → 자격 회복(영구 침묵 아님)",
                G._acts_since_last_user(m2, ACTS) == set()))
    # ⑤ T3 — 지문이 변한다(상수 가드의 영구 침묵과 반대)
    fp1 = (G._last_user_idx(m), frozenset(ACTS - G._eff_called(m)))
    fp2 = (G._last_user_idx(m2), frozenset(ACTS - G._eff_called(m2)))
    r.append(ok("⑤ T3: 대화가 진행되면 지문이 변한다", fp1 != fp2))
    # ⑥ T3 — 같은 원장이면 같은 지문
    r.append(ok("⑥ T3: 같은 원장 → 같은 지문(같은 지문은 1회)",
                fp2 == (G._last_user_idx(m2), frozenset(ACTS - G._eff_called(m2)))))
    # ⑥b 미호출 집합이 실제로 줄어든다 = 지문의 두 번째 축이 산다
    m3 = m2 + [M("assistant", [TC("beta_act")]), M("tool"), M("assistant")]
    r.append(ok("⑥b T3: 행동을 더 치면 미호출 집합이 줄어 지문이 또 변한다",
                (ACTS - G._eff_called(m3)) == {"gamma_act"}))

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "t2_gate_patch.py"), encoding="utf-8").read()
    # ⑦ T2 — 전용 슬롯이 존재하고 CP2 슬롯과 이름이 다르다
    r.append(ok("⑦ T2: 촉구 전용 슬롯 `_t2_demand_pending` 이 있다",
                "_t2_demand_pending" in src))
    r.append(ok("⑦b T2: 촉구가 더는 `_cp2_assign(..., \"ACT_DEMAND\")` 로 공유 슬롯을 쓰지 않는다",
                '_cp2_assign(self, _dm, "ACT_DEMAND")' not in src))
    r.append(ok("⑦c T2: 촉구가 더는 `_t2_cp2_said` 공유 가드를 오염시키지 않는다",
                "self._t2_cp2_said = _dm" not in src))
    # ⑧ T1 미실시 — 문면 불변
    r.append(ok("⑧ ⛔T1 미실시: 촉구 문면이 축자 그대로다", ('_dm = "%s"' % DEMAND) in src))
    # ⑨ 레버 OFF 면 소비 지점 무발화
    r.append(ok("⑨ 소비 지점이 슬롯 부재를 통과시킨다(OFF = ctl 바이트 불변)",
                'getattr(self, "_t2_demand_pending", None)' in src))
    # ⑩ 관용구가 사본이 아니라 함수다([[67]])
    r.append(ok("⑩ `_effall` 인라인 관용구가 `_eff_called` 로 통일됐다",
                "_effall = _eff_called(state.messages)" in src))

    print("ALL PASS" if all(r) else "SOME FAILED")
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
