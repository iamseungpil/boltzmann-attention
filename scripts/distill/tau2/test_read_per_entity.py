#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`T2_READ_PER_ENTITY` 래칫 — 실제 궤적으로 초 단위 검정.

## 무엇을 잠그나 (측정 정본 = `x560_read_entity_gap_scan.py`)

선행 read 요건의 충족 판정이 **도구 이름만** 보면, 다른 주체로 돈 read 가 요건을 영구 충족시킨다.
016 실측(t7363·t7356 두 세대 동일): 계좌 read 는 손님 자신(`86e92f639e`)으로만 돌았고 친구
(`friend_user_5839`)로는 끝내 안 돌았는데 요건은 닫혀 있었다.

## 여기서 잠그는 다섯 (전부 나를 이미 한 번씩 물었다)

⑴ 016 두 세대에서 gap 이 **정확히 친구 하나**다 — 손님 자신은 안 잡힌다.
⑵ **디스패처 페이로드**를 편다. banking 호출은 `call_discoverable_agent_tool{…, arguments:"<JSON>"}`
   라 최상위만 보면 주체가 안 보인다(발화 0/33 이 그 증상이었다·085 에서 같은 사고 선례).
⑶ **선택자는 주체가 아니다**. 래퍼의 `agent_tool_name` 을 주체로 세면 발화면이 79% 로 부푼다.
⑷ read 가 **한 번도 안 돌았으면 빈 dict** — 구판 규칙과 겹치지 않는다(폭발 반경 한정).
⑸ 플래그 OFF 면 `requirements_for` 에 `@` 요건이 **하나도** 안 생긴다.

## ⛔여기서 판정하지 않는 것
*"요구하면 모델이 그 read 를 부르는가"* 는 안 잰다 — 격리나 런이 잰다([[62]]·[[69]]).

실행: PYTHONIOENCODING=utf-8 py -3 test_read_per_entity.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_dominance as DOM                                          # noqa: E402
import t2_forensic as F                                             # noqa: E402
import x560_read_entity_gap_scan as X560                            # noqa: E402

READ = "get_all_user_accounts_by_user_id"
# ★경계까지 잠근다. 술어는 **모델이 그 주체를 인자에 넣은** sim 에서만 발화한다 — 발화는
#   NL 이 아니라 호출에서 온다([[59]]). 016 다섯 sim 중 둘이 그 형상이고, 나머지 셋에서
#   친구 id 는 대화에만 있고 어떤 호출에도 안 들어간다 ⇒ 그 자리는 이 레버가 못 산다.
CASES = (("bank_t7363_hard0_20260827", "task_016#s626729", {"friend_user_5839": "user_id"}),
         ("bank_t7356_grpB3_20260826", "task_016#s373753", {"friend_user_5839": "user_id"}),
         ("bank_t7356_grpB3_20260826", "task_016#s626729", {}),
         ("bank_t7363_hard0_20260827", "task_016#s373753", {}))
FAIL = []


def chk(c, ok, extra=""):
    print(("  OK   " if ok else "  FAIL ") + c + (("  — " + extra) if extra else ""))
    if not ok:
        FAIL.append(c)


def msgs(tag, simtag):
    s = [x for x in F.sims(tag) if F.simtag(x) == simtag]
    return [X560._M(m) for m in (s[0].get("messages") or ())] if s else []


print("## ⑴ 016 — 발화하는 자리와 **안 하는 자리**")
for tag, st, want in CASES:
    ms = msgs(tag, st)
    g = DOM.read_entity_gap(ms, READ, X560._unwrap)
    chk("%s %s → %s" % (tag[10:20], st.split("#")[-1], "친구 하나" if want else "침묵"),
        g == want, repr(g))

print()
print("## ⑵ 디스패처 페이로드를 편다 · ⑶ 선택자는 주체가 아니다")
ms = msgs(CASES[0][0], CASES[0][1])
wrapped = [tc for m in ms for tc in m.tool_calls
           if str(tc.name).startswith("call_") and X560._unwrap(tc).startswith(READ)]
chk("016 에 래퍼 호출이 있다", bool(wrapped), "%d건" % len(wrapped))
if wrapped:
    a = DOM._args_dict(wrapped[0])
    chk("페이로드가 펴진다 (user_id 가 보인다)", "user_id" in a, repr(a))
    chk("선택자는 안 들어온다 (agent_tool_name 없음)", "agent_tool_name" not in a, repr(a))

print()
print("## ⑷ 안 돈 read 는 빈 dict")
chk("돌지 않은 read → {}", DOM.read_entity_gap(ms, "get_user_dispute_history", X560._unwrap) == {})

print()
print("## ⑸ 플래그 OFF 면 `@` 요건 0")
os.environ.pop("T2_READ_PER_ENTITY", None)
import gate_interpreter as GI                                       # noqa: E402
a2 = GI.load_domain_a2("banking_knowledge") or {}
off = DOM.requirements_for(a2, ms, "submit_referral", unwrap=X560._unwrap)
chk("OFF: @ 요건 없음", not [r for r in off if "@" in str(r.get("id"))],
    str([r.get("id") for r in off]))
os.environ["T2_READ_PER_ENTITY"] = "1"
on = DOM.requirements_for(a2, ms, "submit_referral", unwrap=X560._unwrap)
chk("ON: 친구를 지목하는 요건이 선다",
    any("friend_user_5839" in str(r.get("id")) for r in on),
    str([r.get("id") for r in on]))
chk("ON 요건이 무엇을 하면 풀리는지 말한다([[64]])",
    all(r.get("satisfiers") for r in on if "@" in str(r.get("id"))))
os.environ.pop("T2_READ_PER_ENTITY", None)

print()
print("## ⑹ 탐지기의 일반성 — 016 밖의 세 태스크 (2026-08-27 추가)")
# ★왜 이 셋인가: 플래그를 **끈 뒤에도** 이 술어는 진단으로 남는다. 잡히는 값은 전부
#   §12 축(*"배달된 값을 엉뚱한 엔티티에 묶는다"*)이고, 그것이 이 검정이 잠그는 사실이다.
#   ⛔여기서 잠그는 것은 *"그 read 를 요구해야 한다"* 가 **아니다** — 요구는 t7364 가 음성으로
#     판정했다(그 값들은 그 read 의 주체가 아니다). 잠그는 것은 **탐지**뿐이다.
DET = (("bank_t7363_hard0_20260827", "task_074", "get_bank_account_transactions",
        "Dark Green Account"),
       ("bank_t7356_grpA3_20260826", "task_072", "get_bank_account_transactions",
        "Bluest Account"),
       ("bank_t7363_hard0_20260827", "task_085", "get_bank_account_transactions",
        "f7d3a82c91"))
for tag, tid, read, want in DET:
    got = {}
    for sm in F.scored(tag):
        if F.task_id(sm) != tid:
            continue
        g = DOM.read_entity_gap([X560._M(m) for m in (sm.get("messages") or ())],
                                read, X560._unwrap)
        if g:
            got = g
            break
    chk("%s %s 에서 %s 를 잡는다" % (tid, read.split("_by_")[0][-12:], want[:18]),
        want in got, repr(sorted(got))[:110])

print()
print("## 부호표 (x560 · 두 세대 33 sim)")
rows = []
for tag in ("bank_t7363_hard0_20260827", "bank_t7356_grpB3_20260826", "bank_t7356_grpA1_20260826",
            "bank_t7356_grpA2_20260826", "bank_t7356_grpA3_20260826", "bank_t7356_grpA4_20260826"):
    for s in F.scored(tag):
        m2 = [X560._M(x) for x in (s.get("messages") or ())]
        hit = any(DOM.read_entity_gap(m2, r, X560._unwrap)
                  for r in ("get_all_user_accounts_by_user_id", "get_bank_account_transactions"))
        rows.append(((s.get("reward_info") or {}).get("reward"), hit))
fired = [r for r, h in rows if h]
chk("발화 sim 이 있다", bool(fired), "%d/%d" % (len(fired), len(rows)))
chk("발화한 sim 은 **전부 reward 0** (손실 불가)", all((r or 0) < 1.0 for r in fired), str(fired))

print()
print("결과: %s" % ("모두 통과" if not FAIL else "실패 %d — %s" % (len(FAIL), FAIL)))
sys.exit(1 if FAIL else 0)
