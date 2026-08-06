# -*- coding: utf-8 -*-
"""실패했는데 **우리 층이 한 마디도 커밋하지 않은** sim은 몇 개인가.

022 t0에서 관측된 것(2026-08-06): 판정 10행을 받은 직후 이관으로 끝났고, 궤적에 우리 태그가
**0건**이었다. A2가 그 자리에 선언해 둔 후속 사슬은 `resign_th`(사임 턴)로 발화하는데 모델은
사임하지 않고 **행동으로 나갔다**. 같은 구조를 017(갱신)·019(목록)에서도 봤다.

그래서 세는 것은 하나다 — 실패 sim 중 **우리 문구가 궤적에 하나도 없는** 비율. 이 값이 크면
레버군의 발화 창(사임)이 실패 경로(행동)를 못 덮는다는 뜻이고, 처방은 문턱 조정이 아니라
**창의 확장**이다(후보 L).

⚠경계: 궤적에 남는 것은 **커밋된 메시지(deny·표면화)** 뿐이다. reminder 채널은 사이드카에만 남으므로,
사이드카가 있는 sim은 그 값도 함께 찍어 "정말 침묵했는가"를 구분한다([[55]]).

  usage:  x111_silent_failure_census.py [--tag 20260806]
"""

import collections
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x109_task_dossier import load_sims, load_sidecar, sim_key   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TAGRE = re.compile(r"\[([A-Z][A-Z0-9_\- ]{2,40})\]")
TERMINAL = ("transfer_to_human_agents",)


def main():
    sims = load_sims()
    side = load_sidecar()
    silent, spoke = [], []
    term_exit = 0
    tagtot = collections.Counter()
    for s in sims:
        if (s.get("reward_info") or {}).get("reward") == 1.0:
            continue
        tags = collections.Counter()
        ended_by_action = False
        for m in s.get("messages") or []:
            if m.get("role") == "tool":
                for t in TAGRE.findall(str(m.get("content") or "")[:200]):
                    tags[t] += 1
            for tc in (m.get("tool_calls") or []):
                if str(tc.get("name") or "") in TERMINAL:
                    ended_by_action = True
        tagtot.update(tags)
        rec = (s["task_id"], s.get("trial"), len(s.get("messages") or []),
               len(side.get(sim_key(s)) or []), ended_by_action, s["_src"])
        (spoke if tags else silent).append(rec)
        if ended_by_action:
            term_exit += 1

    tot = len(silent) + len(spoke)
    print("== 실패 sim %d개 ==" % tot)
    print("  우리 문구가 궤적에 **0건**인 sim: %d (%.0f%%)" % (len(silent), 100.0 * len(silent) / max(1, tot)))
    # ★arm을 갈라야 한다: 전반부 `*_main_20260806`은 드라이버가 사이드카를 켜지 않았으므로
    #   그 sim의 "사이드카 0건"은 **침묵의 증거가 아니다**([[55]]).
    sc_arms = {r[5] for r in (silent + spoke) if r[3] > 0}
    on = [r for r in silent if r[5] in sc_arms]
    off = [r for r in silent if r[5] not in sc_arms]
    print("  ├ 사이드카가 켜져 있던 arm의 침묵 sim: %d  ⇒ **우리 층이 정말 아무 말도 안 했다**" % len(on))
    print("  └ 사이드카가 없던 arm의 침묵 sim:     %d  ⇒ 커밋된 문구가 없었다는 것만 안다(판정 불가)" % len(off))
    print("  (사이드카가 확인된 arm: %s)" % (", ".join(sorted(sc_arms)) or "없음"))
    print("  이관 호출로 끝난 실패 sim: %d" % term_exit)
    print("\n  침묵 sim 목록(태스크 t시행 · 메시지 · 사이드카 · 이관종료):")
    for t, tr, n, sc, ea, src in sorted(silent)[:40]:
        print("    %-10s t%-2s msgs %3d  사이드카 %3d  %-8s %s" % (t, tr, n, sc, "이관" if ea else "", src))
    print("\n  (참고) 실패 sim 전체의 우리 태그 상위:")
    for k, v in tagtot.most_common(12):
        print("    %-28s %d" % (k, v))


if __name__ == "__main__":
    main()
