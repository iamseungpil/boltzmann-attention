# -*- coding: utf-8 -*-
r"""x304 — U2 문면 측정: x303 능력-후보 컷에서 기존 STEP2 문면+옳은 이름이 여는가.

선행(x303·같은 컷 34·같은 도구):
  A_MIN  0/8 — 전부 `get_user_information_by_name`(인적사항 재조회 어트랙터)
  B_FULL 0/8 — 전부 텍스트(접힘)
사전등록 경로: A≤2∧B≤2 → **U2 문면 측정이 다음**([[64]] 거부는 고칠 방법을 담아라 — 단
측정 후 출시·[[62]]).

라이브 사실(087): STEP2 는 발화**했지만** formalize 가 다른 이름(get_debit_cards·unfreeze)을
골랐다 — 옳은 이름(get_all_user_accounts_by_user_id_3847)은 msg31 KB 로 회수됐는데도.
그래서 질문은 둘로 갈린다:
  ⓐ 문면이 옳은 이름과 함께 닿으면 열리는가? (열리면 잔여 = **이름 선택**·엔진 수사)
  ⓑ 안 열리면 = 접힘 국면(x299 동형·시점 인자) — 문면으론 못 열고 배치([[65]] 서브) 검토.

셀 3 (n=8·컷 = x303 과 동일·문면 전부 출시본 축자[[03b]]):
  B_STEP2   base + DISCOVERY_STEP2_FB 축자(name=TARGET·unlock=실명)   ← 출시 문면 그대로
  C_OWN     base + x300 B_NOTE 동형 소유권 사실 문면(FIX-7 계열 축자)
  D_NEG     base + STEP2 축자(name=**무관 read 도구**)                 ← 이름-특이성 부정통제

성공 = tool_calls 에 unlock/call × TARGET. D_NEG 는 그 무관 이름을 따라가면 문면-순응
(이름이 인자)·안 따라가면 요청-정합 판단이 살아있음을 가른다.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x304_account_fix_wording.py [N]
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
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402
import x303_account_reach_iso as Z                                # noqa: E402
from t2_resolve import DISCOVERY_STEP2_FB                         # noqa: E402

TARGET = Z.TARGET
UNLOCK = "unlock_discoverable_agent_tool"
NEG = "get_debit_dispute_status_7483"        # 회수된 적 있는 무관 read (087 레지스트리 실재)
# x300 B_NOTE 축자 골격(측정본·[[03b]]) — 행동 구만 이 컷의 사실로 치환
OWN_T = ("Note: [CLAIM-PROVENANCE] tool ownership — looking up the customer's bank accounts is "
         "done by a tool in YOUR OWN tool list, not the customer's: %s. The customer cannot "
         "run it, and describing the steps to them does not execute it.")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in X.load(Z.TAG) if s["task_id"] == Z.TASK
               and s.get("reward_info") is not None)
    cut = Z.cut_of(sim)
    tools = U.tools_of(sim)
    P.TAG = Z.TAG
    ours = P.our_lines(sim)
    base = B.render(sim["messages"][:cut], ours)
    step2 = DISCOVERY_STEP2_FB.format(name=TARGET, unlock=UNLOCK)
    step2_neg = DISCOVERY_STEP2_FB.format(name=NEG, unlock=UNLOCK)
    own = OWN_T % TARGET
    print("x304 cut=%d · target=%s · n=%d · URL=%s\n" % (
        cut, TARGET, n, os.environ.get("T2_PROBE_URL", "8140(기본⚠)")))
    for label, note in (("B_STEP2", step2), ("C_OWN", own), ("D_NEG", step2_neg)):
        hit = neg = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(base + "\n[system] " + note, tools, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            k = Z.classify(r)
            blob = " ".join(str(t) for t in (r.get("tool_calls") or []))
            hit += k == "target"
            neg += NEG in blob
            cnt[k] += 1
        print("%-8s target %d/%d · 무관이름추종 %d · %s" % (label, hit, n, neg, dict(cnt)))
    print("\n※ 판정(사전 고정): B≥6 → 잔여=이름 선택(엔진 formalize/후보 수사·문면은 살아있음)."
          " B≤2 ∧ C≥6 → 소유권-형이 인자(FIX-7 계열 확장 검토). 둘 다 ≤2 → 접힘 국면 —"
          " 문면 아닌 배치(서브·[[65]]) 검토. D_NEG 추종 ≥6 = 이름이 인자(문면-순응)·"
          " ≤2 = 정합 판단 생존.")


if __name__ == "__main__":
    main()
