# -*- coding: utf-8 -*-
"""실패한 write 를 성공으로 세지 않는가 (A-7⑵ · 2026-08-23).

`deny_kind` 는 env 거절을 `Error:` 접두로만 알아봤다. 이 환경은 `Failed to …` 로도
거절하고, 그 본문을 성공으로 세면 그 호출이 MATCHED 가 되며 앞선 성공이 DUP 으로
재분류된다 — 079 의 DUP 주장이 그렇게 태어났다.

판정 기준은 [[69]] 다: reward 는 궤적 재실행 후 **DB 해시 비교**이므로 실패한 write 는
상태를 안 바꿔 해시에 안 남는다 ⇒ BLOCKED 가 맞다.

실물 코퍼스로도 잰다 — 이 저장소가 반복해서 진 방식이 *자기 픽스처만 통과하는 술어*라서다.
"""

import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F          # noqa: E402

fail = []


def check(name, ok, detail=""):
    print("  %-54s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


print("① 술어 — 무엇을 거절로 보는가")
check("env `Error:` 는 거절", F.deny_kind("Error: nope")[0] == "env")
check("env `Failed to …` 도 거절", F.deny_kind("Failed to log verification: x")[0] == "env")
check("우리 층 거절은 ours 로 남는다",
      F.deny_kind("[READ-FIRST] fetch it first")[0] == "ours")
check("성공 본문은 거절이 아니다", F.deny_kind('{"ok": true}')[0] == "")
# 접두만 본다 — 본문 **안에** 그 말이 있는 성공 결과를 거절로 접으면 반대 방향 오분류다.
check("본문 중간의 같은 말은 거절이 아니다",
      F.deny_kind('{"note": "Failed to open account is a possible error"}')[0] == "")

print("\n② 실물 — 이 코퍼스에서 그 접두가 실제로 env 실패인가")
pats = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                    "sim_results", "bank_t73*_2026*.results.json.gz")
files = sorted(glob.glob(pats))
if not files:
    print("  · 코퍼스 없음 — 실물 검정 skip")
else:
    shapes, n = set(), 0
    for p in files:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                b = " ".join(str(m.get("content") or "").split()).lstrip()
                if b.startswith("Failed to "):
                    n += 1
                    shapes.add(b.split(":")[0])
    check("`Failed to ` 본문이 실재한다", n > 0, "%d건" % n)
    # 이 접두로 잡히는 것이 전부 실패 서술인지 눈으로 확인 가능한 수의 형상인가.
    check("형상이 소수의 실패 동사구뿐", len(shapes) <= 5, " / ".join(sorted(shapes)))

print("\nRESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
