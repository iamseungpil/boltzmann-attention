# -*- coding: utf-8 -*-
"""로그↔결과 조인 회귀 — **같은 함정에 세 번 걸리지 않게**(2026-08-16·C491⒠ 재발).

사고: 로그의 `[sim=...]` 태그는 `s<seed>` 인데 `sim_key()` 는 **trial 우선**이다. 그것으로 조인하면
전부 미스하고, 빈 결과를 *"레버 미발화"* 로 오독한다. 2026-08-15 에 한 번, 2026-08-16 배달↔선택
정합에서 또 한 번 걸렸다.

불변식:
  ① `simtag()` 가 실제 로그 태그와 **바이트 동일**
  ② `by_sim()` 이 그 태그로 **실제 매치를 돌려준다**(0 이 아니다)
  ③ `sim_key()` 로는 조인이 **안 된다**는 사실 자체를 검정이 붙잡는다(회귀 방지)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import t2_forensic as F  # noqa: E402

TAG = "bank_t7299_ctl_20260816b"      # 이 런은 tracked 라 어느 체크아웃에서도 있다
OK = True


def chk(cond, msg):
    global OK
    OK = OK and bool(cond)
    print("  %s %s" % ("✓" if cond else "✗", msg))


sims = F.sims(TAG)
txt = F.log_text(TAG)
tags_in_log = set(re.findall(r"\[sim=(task_\d+#\w+)\]", txt))

print("[①] simtag == 로그 태그")
chk(bool(sims) and bool(tags_in_log), "sim %d개 · 로그 태그 %d종" % (len(sims), len(tags_in_log)))
mine = {F.simtag(s) for s in sims}
chk(mine <= tags_in_log, "simtag 전부가 로그에 실재 (%d/%d)" % (len(mine & tags_in_log), len(mine)))

print("[②] by_sim 이 실제 매치를 돌려준다")
hits = F.by_sim(TAG, r"(SEARCH_AGENT\] group=\w+)", sims)
chk(sum(len(v) for v in hits.values()) > 0,
    "배달 이벤트 %d건 · sim %d개" % (sum(len(v) for v in hits.values()), len(hits)))

print("[③] sim_key 로는 조인이 안 된다(그래서 simtag 가 필요하다)")
keys = {F.sim_key(s) for s in sims}
chk(not (keys & tags_in_log), "sim_key 집합은 로그 태그와 교집합 0 — 조인 금지 근거")

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
