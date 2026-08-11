#!/bin/bash
# 유료 런 — **검색 에이전트**(`T2_SEARCH_AGENT`)를 070·071 에서 라이브로 잰다. **8141/GPU1**.
#
# 왜 (사용자 지시 2026-08-11: *"8141 로 gpu 1에서 띄워라"*)
#   8140/GPU0 은 **다른 세션의 010·099 자리**다. 자리를 비켜 잡는다([[30]] 포트 분리).
#
# 무엇이 켜지나 (직전 상태와의 차이 = 검색 에이전트 하나)
#   기본 스택(go_stack) + `T2_DECIDE_ANY=1`(훅이 그 안에 산다) + **`T2_SEARCH_AGENT=1`**
#   ⚠`T2_KEEP_DENY_BODY` 는 **끈다** — R9 는 010/099 축에서 판정 중이고(C415), 여기에 섞으면
#     070/071 의 델타가 두 레버의 합이 된다.
#
# 근거 (라이브 전에 이미 잰 것)
#   x235 사다리: `R4_EXPIRED` 0/8 ↔ `R3_PROMO` 8/8 — 만료를 말로 알려 주면 실패한다.
#   x243: 문서 전문/앞400자 + 활성 고지 **8/8**, 고지 없으면 **0/8** ⇒ 축 선별 불요.
#   x248·x252(프로덕션 경로·n=8): **두 축 8/8** — checking `Sky Blue` · savings `Gold Saver
#   Account`. 부정 통제: 문서만 checking **0/8** · 만료 안 빼면 savings **0/8**.
#
# 사전 등록 (보기 전에 적는다)
#   P0 팔 오염   `[T2_SEARCH_AGENT] group=… 문서 N(뺀 것 M)` 이 찍히는가 · `[T2_DOCGROUP]` ·
#                `[T2_DOCDECIDE]`. **뺀 것에 014/016 이 들어 있는가** — 엔진의 유일한 일이다.
#   P1 성적      070·071 각 3 sim. 기준선 = `bank_m3_20260810s` **0/2**(스모크·무발화).
#   P2 표적 칸   gold `open_bank_account_4821` 의 `account_class` (`Sky Blue` / `Gold Saver
#                Account`). **성적보다 이 칸을 먼저** 읽는다.
#   P3 재료 도달 사이드카에 `decided_by_docs_text` 축자가 실렸는가(우리 층이 실제로 말했는가).
#   P4 Δspurious 게이트 거부 수 · gold 밖 쓰기 호출 (재료가 늘면 over-action 이 늘 수 있다·§1.3).
#
# ⚠3 sim×2 태스크는 총점을 못 가른다(C403·C406·C408·C415). 쓸 수 있는 것은 P0·P2·P3 의 계수다.
# ⚠태그는 새 것을 쓴다([[30]]).
#
# usage: run_search071_20260811.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_070,task_071}"
NT="${2:-3}"
TAG="${3:-bank_sa_20260811}"
PORT=8141
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

# ★가드는 **포트별**이다. 구판은 `t2_run_gated` 가 하나라도 돌면 거절했는데, 그러면 다른 세션이
#   8140 에서 도는 동안 8141 을 못 쓴다. GPU 가 다르므로 경합이 아니다 — 같은 포트만 막는다.
if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}"; then
  echo "[run] REFUSING: 포트 ${PORT} 에서 이미 돌고 있다." >&2; exit 1
fi
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다. 다른 TAG 를 쓰라." >&2; exit 1
fi

# 선-점검: 이 런이 의존하는 것이 **실제로 코드·A2 에 있는가** (발사 전 VERIFY OK)
/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import json, os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, d)
bad = []
src = open(os.path.join(d, "t2_gate_patch.py"), encoding="utf-8").read()
if 'os.environ.get("T2_SEARCH_AGENT") == "1"' not in src or "def _search_material(" not in src:
    bad.append("검색 에이전트 배선이 없다")
po = json.load(open(os.path.join(d, "a2", "banking_knowledge.gate.json"),
                    encoding="utf-8"))["policy_ontology"]
for k in ("doc_index", "doc_windows", "group_prompt", "doc_decide_prompt",
          "decided_by_docs_text"):
    if not po.get(k):
        bad.append("A2/A3 에 %s 가 없다" % k)
# 이 런의 표적 = 만료 고지 둘이 실제로 선언돼 있는가
w = {r["doc"] for r in (po.get("doc_windows") or [])}
for doc in ("doc_bank_accounts_bank_accounts_(general)_014",
            "doc_bank_accounts_bank_accounts_(general)_016"):
    if doc not in w:
        bad.append("유효창에 %s 가 없다(뺄 대상이 없으면 잴 것도 없다)" % doc)
for t in ("test_search_agent_wiring.py", "test_search_agent.py", "test_a2_three_layer.py"):
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s 실패: %s" % (t, (r.stdout or "")[-200:]))
if os.environ.get("T2_SEARCH_AGENT") == "1":
    bad.append("검증 프로세스에 플래그가 켜져 있다(런처가 켜야 한다)")
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_DECIDE_ANY=1 T2_SEARCH_AGENT=1 && unset T2_KEEP_DENY_BODY && \
  t2_launch $TAG $PORT '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 12
echo "--- 발사 직후 ---"
head -12 "$LOG/${TAG}.log" 2>/dev/null || true
echo "launched · tasks=$TASKS nt=$NT · port=$PORT(GPU1) · log: $LOG/${TAG}.log"
echo "  sidecar: $LOG/fb_${TAG}.jsonl · trace: $LOG/trace_${TAG}.jsonl"
