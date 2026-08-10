#!/bin/bash
# 유료 런 — A3 예치 문턱 보강(C387)과 창-산수 꼬리말 수정(C388)을 라이브에서 본다.
#
# 무엇을 재는가
#   · task_098 (표적·21 sim 전패) — A3 커버리지가 통과 표에서 자격 미달 행을 실제로 빼는가.
#   · task_010 (표적·55 sim 중 4 통과 = 간헐) — 창-산수 문장의 새 꼬리말이 이유 진술로 이어지나.
#   · task_099·task_100 (**회귀**) — 이 둘은 직전 런 `t` 에서 3/3 이었다. A3 를 채우면 그들의
#     통과 표도 바뀔 수 있으므로, 표적만 보고 회귀를 안 보면 이득과 손실을 못 가른다.
#
# 근거
#   · x197 재측정: `B_sum` 0/8 → 8/8 · `A_ver` 8/8 · `D_null` 0/8. `A_iso` 는 0/8 로 남고
#     오답이 `Gold Years` → 카드로 옮겨갔다(잔여 = 타입 축·x201 이 판정 중).
#   · x200: 같은 계산에 꼬리말만 바꿔 이유 진술 0/8 → 3/8(적용본) · 문서가 닿으면 8/8.
#
# ⚠이 런은 **레버 귀속 런이 아니다** — 두 변경이 함께 들어가 있고([[19]] 합성 우선) 스택은
#   정본 go_stack 그대로다([[60]] 전부 켠다). 귀속이 필요하면 별도 arm 을 세운다.
# ⚠태그는 새 것을 쓴다([[30]]: 같은 tag 재런 = 이전 데이터 덮어씀 + 하네스가 resume 를 묻다 죽음).
#
# usage: run_a3fill_20260810.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_098,task_010,task_099,task_100}"
NT="${2:-3}"
TAG="${3:-bank_a3fill_20260810a}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

# 선-점검 ①: 이미 도는 런이 있으면 멈춘다([[30]] 중복 실행·GPU 경합).
if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[run] REFUSING: t2_run_gated 가 이미 돌고 있다. ps 로 확인하고 PID 지정 kill 후 재시도." >&2
  exit 1
fi
# 선-점검 ②: 같은 태그의 산출물이 있으면 멈춘다(덮어쓰기 방지).
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다. 다른 TAG 를 쓰거나 먼저 영속화하라." >&2
  exit 1
fi
# 선-점검 ③: **발사 전 VERIFY OK** (핸드오프 §5 — 지난번 검증 없이 쏴서 구 코드로 유료 런이
#   돌았다). 이 런이 의존하는 두 변경이 라이브 A2 에 실제로 들어 있는지 여기서 확인한다.
/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import sys
sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")
from gate_interpreter import load_domain_a2
import t2_ledger as LG
import t2_factdag as FD
a2 = load_domain_a2("banking_knowledge") or {}
rows = (a2.get("policy_ontology") or {}).get("rows") or []
sp0 = a2["ledger_metrics"][0]
sp1 = next(x for x in a2["ledger_metrics"] if x.get("eligible_text"))
bad = []
dep = FD._a3_map(rows, {"axis": "qualifying_deposit_usd"})
for s, v in (("Gold Years", 1000), ("Light Blue", 500), ("Light Green", 100)):
    if s not in dep or LG._num(dep[s]) != v:
        bad.append("A3 %s=%s (기대 %s)" % (s, dep.get(s), v))
# ★C393 이후 목표 상태는 **꼬리말 없음**이다(측정상 세 형태가 구별되지 않았고 규칙 E 도 어긴다).
if "retrieve it and say which" in (sp0.get("window_history_text") or ""):
    bad.append("창-산수 꼬리말이 아직 붙어 있다")
if "do not state why" not in (sp0.get("status_text") or "").lower():
    bad.append("상태 문구가 계약을 벗어났다")
sp0 = a2["ledger_metrics"][0]
# ★오늘 변경분까지 검사한다 (발사 전 VERIFY 는 **이 런이 의존하는 것 전부**를 봐야 한다)
if "reply NONE" in (sp1.get("rederive_prompt") or ""):
    bad.append("rederive 의 NONE 조항이 아직 있다")
if "The customer says" in (sp1.get("rederive_prompt") or ""):
    bad.append("rederive 에 빈 손님-말 블록이 아직 있다")
if "retrieve it and say which applies" in (sp0.get("window_history_text") or ""):
    bad.append("창-산수 꼬리말이 아직 있다")
if not sp0.get("status_meaning_text") or sp0.get("status_meaning_axis") != "status_meaning":
    bad.append("상태 정의 선언이 없다")
_sm = [r for r in rows if r.get("axis") == "status_meaning"]
if len(_sm) < 6:
    bad.append("A3 상태 정의가 %d행뿐이다" % len(_sm))
if "REJECTED" not in LG.status_meanings_text([{"referral_status": "REJECTED"}], sp0, rows):
    bad.append("상태 정의 전달이 침묵한다")
if not sp0.get("diagnose_prompt") or not sp0.get("diagnosed_text"):
    bad.append("격리 진단 선언이 없다")
_lr = [{"date": "10/20/2025", "referred_account_type": "A", "referral_status": "COMPLETE"},
       {"date": "10/22/2025", "referred_account_type": "B", "referral_status": "COMPLETE"},
       {"date": "10/25/2025", "referred_account_type": "C", "referral_status": "REJECTED"}]
_blk = LG.onto_context(_lr, sp0, rows)
for _need in ("grouped by the status", "Date arithmetic", "policy document that defines"):
    if _need not in _blk:
        bad.append("온톨로지 문맥에 '%s' 조각이 없다" % _need)
cfg = sp1["eligible"]
if cfg.get("kind_field") != "kind" or "{kinds}" not in (cfg.get("kind_prompt") or ""):
    bad.append("종류 선택 선언이 없다")
kb = LG.subject_kinds(rows, cfg.get("kind_field") or "kind")
if len(kb) < 30:
    bad.append("A3 종류 태깅이 %d 개뿐이다" % len(kb))
maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in (cfg["show_axes"])}
tbl = LG.eligible_text(400, {}, maps, sp1, {"qualifying_deposit_usd": 600}) or ""
named = {l.strip().split(":")[0].strip() for l in tbl.splitlines() if l.startswith("  ")}
if "Gold Years" in named or "Blue" not in named:
    bad.append("098 케이스 표가 기대와 다르다: %s" % sorted(named))
kept, _d = LG.restrict_to_kind(maps, kb, "checking_accounts")
t2 = LG.eligible_text(400, {}, kept, sp1, {"qualifying_deposit_usd": 600}) or ""
n2 = {l.strip().split(":")[0].strip() for l in t2.splitlines() if l.startswith("  ")}
if any(kb.get(s) in ("credit_cards", "business_credit_cards") for s in n2) or "Blue" not in n2:
    bad.append("종류로 거른 표가 기대와 다르다: %s" % sorted(n2))
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  t2_launch $TAG 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 10
echo "--- 발사 직후 로그 ---"
head -20 "$LOG/${TAG}.log" 2>/dev/null || true
echo
echo "launched · tasks=$TASKS nt=$NT · log: $LOG/${TAG}.log"
echo "  sidecar: $LOG/fb_${TAG}.jsonl · trace: $LOG/trace_${TAG}.jsonl"
