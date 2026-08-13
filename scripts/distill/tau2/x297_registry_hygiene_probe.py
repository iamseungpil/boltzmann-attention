# -*- coding: utf-8 -*-
r"""x297 — FIX-3 목록 위생 격리: 45개 전-도메인 목록 ↔ 토큰-겹침 필터 목록 (075 turn30 실물).

배경(t7277 075 fb 사이드카·사용자 지적): 모델 초안은 도구 경로 시도(unlock 'open_account'
날조 맨이름)였는데 FIX-3 deny 가 **45개 전 도메인 레지스트리**를 동봉 → 수동-안내 접힘으로
regen. x287b 실측: 8개 목록 = 8/8 · 31개 = 5/8 (목록 위생=효과 변수) — 45개는 미측정 문면.
선례 = x287b(근거확인 deny 가 "수동 조정" 접힘을 A0/8→B8/8 돌파).

셀 3 (n=8·t7277 075 turn30 직전 컷 = 날조 unlock 초안이 나온 assistant 턴 직전·
      deny 문면 리터럴 = 각 팔 축자):
  A_FULL  현행 45개 전체 목록 deny(라이브 축자) — 재현 대조
  B_TOK   토큰-겹침 필터 목록 deny(날조 이름 'open_account'의 토큰 {open,account} 중
          'open' 겹침 = open_bank_account_4821 하나·기계 필터·판단 0) — 출시 후보 축자
  D_BARE  목록 없는 부재-단정만("there is none" 류) — x287b A 재현(부정통제)

계기: 다음 어시스턴트 턴이 open_bank_account_4821 을 unlock/call 하는가.
판정(사전 고정): A_FULL ≤2/8 ∧ B_TOK ≥6/8 → FIX-3' 토큰-필터 출시(문면 재측정 완료).
  A_FULL ≥6/8 → 45개도 작동 = 접힘은 딴 원인 → L1 재수사. D_BARE ≥3/8 → 계기 오염.
  중간(3~5) → n=16 재측정 1회.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x297_registry_hygiene_probe.py [N]
"""
import collections
import json
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402

TAG = "bank_t7277_b_20260813y"
TASK = "task_075"
TARGET = "open_bank_account_4821"
FAB = "open_account"

REGISTRY = [
    "activate_debit_card_8291", "activate_debit_card_8292", "activate_debit_card_8293",
    "apply_checking_account_credit_5829", "apply_credit_card_account_flag_6147",
    "apply_savings_account_credit_6831", "apply_statement_credit_8472",
    "approve_credit_limit_increase_5847", "change_debit_card_pin_6285",
    "clear_debit_card_fraud_alert_4892", "close_bank_account_7392",
    "close_credit_card_account_7834", "close_debit_card_4721",
    "deny_credit_limit_increase_5848", "emergency_credit_bureau_incident_transfer_1114",
    "example_agent_tool_0000", "file_credit_card_transaction_dispute_4829",
    "file_debit_card_transaction_dispute_6281", "freeze_debit_card_3892",
    "get_all_user_accounts_by_user_id_3847", "get_atm_deposit_images_8473",
    "get_bank_account_transactions_9173", "get_closure_reason_history_8293",
    "get_credit_limit_increase_history_4829", "get_debit_cards_by_account_id_7823",
    "get_debit_dispute_status_7483", "get_payment_history_6183",
    "get_pending_replacement_orders_5765", "get_user_dispute_history_7291",
    "initial_transfer_to_human_agent_0218", "initial_transfer_to_human_agent_1822",
    "log_credit_card_closure_reason_4521", "open_bank_account_4821",
    "order_debit_card_5739", "order_replacement_credit_card_7291",
    "pay_credit_card_from_checking_9182", "request_temporary_debit_card_limit_increase_8374",
    "reset_debit_card_pin_6284", "set_debit_card_recurring_block_7382",
    "submit_credit_limit_increase_request_7392", "submit_interest_discrepancy_report_7294",
    "transfer_funds_between_bank_accounts_7291", "unfreeze_debit_card_3893",
    "update_transaction_rewards_3847",
]


def tok_filter(fab, names):
    """날조 이름 토큰과 겹치는 레지스트리 이름(기계 필터·판단 0·접미 숫자 제거 후 토큰)."""
    import re
    toks = [t for t in re.split(r"[_\W]+", fab.lower()) if t and t not in ("get", "the")]
    out = []
    for n in names:
        base = re.sub(r"_\d+$", "", n).lower()
        if any(t in base.split("_") for t in toks):
            # 가장 강한 겹침 우선: 전 토큰 포함 이름이 있으면 그것만
            out.append((sum(t in base.split("_") for t in toks), n))
    if not out:
        return []
    mx = max(o[0] for o in out)
    return [n for c, n in out if c == mx]


DENY_HEAD = ("Error: '%s' is not one of the discoverable tools in this domain, so it cannot "
             "be unlocked - unlock_discoverable_agent_tool does not apply to it. If it is a "
             "tool you already have, call it directly. Do not search the knowledge base for a "
             "suffixed version of this name: there is none." % FAB)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    import x238_action_forensic as X
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and s.get("reward_info") is not None]
    sim = sims[0]
    msgs = sim["messages"]
    # 컷 = log_verification 성공 직후(turn30 초안이 나온 지점)
    cut = None
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and "Verification logged successfully" in str(
                m.get("content") or ""):
            cut = i + 1
            break
    if cut is None:
        print("컷 없음")
        return
    filt = tok_filter(FAB, REGISTRY)
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    base = B.render(msgs[:cut], ours)
    base = base[:base.rfind("\n[user] ")]
    draft = ("[assistant calls] unlock_discoverable_agent_tool"
             "({'agent_tool_name': '%s'})" % FAB)
    arms = (
        ("A_FULL", DENY_HEAD + " The discoverable tools that DO exist in this domain are: "
         + ", ".join(REGISTRY) + ". If one of these performs the action, unlock and call "
         "that exact name."),
        ("B_TOK", DENY_HEAD + " The discoverable tool name(s) on file that overlap what you "
         "asked for: " + ", ".join(filt) + ". If one of these performs the action, unlock "
         "and call that exact name."),
        ("D_BARE", DENY_HEAD),
    )
    print("075 t%s cut=%d · 필터 목록=%s · n=%d · URL=%s\n" % (
        sim.get("trial"), cut, filt, n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    for label, deny in arms:
        body = base + "\n" + draft + "\n[tool] " + deny
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 400)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            blob = " ".join(str(tc) for tc in (r.get("tool_calls") or []))
            if TARGET in blob:
                hit += 1
            else:
                first = ""
                for tc in (r.get("tool_calls") or []):
                    first = str(tc.get("name") or (tc.get("function") or {}).get("name") or "")
                    a = str(tc.get("arguments") or "")[:40]
                    first += ":" + a
                    break
                cnt[first or ("(text)" if r.get("content") else "(empty)")] += 1
        print("%-7s target-unlock %d/%d · 기타 %s" % (label, hit, n, dict(cnt)))
    print("\n※ 판정(사전 고정): A_FULL ≤2/8 ∧ B_TOK ≥6/8 → FIX-3' 출시. A_FULL ≥6/8 → 45개도"
          " 작동(접힘 딴 원인·L1 재수사). D_BARE ≥3/8 → 계기 오염. 중간(3~5) → n=16 1회.")


if __name__ == "__main__":
    main()
