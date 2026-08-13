# -*- coding: utf-8 -*-
"""회귀 — FIX-7 소유권-정정 deny (오프라인·모델 0·x298→출시·2026-08-13).

검정 축: ⑴ `_tok_overlap` 기계 필터(최대 겹침만·동률 전부·무겹침=침묵)
⑵ A2 키 3사본 동일 + 문면이 x298 B_OWN 축자와 일치(측정한 문구 = 출시 문구·[[03b]])
⑶ 채널 인자 선언(user_tool_channel_args) ⑷ 렌더 결과에 매치 이름·소유권 사실 포함.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from t2_gate_patch import _tok_overlap                            # noqa: E402
import x298_ownership_deny_probe as X298                          # noqa: E402

OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


REG = ["open_bank_account_4821", "close_bank_account_7392", "transfer_funds_between_bank_accounts_7291",
       "apply_checking_account_credit_5829", "get_all_user_accounts_by_user_id_3847",
       "freeze_debit_card_3892", "order_debit_card_5739"]

# ⑴ 토큰 필터
chk("단일 최대겹침", _tok_overlap("open_account", REG) == ["open_bank_account_4821"],
    _tok_overlap("open_account", REG))
chk("동률 전부 유지", _tok_overlap("debit_card", REG) == ["freeze_debit_card_3892",
                                                       "order_debit_card_5739"],
    _tok_overlap("debit_card", REG))
chk("무겹침=침묵", _tok_overlap("teleport_customer", REG) == [], _tok_overlap("teleport_customer", REG))
chk("빈 이름=침묵", _tok_overlap("", REG) == [] and _tok_overlap("x", []) == [])

# ⑵⑶ A2 3사본
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]
blocks = [json.load(io.open(os.path.join(HERE, p), encoding="utf-8"))
          .get("discoverable_name_check") or {} for p in PATHS]
fbs = {b.get("feedback_user_tool_is_agents") for b in blocks}
args = {tuple(b.get("user_tool_channel_args") or ()) for b in blocks}
chk("3사본 문면 동일", len(fbs) == 1 and None not in fbs)
chk("3사본 채널인자 동일", len(args) == 1 and args != {()}, args)
chk("채널 인자 = give/user 계열", set(list(args)[0]) == {"discoverable_tool_name", "user_tool_name"},
    list(args)[0])

# 측정 축자 대조: 프로브 b_own() 이 만든 문자열과 A2 템플릿 렌더가 같아야 한다
tpl = list(fbs)[0]
rendered = tpl.replace("{name}", X298.FAB).replace("{matches}", "open_bank_account_4821")
chk("x298 B_OWN 축자 일치", rendered == X298.b_own(["open_bank_account_4821"]),
    rendered[:90])

# ⑷ 렌더 내용
chk("소유권 사실 포함", "YOUR OWN agent tools" in rendered
    and "open_bank_account_4821" in rendered and "call it yourself" in rendered)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
