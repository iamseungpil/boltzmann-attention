# -*- coding: utf-8 -*-
"""회귀 — FIX-8 소유권 회수 (오프라인·모델 0·x300→출시·2026-08-13).

격리 근거: x300_early_note_probe.py (3셀 n=8) — B_NOTE(소유권 사실) **8/8** ·
A_NONE 0/8 · D_GEN(도구명 없는 일반 촉구) **0/8** ⇒ 인자는 촉구가 아니라 사실.
라이브(t7278 075 turn30)는 도구 미지 주장이라 D_GEN 형 문구가 나갔다.

검정 축: ⑴ 도구 미지 주장의 회수(문턱 2·유일 최대) ⑵ 모호·부족이면 unknown 유지(구판 보존)
⑶ 기존 3분류 거동 불변 ⑷ registry 미전달이면 구판과 동일(거동 보존).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from t2_gate_patch import _split_claims_by_owner, _tok_hits       # noqa: E402

OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


REG = ["open_bank_account_4821", "close_bank_account_7392", "apply_checking_account_credit_5829",
       "freeze_debit_card_3892", "unfreeze_debit_card_3893", "order_debit_card_5739"]
AGENT = ["verify_identity", "log_verification", "get_current_time"]
USER = ["submit_referral", "submit_transaction"]

# ⑴ 라이브 실물: t7278 075 turn30 의 미이행 주장(도구 미지)
c1 = [{"kind": "write", "what": "open Green Fee-Free Account"}]
own, theirs, unk = _split_claims_by_owner(c1, AGENT, USER, registry=REG)
chk("도구 미지 주장 회수", len(own) == 1 and own[0]["tool"] == "open_bank_account_4821"
    and not unk, (own, unk))

# ⑵ 문턱 미달(겹침 1) → unknown 유지
own2, _, unk2 = _split_claims_by_owner(
    [{"kind": "write", "what": "review the account terms"}], AGENT, USER, registry=REG)
chk("겹침 부족=unknown", not own2 and len(unk2) == 1, (own2, unk2))

# ⑵b 동률(freeze/unfreeze 둘 다 2겹) → unknown 유지(엔진은 고르지 않는다)
own3, _, unk3 = _split_claims_by_owner(
    [{"kind": "write", "what": "handle the debit card"}], AGENT, USER, registry=REG)
chk("동률=unknown(선택 안 함)", not own3 and len(unk3) == 1, (own3, unk3))

# ⑶ 기존 3분류 불변
own4, th4, unk4 = _split_claims_by_owner(
    [{"kind": "verify", "what": "verify id", "tool": "verify_identity"},
     {"kind": "give", "what": "referral", "tool": "submit_referral"},
     {"kind": "write", "what": "zzz qqq"}], AGENT, USER, registry=REG)
chk("기존 분류 불변", len(own4) == 1 and len(th4) == 1 and len(unk4) == 1, (own4, th4, unk4))

# ⑷ registry 미전달 = 구판 거동
own5, _, unk5 = _split_claims_by_owner(c1, AGENT, USER)
chk("registry 없으면 구판", not own5 and len(unk5) == 1, (own5, unk5))

# 보조: _tok_hits
chk("_tok_hits 계수", _tok_hits("open Green Fee-Free Account", "open_bank_account_4821") == 2
    and _tok_hits("nothing here", "open_bank_account_4821") == 0)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
