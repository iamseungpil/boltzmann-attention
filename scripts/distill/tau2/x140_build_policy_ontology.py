# -*- coding: utf-8 -*-
"""x140 — **A3 정책 온톨로지 생성기**: 문서 전수를 한 번 읽고 관계로 굳힌다 (유료 0·로컬 vllm).

정본 = `A3_POLICY_ONTOLOGY_DESIGN_2026_08_08.md`.
§0z 정의: *"scaffold·결정론이 비용과 완결성을 위해 미리 formalize한 정책 규칙을,
각 도구·결정기에서 찾기 쉽게 형식화한 관계"*.

이 도구가 지키는 것:
  · **문서 전수**(698개) — 완결성의 근거는 *모집단을 안다*는 것이다(§0a). 표본을 뽑지 않는다.
  · **문서 하나씩**(`split`) — C313 실측: 덩어리로 물으면 12~44%, 항목별이면 92~100%.
  · **인용 실재 검증** — 모델이 낸 인용문이 **그 문서에** 실제로 있어야 채택한다([[22]] 근거-우선).
  · **축은 닫힌 집합** — 선언 밖 축은 버린다(§2a op 동결과 같은 규율).
  · **증분 저장** — 중단돼도 이어서 한다(오늘 캐시를 끝에서만 저장해 25분을 날릴 뻔했다).
  · **충돌은 인쇄** — 같은 (주어, 축)에 다른 값이면 **양쪽 인용과 함께** 남긴다([[25]]).

⚠이 파일은 **빌드 도구**이지 런타임 엔진이 아니다. 프롬프트에 도메인 어휘가 있는 것은 여기까지
허용된다([[59]]는 엔진을 규율한다) — 산출물만 A3로 가고, 이 스크립트는 런타임에 안 돈다.
⚠v0의 축은 **둘**(관계기간 문턱·연간 상한)이다. 늘리는 절차는 설계서 §9-2.

★**v1 (2026-08-08 · C317 반영)** — v0 독립 검사가 낸 결함 둘을 여기서 닫는다:
  ⓐ **축별 질의**(`--per-axis`) — v0는 한 호출에 두 축을 물었고, **두 축을 다 말하는 문서 13개 중
     5개서 한 축을 흘렸다**(38%). C313의 *"덩어리로 묻지 마라"* 가 축 방향으로도 산다.
  ⓑ **축 적합성 형식 검사**(`--axis-fit`·`--audit`) — v0에 *"rolling 9-day window"* 문장이
     **연간 상한** 행으로 들어갔다. 인용이 실재해도 **축이 틀릴 수 있다**([[22]] 근거-우선의
     필요조건일 뿐). 검사는 **형식적인 것 둘뿐**이다(의미 판단은 LLM 몫·[[52]]):
       B1 값이 인용에 축자로 나온다 · B2 인용에 그 축의 표지가 있다.
     ⚠게이트 자신도 역효과가 있다(등대 §1.3) ⇒ **거절을 이유별로 세어 인쇄**하고,
     `--audit`로 **기존 산출물을 먼저 통과시켜** 과잉 거절을 잰다.

usage: x140_build_policy_ontology.py --docs <dir> --out ontology.json \
         [--per-axis] [--base http://localhost:8140/v1] [--limit 20] [--save-every 20]
       x140_build_policy_ontology.py --audit <ontology.json[.gz]>   # 축 적합성 관문만
"""

import argparse
import collections
import glob
import gzip
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 축 정의 — **닫힌 집합**. 각 축은 (이름, 산문 정의, 맞댈 사실 이름, 비교).
# ★축은 **코퍼스가 제안한 것**만 넣는다(`x142` 인구조사·설계서 §9-2). 내가 지어낸 축도,
#   gold(`tasks.json` 노트)에서 온 축도 없다 — 이 세션은 그 노트를 읽었으므로 특히 엄격히 지킨다([[23]]).
#   각 축 옆의 인용은 **정책 문서 축자**이고, 그것이 그 축의 출처 증명이다.
AXES = {
    # ── v0부터 있던 둘 ──────────────────────────────────────────────────────
    "referrer_tenure_days": {
        "applies_to": {"consumers": [{"tool": "submit_referral", "operand_arg": "account_type"}],
                       "basis": "env `submit_referral(user_id, account_type)` + A2 `action_tools` + 정책 축자 \"you must check that the user is eligible to submit referrals first\" (…bank_accounts_(general)_047)"},
        "desc": "the minimum number of days the REFERRER must already have held a checking "
                "account (relationship duration / tenure) to be eligible to refer",
        "against": "tenure_days", "compare": "ge",
        # "Eligibility: A minimum relationship duration of 60 days as a checking account
        #  holder is required." (…_hunter_green_001)
        "fit": r"\b\d{1,4}[\s-]*(?:calendar[\s-]+)?days?\b"},
    "annual_referral_limit": {
        "applies_to": {"consumers": [{"tool": "submit_referral", "operand_arg": "account_type"}],
                       "basis": "env `submit_referral(user_id, account_type)` + A2 `action_tools` + 정책 축자 \"you must check that the user is eligible to submit referrals first\" (…bank_accounts_(general)_047)"},
        "desc": "the maximum number of referral bonuses allowed per year for that product",
        "against": "type_usage", "compare": "le",
        # "Annual limit: Up to 10 referral bonuses per year" (…_hunter_green_001)
        "fit": r"(annual|annually|per\s+year|per-year|a\s+year|/\s*year|"
               r"per\s+calendar\s+year|calendar\s+year|each\s+year|yearly)"},

    # ── v2 신설 — `x142`가 센 라벨에서 왔다(라벨 병합표는 안 만든다·[[52]] 해석은 LLM 몫) ──
    "referrer_bonus_usd": {
        "applies_to": {"consumers": [],
                       "basis": "**도구 경계에서 판정되지 않는다** — 권고를 만들 때 쓰는 피연산자다. 억지로 도구에 붙이면 없는 게이트를 지어내는 것이라 비워 둔다(엔진은 표면화만 한다)."},
        # 라벨 `your bonus`·`you earn`·`referrer bonus`·`your reward`
        # "Your bonus: $200 for each successful referral" (…_lime_green_003)
        "desc": "the dollar amount the REFERRER receives for one successful referral",
        "against": None, "compare": None, "fit": r"\$"},
    "referred_bonus_usd": {
        "applies_to": {"consumers": [],
                       "basis": "**도구 경계에서 판정되지 않는다** — 권고를 만들 때 쓰는 피연산자다. 억지로 도구에 붙이면 없는 게이트를 지어내는 것이라 비워 둔다(엔진은 표면화만 한다)."},
        # 라벨 `their bonus`·`they receive`·`referred bonus`·`new member bonus`
        # "Their bonus: $150 welcome bonus for the referred business" (…_lime_green_003)
        "desc": "the dollar amount the REFERRED person or business receives as a welcome bonus",
        "against": None, "compare": None, "fit": r"\$"},
    "qualifying_deposit_usd": {
        "applies_to": {"consumers": [],
                       "basis": "**도구 경계에서 판정되지 않는다** — 권고를 만들 때 쓰는 피연산자다. 억지로 도구에 붙이면 없는 게이트를 지어내는 것이라 비워 둔다(엔진은 표면화만 한다)."},
        # 라벨 `qualifying deposit`·`required deposit`
        # "Qualifying deposit: The referred business must deposit at least $7,500 to trigger
        #  the referral bonus" (…_cobalt_blue_005)
        "desc": "the minimum dollar amount the REFERRED party must deposit to trigger the bonus",
        "against": None, "compare": None, "fit": r"\$"},
    "deposit_window_days": {
        "applies_to": {"consumers": [],
                       "basis": "**도구 경계에서 판정되지 않는다** — 권고를 만들 때 쓰는 피연산자다. 억지로 도구에 붙이면 없는 게이트를 지어내는 것이라 비워 둔다(엔진은 표면화만 한다)."},
        # 라벨 `deposit window`·`deposit deadline`·`time window`
        # "Time window: This deposit must be made within 90 days of account opening"
        #  (…_cobalt_blue_005)
        "desc": "the number of days the REFERRED party has to make the qualifying deposit",
        "against": None, "compare": None, "fit": r"\b\d{1,4}[\s-]*days?\b"},
    "rolling_window_referrals": {
        "applies_to": {"consumers": [{"tool": "submit_referral", "operand_arg": "account_type"}],
                       "basis": "env `submit_referral(user_id, account_type)` + A2 `action_tools` + 정책 축자 \"you must check that the user is eligible to submit referrals first\" (…bank_accounts_(general)_047)"},
        # 라벨 `weekly limit` + 산문. **상품이 아니라 계좌 부류 전체에 걸리는 규칙**이고
        # 값이 부류마다 다르다 — 그래서 연간 상한과 **다른 축**이다(v0는 이걸 연간으로 넣어 틀렸다).
        # "You can receive at most 2 referral bonuses in any rolling 9-day window."
        #  (…bank_accounts_(general)_047) / "…in any rolling 7-day window"
        #  (…credit_cards_(general)_009)
        "desc": "the maximum number of referral bonuses allowed inside one rolling window "
                "(NOT per year) — this rule applies across account types, not to one product",
        "against": None, "compare": None, "fit": r"(rolling|window)"},
    "rolling_window_days": {
        "applies_to": {"consumers": [{"tool": "submit_referral", "operand_arg": "account_type"}],
                       "basis": "env `submit_referral(user_id, account_type)` + A2 `action_tools` + 정책 축자 \"you must check that the user is eligible to submit referrals first\" (…bank_accounts_(general)_047)"},
        # 같은 문장의 **창 길이**. 건수와 창 길이는 둘 다 있어야 판정이 된다.
        "desc": "the length in days of that rolling window (e.g. 'rolling 9-day window')",
        "against": None, "compare": None, "fit": r"(rolling|window)"},
    "company_max_age_years": {
        "applies_to": {"consumers": [{"tool": "open_bank_account", "operand_arg": "account_class"}],
                       "basis": "env `open_bank_account_4821(user_id, account_type, account_class)` docstring 축자 \"account_class (string): The full official account class name\" + 정책 축자 \"## Verify Eligibility — Confirm your company is within 4 years of formation.\" (…sky_blue_001)"},
        # ⚠이 축은 `x142` 라벨 인구조사가 **구조적으로 못 본다** — 라벨 없는 산문이라서다.
        #   내가 처음에 *"코퍼스에 없다"* 고 단정했다가 실물에서 뒤집혔다(자기정정).
        # "Confirm your company is within 4 years of formation." (…_sky_blue_001)
        # "Eligibility criteria tied to company age are evaluated when you open the account;
        #  the threshold is 4." (…_sky_blue_008)
        # "Verify your company age is within 4." (…_sky_blue_009)
        "desc": "the maximum age in years, counted from formation, that a company may have and "
                "still be eligible to OPEN this account (a company older than this cannot open it)",
        "against": None, "compare": None, "fit": r"(year|age|formation|threshold)"},

    # ── 산문 pass(`x142 --prose`)가 낸 넷 — 라벨 pass는 이것들도 못 봤다 ──────────────
    "qualifying_spend_usd": {
        "applies_to": {"consumers": [],
                       "basis": "**도구 경계에서 판정되지 않는다** — 권고를 만들 때 쓰는 피연산자다. 억지로 도구에 붙이면 없는 게이트를 지어내는 것이라 비워 둔다(엔진은 표면화만 한다)."},
        # "The referred business must spend at least $3,000 within 90 days of account opening
        #  for you to receive the bonus" (…_business_silver_rewards_card_013)
        # 예치(deposit)와 **다른 축**이다 — 카드는 지출로, 계좌는 예치로 자격을 준다.
        "desc": "the minimum dollar amount the REFERRED party must SPEND (purchases, not a "
                "deposit) to trigger the referral bonus",
        "against": None, "compare": None, "fit": r"\$"},
    "holder_min_age_years": {
        "applies_to": {"consumers": [{"tool": "open_bank_account", "operand_arg": "account_class"},
                                     {"tool": "submit_referral", "operand_arg": "account_type"}],
                       "basis": "계좌 보유 자격이면서 동시에 추천 자격이다 — 정책 축자 \"The referred person must be 18 years or older\" (…047·추천 자격) 와 \"You must be … between 13 and 24 years old to open the account.\" (…light_green_002·개설 자격)"},
        # "The referred person must be 18 years or older" (…bank_accounts_(general)_047)
        # "The person you refer must be eligible for a Gold Years Account (age 62+)"
        # "You must be … between 13 and 24 years old to open the account." (…light_green_002)
        "desc": "the minimum age in years a person must be to hold or open this account",
        "against": None, "compare": None, "fit": r"(year|age)"},
    "holder_max_age_years": {
        "applies_to": {"consumers": [{"tool": "open_bank_account", "operand_arg": "account_class"}],
                       "basis": "env `open_bank_account_4821(user_id, account_type, account_class)` docstring 축자 \"account_class (string): The full official account class name\" + 정책 축자 \"## Verify Eligibility — Confirm your company is within 4 years of formation.\" (…sky_blue_001)"},
        # "You must remain within the 13–24 age range to maintain the account." (…light_green_002)
        "desc": "the maximum age in years a person may be and still hold this account "
                "(only where the policy states an upper bound)",
        "against": None, "compare": None, "fit": r"(year|age)"},
    "referred_no_prior_accounts_months": {
        "applies_to": {"consumers": [{"tool": "submit_referral", "operand_arg": "account_type"}],
                       "basis": "env `submit_referral(user_id, account_type)` + A2 `action_tools` + 정책 축자 \"you must check that the user is eligible to submit referrals first\" (…bank_accounts_(general)_047)"},
        # "The referred person must be a new Rho-Bank customer with no existing checking,
        #  savings, or closed accounts within the past 12 months." (…bank_accounts_(general)_047)
        "desc": "how many months back the REFERRED party must have had no Rho-Bank accounts "
                "to count as a new customer",
        "against": None, "compare": None, "fit": r"(month|year)"},
}

# ⓑ 축 적합성 — **형식 검사 둘뿐**. 의미 판단은 하지 않는다([[52]]: 엔진=이론·LLM=해석).
#   B2 표지는 *그 축이 무엇을 재는가*에서 나온다: 관계기간=일수 / 연간 상한=1년 주기.
AXIS_FIT = {k: re.compile(v["fit"], re.I) for k, v in AXES.items() if v.get("fit")}
COMMA = re.compile(r"(?<=\d),(?=\d\d\d)")


def to_int(s):
    """`$7,500` · `7,500` → 7500. 금액 축을 넣으며 필요해졌다 — 쉼표에 걸려 죽으면 사실을 잃는다."""
    return int(COMMA.sub("", str(s).replace("$", "").strip()))


# 인용 실재 검사의 **서식 잡음만** 지운다 — 마크다운 강조(`**`·`` ` ``)뿐이고 글자는 안 건드린다.
# ⚠왜 필요한가: 스모크서 `- **Maximum per year**: 6 referral bonuses`를 모델이 강조 없이 인용해
#   **진짜 사실이 떨어졌다**(Dark Green 연간 상한 6). 게이트의 역효과가 실물로 나온 자리다.
# ⚠`_`는 안 지운다 — `user_id` 같은 식별자를 뭉개 **없는 인용을 통과시킬** 수 있다.
EMPH = re.compile(r"[*`]")


def deformat(s):
    return " ".join(EMPH.sub("", str(s or "")).split())


def axis_fit(axis, value, quote):
    """(통과?, 사유) — B1 값이 인용에 있는가 · B2 축 표지가 인용에 있는가."""
    q = COMMA.sub("", " ".join(str(quote or "").split()))
    if not re.search(r"(?<!\d)%d(?!\d)" % int(value), q):
        return False, "B1 값이 인용에 없음"
    pat = AXIS_FIT.get(axis)
    if pat and not pat.search(q):
        return False, "B2 축 표지가 인용에 없음"
    return True, ""


PROMPT = """You are reading ONE internal banking policy document.

DOCUMENT TITLE: {title}

Extract ONLY facts of these kinds:
{axes}

Rules:
- Report a fact ONLY if this document states it explicitly. Do not infer.
- "subject" is the PRODUCT the fact is about. The sentence often omits it because the
  whole document is about one product — in that case use the product named in the TITLE.
  Never use a program name ("referral program", "Startup Referral Program") as the subject.
- "quote" must be copied VERBATIM from the document (a contiguous span, >= 12 chars).
- If the document states nothing of these kinds, return an empty list.

Return JSON only, no prose:
[{{"subject": "...", "axis": "...", "value": <integer>, "quote": "..."}}]

DOCUMENT:
{text}
"""


class Agent(object):
    def __init__(self, model, base):
        self.llm = model if model.startswith("openai/") else "openai/" + model
        self.llm_args = {"temperature": 0.0, "api_base": base, "api_key": "dummy"}


def ask(agent, la, UM, text, title, only_axis=None):
    """`only_axis`가 있으면 **그 축만** 묻는다(ⓐ 축별 질의)."""
    keys = [only_axis] if only_axis else list(AXES)
    axes = "\n".join("- %s: %s" % (k, AXES[k]["desc"]) for k in keys)
    prompt = PROMPT.format(axes=axes, text=text[:90000], title=title or "(none)")
    try:
        um = UM(role="user", content=prompt)
    except TypeError:
        um = UM(content=prompt)
    kw = {k: v for k, v in dict(agent.llm_args).items() if "tool" not in k}
    sub = la.generate(model=agent.llm, tools=None, messages=[um], call_name="x140", **kw)
    return getattr(sub, "content", None) or ""


def parse(raw, doc_text, doc_id, reasons=None, use_axis_fit=True, only_axis=None,
          debug=False, dropped=None):
    """모델 응답 → 채택 행. **인용이 그 문서에 실재할 때만** 채택한다(+ⓑ 축 적합성)."""
    def rej(why, row=None):
        if reasons is not None:
            reasons[why] += 1
        # 거절된 행을 **산출물에 남긴다** — 집계만 남기면 게이트의 반대편(과잉 거절)을 per-case로
        # 못 읽는다. v2 빌드가 정확히 그 상태였다(거절 12인데 실물이 로그에도 없었다).
        if dropped is not None and row is not None:
            dropped.append({"doc": doc_id, "why": why, "row": row})
        if debug:
            print("    ✗거절 %s :: %s" % (why, json.dumps(row, ensure_ascii=False)[:200]
                                          if row else ""), file=sys.stderr)

    m = re.search(r"\[.*\]", str(raw or ""), re.S)
    if not m:
        rej("응답에 JSON 배열 없음")
        return [], 0
    try:
        rows = json.loads(m.group(0))
    except Exception:
        rej("JSON 파싱 실패")
        return [], 0
    hay = " ".join(str(doc_text).split())
    out, rejected = [], 0
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            rejected += 1
            rej("행이 객체가 아님")
            continue
        ax, subj = str(r.get("axis") or ""), str(r.get("subject") or "").strip()
        q = " ".join(str(r.get("quote") or "").split())
        try:
            val = to_int(r.get("value"))
        except Exception:
            rejected += 1
            rej("값이 정수가 아님")
            continue
        # 축별 질의에서는 **물은 축만** 받는다 — 안 물은 축을 끼워 넣으면 분할이 무의미해진다
        if only_axis and ax != only_axis:
            rejected += 1
            rej("안 물은 축을 답함")
            continue
        if ax not in AXES or not subj or val <= 0 or len(q) < 12:
            rejected += 1
            rej("축 밖·주어 없음·인용 짧음", r)
            continue
        if q in hay:
            match = "exact"
        elif deformat(q) and deformat(q) in deformat(hay):
            match = "normalized"             # 서식만 다르다 — 글자는 그 문서의 것
        else:
            rejected += 1
            rej("인용이 문서에 없음", r)
            continue
        if use_axis_fit:
            ok, why = axis_fit(ax, val, q)
            if not ok:
                rejected += 1
                rej(why, r)                   # ⓑ 인용은 실재하나 **축이 안 맞는다**
                continue
        out.append({"applies_to": {"axis_default": True},
                    "subject": subj, "axis": ax, "value": val,
                    "against": AXES[ax]["against"], "compare": AXES[ax]["compare"],
                    "when": [],
                    "source": {"doc": doc_id, "quote": q, "quote_match": match}})
    return out, rejected


def audit(path):
    """ⓑ의 **1차 관문** — 기존 산출물에 축 적합성을 걸어 보고 과잉 거절을 잰다(LLM 0)."""
    op = gzip.open if path.endswith(".gz") else io.open
    with op(path, "rt", encoding="utf-8") as f:
        onto = json.load(f)
    rows = onto.get("rows") or []
    bad = []
    for r in rows:
        ok, why = axis_fit(r.get("axis"), r.get("value"),
                           (r.get("source") or {}).get("quote"))
        if not ok:
            bad.append((r, why))
    print("축 적합성 관문: 행 %d 중 **통과 %d · 거절 %d**" % (len(rows), len(rows) - len(bad), len(bad)))
    for r, why in bad:
        print("  ✗ %s / %s = %s — %s" % (r.get("subject"), r.get("axis"), r.get("value"), why))
        print("     인용: %s" % " ".join(str((r.get("source") or {}).get("quote") or "").split())[:130])
    print("⚠거절된 행은 **per-case로 읽어야** 한다 — 진짜 오행과 과잉 거절이 여기서 갈린다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", help="기존 온톨로지에 ⓑ 축 적합성만 걸어 본다(LLM 0)")
    ap.add_argument("--docs")
    ap.add_argument("--out")
    ap.add_argument("--per-axis", action="store_true",
                    help="ⓐ 축 하나씩 따로 묻는다 (호출 = 문서 × 축)")
    ap.add_argument("--no-axis-fit", action="store_true", help="ⓑ를 끈다(대조 arm용)")
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--limit", type=int, default=0, help="스모크용 문서 수 제한")
    ap.add_argument("--only", default="", help="쉼표로 나눈 부분문자열 — 그 문서만 (표적 스모크)")
    ap.add_argument("--contains", default="",
                    help="본문에 이 말이 있는 문서만. ⚠**표적 빌드**이지 전수가 아니다 — "
                         "이걸 쓴 산출물로 *'모집단을 다 봤다'*(§0a)를 주장하면 안 된다. "
                         "재개 키가 (문서|축)이라 나중에 전수로 이어 돌리면 여기서 한 질문은 "
                         "그대로 재사용된다.")
    ap.add_argument("--debug-reject", action="store_true", help="거절된 행을 그대로 인쇄")
    ap.add_argument("--save-every", type=int, default=20)
    a = ap.parse_args()

    if a.audit:                               # 관문 모드 — 모델도 문서도 안 쓴다
        return audit(a.audit)
    if not a.docs or not a.out:
        ap.error("--docs 와 --out 이 필요하다 (또는 --audit)")

    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage as UM
    agent = Agent(a.model, a.base)
    use_fit = not a.no_axis_fit
    reasons = collections.Counter()
    dropped = []

    docs = sorted(glob.glob(os.path.join(a.docs, "*.json")))
    if a.contains:
        keep, need = [], a.contains.lower()
        for p in docs:
            try:
                d = json.load(io.open(p, encoding="utf-8"))
            except Exception:
                continue
            if need in (str(d.get("content") or "") + str(d.get("title") or "")).lower():
                keep.append(p)
        print("⚠표적 빌드: 본문에 %r 있는 문서 %d개만 (전수 %d개 아님·완결성 주장 금지)"
              % (a.contains, len(keep), len(docs)))
        docs = keep
    if a.only:
        pats = [s.strip() for s in a.only.split(",") if s.strip()]
        docs = [p for p in docs if any(s in os.path.basename(p) for s in pats)]
    if a.limit:
        docs = docs[:a.limit]

    done = {}
    if os.path.exists(a.out):                 # ★재개 — 이미 읽은 문서는 건너뛴다
        prev = json.load(io.open(a.out, encoding="utf-8"))
        done = {d: True for d in prev.get("docs_done", [])}
        rows = list(prev.get("rows", []))
        stats = collections.Counter(prev.get("stats", {}))
        print("재개: 문서 %d개 완료 · 행 %d" % (len(done), len(rows)))
    else:
        rows, stats = [], collections.Counter()

    # ★질문 단위 = (문서, 축). 축별 질의면 축마다 하나, 아니면 문서마다 하나.
    #   ⚠재개 키에 **축이 들어간다** — 한 프롬프트를 두 단위가 쓰면 양쪽이 같은 답을 받고
    #     그대로 통과한다(C311의 재생 캐시 오염이 정확히 그 사고였다).
    ax_keys = list(AXES) if a.per_axis else [None]

    def save():
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(
            {"axes": {k: v["desc"] for k, v in AXES.items()},
             "per_axis": bool(a.per_axis), "axis_fit": use_fit,
             "targeted_contains": a.contains or None,   # 표적 빌드면 여기 남는다(완결성 주장 금지)
             "docs_total": len(docs), "units_total": len(docs) * len(ax_keys),
             "docs_done": sorted(done), "rows": rows,
             "stats": dict(stats), "reject_reasons": dict(reasons),
             "dropped": dropped},
            ensure_ascii=False, indent=1))

    for i, p in enumerate(docs, 1):
        d = json.load(io.open(p, encoding="utf-8"))
        did = d.get("id") or os.path.basename(p)
        for ax in ax_keys:
            key = did if ax is None else "%s|%s" % (did, ax)
            if key in done:
                continue
            try:
                raw = ask(agent, la, UM, str(d.get("content") or ""),
                          str(d.get("title") or ""), only_axis=ax)
            except Exception as e:
                stats["호출 실패"] += 1
                print("  ⚠%s 호출 실패 %r" % (key[-34:], e), file=sys.stderr)
                continue
            got, rej = parse(raw, d.get("content") or "", did, reasons=reasons,
                             use_axis_fit=use_fit, only_axis=ax, debug=a.debug_reject,
                             dropped=dropped)
            rows.extend(got)
            done[key] = True
            stats["질문"] += 1
            stats["채택"] += len(got)
            stats["거절"] += rej
            if got:
                print("  [%3d/%d] %-34s +%d행" % (i, len(docs), key[-34:], len(got)), flush=True)
            if len(done) % a.save_every == 0:
                save()

    save()

    # ── 병합·충돌 ────────────────────────────────────────────────────────────
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["subject"], r["axis"])].append(r)
    conflicts = {k: v for k, v in by_key.items() if len({x["value"] for x in v}) > 1}

    print("\n" + "=" * 92)
    print("질문 %d/%d(문서 %d × 축 %d) · 채택 행 %d · 거절 %d · 호출 실패 %d"
          % (stats["질문"], len(docs) * len(ax_keys), len(docs), len(ax_keys),
             stats["채택"], stats["거절"], stats["호출 실패"]))
    print("설정: 축별 질의 %s · 축 적합성 %s" % (bool(a.per_axis), use_fit))
    print("서로 다른 (주어, 축) = **%d개** · 그중 **값 충돌 %d개**" % (len(by_key), len(conflicts)))
    for ax in AXES:
        n = len({k for k in by_key if k[1] == ax})
        print("   %-24s 주어 %d개" % (ax, n))
    if reasons:
        print("\n거절 사유 (게이트의 반대편을 본다 · 등대 §1.3):")
        for why, n in reasons.most_common():
            print("   %-30s %d" % (why, n))
    if conflicts:
        print("\n★값 충돌 — **자동으로 고르지 않는다**(사람이 판정·[[25]]):")
        for (subj, ax), v in sorted(conflicts.items())[:10]:
            print("   %s / %s" % (subj, ax))
            for x in v:
                print("      %3d  [%s] %s" % (x["value"], x["source"]["doc"][-24:],
                                              x["source"]["quote"][:80]))
    print("\n저장: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
