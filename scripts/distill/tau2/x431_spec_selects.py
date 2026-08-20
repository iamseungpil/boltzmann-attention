# -*- coding: utf-8 -*-
r"""x431 — **스펙이 gold 를 고르나**: 빈칸 사유 확정 → 제약 형식화 → 결정론 필터 (사용자 지시 2026-08-20)

## 위치 ([[62]] 절차)
    ① 사실표 저작        x430 (LLM 형식화 + 엔진 인용 검산) — 1차 완료
    ①b 빈칸 사유 확정    여기 · 문서 미기재인가 추출 실패인가
    ② 제약 형식화        여기 · **손님 발화 축자만** 보고 닫힌 술어로 (LLM)
    ③ 결정론 필터        여기 · 엔진이 표에 술어를 적용 (판단 0·산술과 비교뿐)
    ④ 판정              스펙이 gold 를 고르나 · 못 고르면 그 태스크는 **미결정**

## 규율
★②의 입력은 **손님 발화뿐**이다. 문서도 gold 도 안 넣는다 — 제약은 손님이 말한 것이지
  정답에서 역산한 것이 아니어야 한다([[23]]).
★③의 엔진은 **비교만** 한다(==·<=·>=·존재). 어느 속성을 볼지는 ②가 정한다([[10]]).
★gold 는 ④ 채점에만 등장한다.
★스펙이 못 고르면 그것은 **모델의 실패가 아니라 우리 표·스펙의 한계**로 먼저 읽는다([[55]]).

사용: py -3 x431_spec_selects.py [--port 8141] [--fill-blanks]
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x423_choice_isolation as I  # noqa: E402
import x426_free_gates as G  # noqa: E402
import x427_catalog_minimal as CM  # noqa: E402
import x430_account_facts as FT  # noqa: E402

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TBL = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x430_account_facts_llm.json")
ATTRS = [x[0] for x in FT.ATTRS]


def card_table():
    """★카드 사실표는 **A2 에 이미 있다** — `check_card_application_fit.op.table`([[67]] 사본 금지).

    2026-08-20 실측: 우리 추출 속성 16개는 **계좌 어휘**라 카드 계열에서 값이 4/284 밖에 안 붙었다.
    그런데 우리 fit 도구가 카드별 사실(annual_fee·fx_fee·cashback·limit_max·min_score·
    purchase_protection·virtual_card)을 **문서 출처와 함께** 이미 들고 있다. 그것을 그대로 쓴다.
    타입은 그 표의 값 꼴에서 유도한다(불리언·수치) — 우리가 새로 정하지 않는다.
    """
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        d = json.load(f)
    tools = [x for x in (d.get("scaffold_get_tools") or [])
             if x.get("name") == "check_card_application_fit"]
    if not tools:
        return {}, []
    rows = ((tools[0].get("op") or {}).get("table")) or []
    PCT = {"fx_fee", "cashback", "min_payment_pct", "base_cashback"}
    table, attrs = {}, []
    for r in rows:
        name = r.get("card")
        if not name:
            continue
        cell = {"_family": "credit_cards"}
        for k, v in r.items():
            if k in ("card", "source") or isinstance(v, (dict, list)):
                continue
            unit = ("percent" if k in PCT else
                    ("boolean" if isinstance(v, bool) else
                     ("count" if k == "min_score" else "USD")))
            cell[k] = {"values": [str(v)], "unit": unit, "cap": None,
                       "evidence": [{"value": str(v), "doc": r.get("source", ""), "quote": ""}]}
            if k not in attrs:
                attrs.append(k)
        table[name] = cell
    # ★조건부 선언을 **A2 에서 그대로 실어 온다**(2026-08-20 밤·계기 충실도 수리).
    #   라이브 정본(`t2_compute.py:353-379`)은 `op.conditional_fields` 를 읽어 *조건 참=대체값 ·
    #   거짓=기본값 · **미지이고 판정이 갈리면 unverified(배제 금지)*** 로 처리한다. 이 프로브만 그 선언을
    #   안 읽어서 003 의 gold(`Silver`: fx 2.75 / **fx_fee_with_premium 0.0**)가 여기서만 죽었다 —
    #   같은 태스크의 라이브 궤적에서는 Silver 가 `eligible` 에 **들어간다**(t7326 실측). 계기가 라이브보다
    #   가혹하면 그 프로브가 만든 결손은 **우리 것**이다([[55]]·[[67]]).
    cond = ((tools[0].get("op") or {}).get("conditional_fields")) or {}
    for _name, cell in table.items():
        for base, spec in cond.items():
            alt = (spec or {}).get("alt")
            if isinstance(cell.get(base), dict) and alt in cell:
                cell[base]["cond_alt"] = alt
                cell[base]["cond_when"] = (spec or {}).get("when")
    return table, attrs


def arg_families(arg):
    """그 인자의 후보가 올 **문서 계열**(A2 구조 선언). 없으면 None = 제한 없음."""
    try:
        with io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8") as f:
            return (json.load(f).get("catalog_arg_families") or {}).get(arg)
    except Exception:
        return None


CARD_PCT = {"fx_fee", "cashback", "min_payment_pct", "base_cashback", "fx_fee_with_premium"}


def ops_suffix(at, caps):
    """속성 옆에 **그 표가 지지하는 op** 를 붙인다 — 모델이 만족 불가능한 술어를 고르지 못하게."""
    o = sorted(caps.get(at) or (), key=lambda x: ("==", "<=", ">=", "exists", "absent").index(x))
    return ("; ops %s" % ",".join(o)) if o else "; ops none"


def attr_menu_for(arg, tbl=None):
    """★인자에 맞는 **속성 메뉴**를 준다(2026-08-20 수리).

    `card_type` 사례에 계좌 속성 목록을 주고 있었다 — 모델이 *"credit limit at least $100,000"* 을
    `monthly_deposit_limit`(계좌 칸)로 옮기고 `foreign_currency_holding exists` 를 걸어, 카드 표에
    없는 칸이라 **후보가 전멸**했다(003 생존 0). 카드는 A2 fit 표의 속성으로 묻는다.
    """
    if arg == "card_type":
        _t, attrs = card_table()
        def _ty(k):
            if k in CARD_PCT:
                return "percent"
            if k in ("purchase_protection", "virtual_card", "invite_only", "business"):
                return "boolean"
            if k == "min_score":
                return "count"
            return "USD"
        caps = table_caps(_t, arg_families(arg)) if tbl is None else table_caps(tbl, arg_families(arg))
        return ", ".join("%s (%s%s)" % (k, _ty(k), ops_suffix(k, caps))
                         for k in attrs if caps.get(k))
    return attr_menu(tbl)


def attr_menu(tbl=None):
    """형식화에 **타입까지** 준다 — 어느 op 가 맞는지는 타입이 정한다([[10]] LLM 이 쓰고 엔진은 비교만).

    실물(2026-08-20): `foreign_currency_holding == 1` 은 *존재*를 뜻했는데 값은 **30**(지원 통화 수)이라
    벌점 |30-1| = 29 가 붙어 **가장 잘 만족하는 계좌가 가장 크게 깎였다**. 타입을 안 주면 모델은
    개수 칸에 존재 술어를 쓴다. 타입 선언의 자리는 A2 `catalog_attrs` 다([[24]] 구조 선언).
    """
    try:
        p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
        with io.open(p, encoding="utf-8") as f:
            d = json.load(f).get("catalog_attrs") or {}
        caps = table_caps(tbl, arg_families("account_class")) if tbl else {}
        return ", ".join("%s (%s%s)" % (k, (v.get("type") or "unknown"),
                                        ops_suffix(k, caps) if tbl else "")
                         for k, v in d.items() if (not tbl) or caps.get(k))
    except Exception:
        return ", ".join(ATTRS)
RE_MONEY = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def ask(port, sysmsg, body, maxtok=700):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
                         "messages": [{"role": "system", "content": sysmsg},
                                      {"role": "user", "content": body}]}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
    i, j = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[i:j + 1]) if i >= 0 and j > i else {}
    except Exception:
        return {}


OPS = ("==", "<=", ">=", "exists", "absent")

# ★서법을 **LLM 이 인자로 선언**한다(사용자 설계 2026-08-20 · [[22]] 따름정리·선언-우선).
#   엔진이 `?`·`would` 를 세는 것은 [[59]] 위반이었고 실제로 gold 를 죽였다. 그렇다고 서법을 아예
#   안 보면 057 t0 처럼 *"Would that be enough to avoid any monthly fee?"* 가 하드 제약이 된다.
#   해법은 **판단의 주체를 옮기는 것**이지 판단을 없애는 게 아니다 — LLM 이 `stated_as` 를 선언하고
#   엔진은 그 **선언된 라벨만** 읽는다(낱말은 안 본다·판단 0).
ACTS = ("requirement", "question", "background")

# ★인용 검산 전 **표기 정규화**(2026-08-20·x432 로 확정). 손님 발화에는 user-sim 이 넣은 마크다운
#   강조가 섞여 있다 — *"I need an account with absolutely **no overdraft fees**."* 모델은 별표 없이
#   인용하므로 순수 substring 검산이 **전부 튕겼다**(거절 8건 전부 이것). 유니코드 대시·따옴표도 같다.
#   ⚠이건 **형태** 정규화다. 뜻은 건드리지 않는다([[59]]).
_PUNCT = {"‘": "'", "’": "'", "“": '"', "”": '"',
          "–": "-", "—": "-", "−": "-", " ": " "}


def cite_norm(t):
    t = str(t or "")
    for k, v in _PUNCT.items():
        t = t.replace(k, v)
    t = t.replace("*", "").replace("_", "")
    return " ".join(t.split()).lower()


def check_spec(spec, said="", allow=None):
    """★G6 — 스펙 스키마 **강제** + **근거 검산**. 통과한 제약과 **거절 사유 문장**을 함께 돌려준다.

    x431 1차: 063 이 `"because"` 를 속성명으로 쓰고 `apy == None` 을 냈다(스키마 붕괴).
    x431 2차: 서법·근거가 틀렸다 — 057 t0 은 *"**Would** that be enough to avoid any monthly fee?"*
    (질문)을 `monthly_maintenance_fee == 0` 하드 제약으로 만들어 gold 를 죽였고, 063 은 `apy exists`
    의 근거로 **종이명세서 문장**을 달았다.

    그래서 [[66]] 이 확정한 형태를 그대로 쓴다 — **가리키기 + substring 검산**:
        ⒜ 스키마(속성 enum · op enum · 값 수치 — `bool` 은 `float()` 를 통과하므로 따로 막는다)
        ⒝ `because` 가 손님 발화의 **부분문자열**인가(근거 불일치 차단)
    둘 다 **형태**이고 엔진은 뜻을 해석하지 않는다([[10]]·[[22]]).

    ⛔**서법 검산은 뺐다(2026-08-20·사용자 지적).** `?`·`would`·`is there` 목록으로 *"요구인가 질문인가"*
      를 가르는 것은 **화행 판단 = 의미론**이고, 어휘 목록으로 그것을 대체하는 것이 [[59]] 가 금지한
      자리다. 실측도 같은 말을 했다 — 켠 판에서 **제약이 0개로 남은 사례가 둘**, gold 생존 1/5 → 0/5.
      서법 판단은 **LLM 이 한다**(프롬프트). 엔진은 그 판단이 **가리킨 근거의 실재**만 본다.
    """
    norm = cite_norm(said)
    ok, bad, dropped, dropped_cons = [], [], [], []
    for con in (spec.get("constraints") or []):
        if not isinstance(con, dict):
            bad.append("constraint is not an object")
            continue
        at, op, val = con.get("attribute"), con.get("op"), con.get("value")
        if at not in (allow or ATTRS):
            bad.append("unknown attribute %r (not in the allowed list)" % at)
            continue
        alts = con.get("any_of")
        if alts is not None:
            # ★이접 — 대안 하나하나를 **같은 규칙**으로 검산한다(형태만·뜻은 안 본다).
            okops = set(allow[at]) if isinstance(allow, dict) else set(OPS)
            if not isinstance(alts, list) or len(alts) < 2:
                bad.append("attribute %s has 'any_of' that is not a list of at least two alternatives" % at)
                continue
            worst = None
            for alt in alts:
                if not isinstance(alt, dict) or alt.get("op") not in okops:
                    worst = ("attribute %s has an 'any_of' alternative with op %r (allowed here: %s)"
                             % (at, (alt or {}).get("op") if isinstance(alt, dict) else alt,
                                ",".join(sorted(okops)) or "none"))
                    break
                if alt.get("op") in ("==", "<=", ">="):
                    try:
                        if isinstance(alt.get("value"), bool):
                            raise TypeError
                        float(alt.get("value"))
                    except Exception:
                        worst = ("attribute %s has an 'any_of' alternative with op %s but value %r"
                                 % (at, alt.get("op"), alt.get("value")))
                        break
            if worst:
                bad.append(worst)
                continue
        # ★`allow` 가 **속성→op 집합** 이면 표의 능력까지 검산한다(2026-08-20 밤).
        #   거절은 **고칠 것을 이름 대고** 말한다([[64]]) — 무엇이 없는지 + 무엇이 되는지.
        okops = set(allow[at]) if isinstance(allow, dict) else set(OPS)
        if alts is None and op not in OPS:
            bad.append("attribute %s has op %r (allowed: %s)" % (at, op, ",".join(OPS)))
            continue
        if alts is None and op not in okops:
            bad.append("attribute %s does not support op %r in this catalogue "
                       "(no row records that; use one of: %s)"
                       % (at, op, ",".join(sorted(okops)) or "none"))
            continue
        if alts is None and op in ("==", "<=", ">="):
            try:
                if isinstance(val, bool):     # ⚠`float(False)` 는 0.0 으로 **통과한다** — 055 가 그 구멍으로
                    raise TypeError           #   `foreign_transaction_fee == False` 를 냈다(형태 결함)
                float(val)
            except Exception:
                bad.append("attribute %s with op %s needs a number, got %r" % (at, op, val))
                continue
        act = con.get("stated_as")
        if act not in ACTS:
            bad.append("attribute %s has stated_as %r (allowed: %s)" % (at, act, ",".join(ACTS)))
            continue
        why = " ".join(str(con.get("because") or "").split())
        whyn = cite_norm(why)
        if norm:
            if not why:
                bad.append("attribute %s has no 'because' span" % at)
                continue
            if whyn not in norm:
                bad.append("the 'because' for %s is not a verbatim span of what the customer said: %r"
                           % (at, why[:60]))
                continue
        if act != "requirement":
            # ★축자를 자르지 않는다(2026-08-20 밤): 60자 절단본이라 저장된 span 으로 표면형을 세면
            #   *"물음표가 없다"* 가 모델의 성질인지 **우리 절단**인지 구분이 안 됐다(19건 중 8건이 정확히 60자).
            dropped.append((at, act, why))           # 요구가 아니라고 **모델이 선언한 것**은 안 건다
            dropped_cons.append(con)                 #   버리지 않고 **선호 채널**로 넘긴다
            continue
        ok.append(con)
    return ok, bad, dropped, dropped_cons


def term_cost(row_attr, times, of_amount):
    """속성 하나를 **사용량으로 환산해 USD** 로 만든다. 엔진의 몫은 여기까지 — 산술뿐이다([[10]]).

    `1% of withdrawal amount (max $3.00)`(unit=percent·cap=3.0) 와 `$2.50`(unit=USD) 은 환산 없이는
    비교할 수 없다. 어느 속성을 · 몇 번 · 얼마에 대해 볼지는 **스펙이** 정하고(=LLM 형식화),
    엔진은 곱셈·상한·합만 한다.
    """
    vals = (row_attr or {}).get("values") or []
    if not vals:
        return None
    v = num(vals[0])
    if v is None:
        return None
    unit = (row_attr or {}).get("unit") or ""
    cap = (row_attr or {}).get("cap")
    per = (v / 100.0 * float(of_amount)) if (unit == "percent" and of_amount) else v
    if cap is not None:
        per = min(per, float(cap))
    return per * float(times or 1)


def objective_pick(obj, table, survivors):
    """스펙의 목적함수로 생존 후보를 **순위**짓는다 → (승자 목록, 점수표).

    ★이 자리가 우리 카드 도구가 명세로 **거부**한 자리다(*"the customer's soft preferences before the
      final pick"*). 057·024·003 이 전부 여기서 갈린다(argmin 사용량×수수료 / argmax 요율×금액−연회비).
    """
    mode = (obj or {}).get("mode")
    terms = [t for t in ((obj or {}).get("terms") or []) if isinstance(t, dict)]
    if mode not in ("argmin", "argmax") or not terms:
        return None, {}
    # ★항이 **하나라도 모름이면 그 후보는 순위에서 뺀다**(2026-08-20 수리·사용자 지적).
    #   전에는 풀린 항만 더해 **부분합**으로 겨루게 했다 — 값이 없는 후보가 0.00 으로 1위에 서고
    #   문서가 값을 준 후보(blue 3.20)가 4위로 밀렸다. 모름은 0 이 아니라 **판정 보류**다.
    score, held = {}, []
    for cls in survivors:
        row = table.get(cls) or {}
        tot, miss = 0.0, False
        for t in terms:
            at = t.get("attribute")
            if not at:
                continue
            c = term_cost(row.get(at), t.get("times", 1), t.get("of_amount"))
            if c is None:
                miss = True
                break
            tot += c * float(t.get("sign", 1))
        if miss:
            held.append(cls)
        else:
            score[cls] = tot
    if held:
        print("        ⓘ순위 보류(항이 미기재) %d 후보: %s" % (len(held), ", ".join(held[:4])))
    if not score:
        return None, {}
    best = (min if mode == "argmin" else max)(score.values())
    return [c for c, v in score.items() if abs(v - best) < 1e-9], score


def pref_penalty(row_attr, op, val):
    """선호 하나에 대한 **벌점**(0 = 완전 만족). 모르면 None — 0 이 아니다."""
    vals = (row_attr or {}).get("values") or []
    if op in ("exists", "absent"):
        has = bool(vals)
        return 0.0 if (has if op == "exists" else not has) else 1.0
    if not vals:
        return None
    x = num(vals[0])
    if x is None or val is None:
        return None
    v = float(val)
    if op == "==":
        return abs(x - v)
    if op == "<=":
        return max(0.0, x - v)
    if op == ">=":
        return max(0.0, v - x)
    return None


def preference_rank(prefs, table, survivors):
    """★선호 채널(2026-08-20·사용자 설계): 요구가 아니라고 **모델이 선언한 것**을 버리지 않고 **순위**로 쓴다.

    x433 실측이 이 자리를 가리켰다 — 생존을 가르는 속성 17 중 4가 *"손님이 말했는데 우리가 버린 것"*
    이었고, 그 넷은 전부 `question`·`background` 로 선언-기각한 항목이었다. 하드 제약으로 쓰면 gold 가
    죽고(생존 0) 버리면 안 좁혀진다(생존 36) — **필터가 아니라 랭킹**이면 둘 다 피한다([[70]] 되사기).
    우리 카드 도구 명세도 같은 말을 한다: *"the customer's **soft preferences** before the final pick"*.

    ⚠null 규율은 그대로 — 항이 하나라도 모르면 그 후보는 **순위 보류**(0 으로 섞지 않는다).
    """
    terms = [x for x in (prefs or []) if x.get("attribute")]
    if not terms:
        return None, {}, []
    score, held = {}, []
    for cls in survivors:
        row = table.get(cls) or {}
        tot, miss = 0.0, False
        for t in terms:
            pen = pref_penalty(row.get(t["attribute"]), t.get("op"), t.get("value"))
            if pen is None:
                miss = True
                break
            tot += pen
        (held.append(cls) if miss else score.setdefault(cls, tot))
    if not score:
        return None, {}, held
    best = min(score.values())
    return [c for c, v in score.items() if abs(v - best) < 1e-9], score, held


def clsname(x):
    """표 키·gold 이름을 같은 꼴로 — 소문자·밑줄→공백·계열꼬리·괄호 제거."""
    x = str(x).split("@")[0].lower().replace("_", " ").replace("(checking)", "")
    return " ".join(x.split())


def num(v):
    """'$0.00' · 'None' · '1% of withdrawal' → 수. **비어 있음은 0 이 아니라 None** 이다.

    ★수리(2026-08-20·사용자 지적): `"-"`·`""`·`"n/a"` 를 0.0 으로 접고 있었다. 문서가 *"None"* 이라고
      **말한 것**은 수수료 $0 이 맞지만, 우리가 **값을 못 가진 칸**은 0 이 아니라 **모름**이다.
      그 둘을 섞으면 미기재 후보가 공짜 후보와 동률이 되어 순위를 오염시킨다.
    """
    s = str(v).strip().lower()
    if s in ("", "-", "n/a", "unknown", "not stated"):
        return None                     # 모름 — 비교에서 빠진다
    if s in ("none", "no"):
        return 0.0                      # 문서가 "없다"고 **말한** 것 = $0
    m = RE_MONEY.search(s.replace(",", ""))
    return float(m.group(0)) if m else None


def cellval(cell):
    """셀 하나를 **비교 가능한 수**로 — 단위는 표가 선언한 것을 엔진이 읽을 뿐이다([[10]] 비교만).

    ★수리(2026-08-20 밤·C553 후속): 불리언 칸은 값이 `"True"`/`"False"` 라 `num()` 이 **None** 을 내고,
      그 자리에서 비교가 통째로 **건너뛰어졌다**. 실측: 003 의 `purchase_protection == 1` 이 후보 13 을
      **전부** 통과시켰다 — 거르는 척하는 제약이다. `absent` 가 아무도 못 통과시킨 것의 **거울상**이고,
      둘 다 우리 층이 만든 조용한 왜곡이다([[55]]).
    """
    vals = (cell or {}).get("values") or []
    if not vals:
        return None
    if (cell or {}).get("unit") == "boolean":
        t = str(vals[0]).strip().lower()
        return 1.0 if t in ("true", "yes") else 0.0 if t in ("false", "no") else None
    return num(vals[0])


def table_caps(tbl, fams):
    """그 표가 **실제로 만족시킬 수 있는 술어**만 유도한다 — 도메인 지식 0·표의 형태만 본다.

    ★결함 확정(2026-08-20 밤·C553): 카드표에는 `absent` 표시가 **한 칸도 없다**(`card_table()` 이 값 셀만
      짓는다). 그런데 속성 메뉴는 `absent` 를 제시했고, 그 술어가 하나만 들어오면 후보는 **반드시 0** 이
      됐다 — 063 t0 의 전멸이 그 자리다. 만족 불가능한 술어를 권해 놓고 전멸을 모델의 과잉 필터로 읽는 것은
      우리 층의 결함이다([[55]]). 능력은 **표에서 유도**하고, 없으면 **메뉴에 올리지 않는다**.
    """
    caps = {}
    for _cls, row in (tbl or {}).items():
        if not isinstance(row, dict):
            continue
        if fams and (row.get("_family") not in fams):
            continue
        for at, c in row.items():
            if not isinstance(c, dict):
                continue
            s = caps.setdefault(at, set())
            if c.get("values"):
                s.add("exists")
                if cellval(c) is not None:
                    # 불리언 칸에 순서 비교는 뜻이 없다 — 타입이 op 를 정한다([[10]]).
                    s.update(("==",) if c.get("unit") == "boolean" else ("==", "<=", ">="))
            if c.get("absent"):
                s.add("absent")
    return caps


def _cmp(cell, op, target):
    """한 칸을 한 수와 비교한다 — 값이 없거나 기준이 없으면 **거르지 않는다**(과차단 방지)."""
    v = cellval(cell)
    if v is None or target is None:
        return True
    t = float(target)
    return (v == t) if op == "==" else (v <= t) if op == "<=" else (v >= t) if op == ">=" else True


def passes(row, con):
    """제약 하나를 후보 하나에 건다 — **비교뿐**이다([[10]]). 미기재 칸은 거르지 않는다(과차단 방지·C462).

    ★`any_of` = **이접**(2026-08-20 밤 추가·C555 후속). 한 마디가 표에 **두 꼴로 적힐 수 있을 때**
      — *"no foreign transaction fee"* 가 어떤 계좌엔 값 `0%`, 어떤 계좌엔 *"해당 없음"* 으로 —
      우리 op 은 둘 중 하나를 고르게 강제했고 그래서 gold 만 떨어졌다(055 생존 0). 이접은 **언어 확장**
      이지 뜻의 입법이 아니다: 대안 목록은 **LLM 이 짜고** 엔진은 여전히 **비교와 OR** 만 한다.
    """
    at = con.get("attribute")
    if not at:
        return True
    alts = con.get("any_of")
    if isinstance(alts, list) and alts:
        return any(passes(row, dict(alt, attribute=at)) for alt in alts if isinstance(alt, dict))
    op = con.get("op")
    c = (row.get(at) or {})
    alt = c.get("cond_alt")
    if alt and op in ("==", "<=", ">="):
        # 조건부 칸 — **조건의 참·거짓을 우리가 모르므로**, 두 값의 판정이 갈리면 배제하지 않는다
        # (라이브 `t2_compute` 의 unverified 와 같은 자리·이 프로브에는 조건을 채울 인자가 없다).
        base_ok = _cmp(c, op, con.get("value"))
        alt_ok = _cmp(row.get(alt) or {}, op, con.get("value"))
        return True if base_ok != alt_ok else base_ok
    if op == "exists":
        return bool(c.get("values"))
    if op == "absent":
        return bool(c.get("absent"))          # 문서가 **없다고 말한 것**만 absent 다
    v, t = cellval(c), con.get("value")
    if v is None or t is None:
        return True
    t = float(t)
    return (v == t) if op == "==" else (v <= t) if op == "<=" else (v >= t) if op == ">=" else True


def fill_blanks(table, docdir, family, port):
    """빈칸마다 **표적 질의** — 문서가 값을 주나. 안 주면 `absent` 로 확정한다.

    ⚠**계열을 다 돌아야 한다**(2026-08-20 수리·사용자 지적): 표가 4계열 38 클래스로 넓어졌는데 이 함수는
      `--family` 하나만 받아 나머지 계열의 문서를 못 찾고 **조용히 건너뛰었다**(`if not docs: continue`).
      그 결과 608 칸 중 **313(51%)이 질의된 적조차 없는데** 우리는 그것을 *"문서 미기재"* 로 읽고
      순위 보류를 문서 탓으로 돌렸다. 빈칸의 사유를 모르면 **모른다고 적어야** 한다([[25]]).

    ★그리고 질의는 **문서 단위**로 한다 — x434 민감도 통제: 한 덩어리 80% ↔ 문서별 **90%** ↔ 3등분 100%
      (단 3등분은 오추출을 산다 — 월 수수료를 당좌차월료로 붙였다). 그래서 **문서별이 최적점**이다.
    """
    prefix = "doc_%s_" % family
    byc = collections.defaultdict(list)
    for f in sorted(os.listdir(docdir)):
        if not (f.startswith(prefix) and f.endswith(".json")):
            continue
        cls = re.sub(r"_\d+\.json$", "", f[len(prefix):])
        with io.open(os.path.join(docdir, f), encoding="utf-8") as fh:
            d = json.load(fh)
        byc[cls].append((f.replace(".json", ""), (d.get("title") or "") + ". " + (d.get("content") or "")))
    sysmsg = ("Answer ONLY from the documents. Reply with ONE JSON object: "
              "{\"value\": \"<verbatim>\", \"quote\": \"<verbatim sentence>\"} if the documents state it, "
              "otherwise {\"absent\": true}. Never paraphrase.")
    filled = absent = dropped = 0
    for cls, row in table.items():
        docs = byc.get(cls) or []
        if not docs:
            continue
        joined = " ".join(" ".join(t.split()) for _i, t in docs)
        blob = "\n\n".join("### %s\n%s" % (i, t) for i, t in docs)[:40000]
        for attr in ATTRS:
            if row.get(attr, {}).get("values"):
                continue
            got = ask(port, sysmsg, "# Documents for the %s\n%s\n\n# Question\nWhat is the %s?\n"
                      % (cls, blob, attr.replace("_", " ")), maxtok=300)
            if got.get("absent"):
                row[attr] = {"values": [], "conflict": False, "evidence": [], "absent": True}
                absent += 1
                continue
            val, q = str(got.get("value", "")).strip(), " ".join(str(got.get("quote", "")).split())
            if val and q and q in joined and val.replace(" ", "") in q.replace(" ", ""):
                row[attr] = {"values": [val], "conflict": False,
                             "evidence": [{"value": val, "doc": cls, "quote": q[:220]}]}
                filled += 1
            else:
                row[attr] = {"values": [], "conflict": False, "evidence": [], "unresolved": True}
                dropped += 1
    print("빈칸 처리 — 채움 %d · 문서 미기재 확정 %d · 검산 탈락(미해결) %d" % (filled, absent, dropped))
    return table


def sysmsg_spec(with_or=False, filler=False):
    """제약 형식화 프롬프트 — **x432 가 같은 것을 쓴다**(프롬프트가 갈리면 안정성 측정이 무효).

    ★`with_or` = 이접 어휘를 **스키마로만** 알린다(2026-08-20 밤·C555 후속). *언제* 쓰라는 규칙은
      한 줄도 넣지 않는다 — 케이스를 열거하면 그 열거가 독립 트리거가 되어 복합 발화를 밀어버린다
      ([[66]]·C453 은 formalize 프롬프트에 케이스 한 줄을 넣어 judge6 **−6** 을 냈다). 여기서 재는 것은
      *"어휘를 주면 모델이 스스로 쓰는가"* 이지 *"쓰라고 시키면 쓰는가"* 가 아니다.
    ★`filler` = 부정통제([[57]]): **길이만 같고 능력은 0** 인 문장을 대신 붙인다. 프롬프트가 길어져서
      제약이 늘어난 것인지, 이접이라는 **능력** 때문인지를 가른다.
    """
    orline = ("A constraint may instead list alternatives: {\"attribute\": \"<name>\", \"any_of\": "
              "[{\"op\": \"<op>\", \"value\": <number or null>}, ...], \"because\": \"...\", "
              "\"stated_as\": \"...\"}. It holds when any one alternative holds. ") if with_or else ""
    if filler:
        orline = ("A constraint is written as one JSON object and every field name is lower case; the "
                  "attribute name must be copied exactly as given, character for character, and the "
                  "value must be written as a bare number without a currency symbol or a percent sign. ")
    return (orline +"You turn a customer's stated requirements into a machine-checkable spec. "
              "Reply with ONE JSON object only: {\"constraints\": [{\"attribute\": \"<one of the given "
              "names>\", \"op\": \"==|<=|>=|exists|absent\", \"value\": <number or null>, "
              "\"because\": \"<the customer's own words>\", "
              "\"stated_as\": \"requirement|question|background\"}], "
              "\"objective\": {\"mode\": \"argmin|argmax\", \"terms\": [{\"attribute\": \"<name>\", "
              "\"times\": <how many times per month the customer said they do this>, "
              "\"of_amount\": <the dollar amount each time, or null>, \"sign\": 1 or -1}]}}. "
              "Use ONLY requirements the customer actually stated. Omit attributes they did not "
              "mention. Add 'objective' ONLY if the customer asked for the best/cheapest option or "
              "gave a usage pattern; otherwise omit it. "
              "'because' MUST be copied character-for-character from the customer's words. "
              "For every constraint you MUST label 'stated_as': 'requirement' if the customer said "
              "they need or must have it; 'question' if they only asked about it or asked whether "
              "something would be enough; 'background' if it is a fact about themselves rather than "
              "something they demand. Label honestly - do not relabel a question as a requirement. "
              "For an eligibility range give BOTH bounds (min_age <= their age AND max_age >= their age).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--docdir", default=FT.DOCDIR)
    ap.add_argument("--family", default="checking_accounts")
    ap.add_argument("--fill-blanks", action="store_true")
    ap.add_argument("--tag", default="wide", help="반복 산출물 이름 — 커밋된 것을 덮지 말 것")
    ap.add_argument("--table", default="auto",
                    help="auto = 전수 표(_filled)가 있으면 그것 · raw = 채우기 전 표")
    ap.add_argument("--repeat", type=int, default=1,
                    help="같은 코드로 N회 반복 — **판정선**(무처치 변동폭)을 잰다")
    ap.add_argument("--args-filter", default="account_class")
    a = ap.parse_args()

    # ★기본 표 정정(2026-08-20 밤): 오후에 코퍼스 전수로 완성한 표는 `_filled` 에 쓰이는데 여기는
    #   **채우기 전 표**를 읽고 있었다 — 확대 표본(2/8)이 값 192·미해결 704 짜리 표에 대고 걸린 이유다.
    #   미해결 칸은 `absent` 를 만족시키지 못하므로 그 차이는 **전멸로 나타난다**(063 t1: 0 ↔ 16).
    tbl_path = os.path.abspath(TBL)
    filled = tbl_path.replace(".json", "_filled.json")
    if a.table == "auto" and os.path.exists(filled) and not a.fill_blanks:
        tbl_path = filled
    with io.open(tbl_path, encoding="utf-8") as f:
        table = json.load(f)
    print("표 = %s" % os.path.basename(tbl_path))
    if a.fill_blanks:
        fams = FT.FAMILIES if a.family == "all" else [a.family]
        for fam in fams:
            table = fill_blanks(table, a.docdir, fam, a.port)
        with io.open(os.path.abspath(TBL).replace(".json", "_filled.json"), "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=1)

    # ② 제약 형식화 — 손님 발화만 본다
    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] not in [x for x in a.args_filter.split(",") if x]:
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)
    sysmsg = sysmsg_spec()
    print("\n=== ②③④ 스펙 → 필터 → 판정 (사례 %d) ===" % len(cs))
    # ★`--repeat` 은 선언만 돼 있고 **아무 일도 안 하고 있었다**(2026-08-20 밤 발견).
    #   확대 표본의 '×3 재현'은 바깥에서 세 번 부른 것이었다 — 플래그가 거짓말을 하면
    #   재현 주장 자체가 검증 불가다(핸드오프 §6 '런처 플래그 불일치'와 같은 부류).
    all_rows = []
    for _rep in range(max(1, a.repeat)):
      rows, rejected = [], {}
      for c in cs:
          said = G.customer_said(c["sim"], c["msg_i"])
          # ★표는 사례마다 고른다(카드/계좌) — 능력·메뉴·검산이 **같은 표**를 봐야 한다.
          tbl = card_table()[0] if c["arg"] == "card_type" else table
          fams = arg_families(c["arg"])
          caps = table_caps(tbl, fams)
          body = ("# Customer's own words\n%s\n\n# Attribute names you may use\n%s\n"
                  % (said[:6000], attr_menu_for(c["arg"], tbl)))
          spec = ask(a.port, sysmsg, body)
          allow = {k: sorted(v) for k, v in caps.items()}   # ★속성만이 아니라 **op 능력**까지 검산한다
          cons, bad, dropped, dropped_full = check_spec(spec, said, allow)
          n0, reask = len(bad), False
          if bad:
              reask = True                       # ★G6: 위반을 **이름 대고** 되묻는다([[64]] 거부는 고칠 것을 말해야)
              spec2 = ask(a.port, sysmsg, body + "\n# Your previous answer was rejected\n"
                          + "\n".join("- %s" % b for b in bad[:6])
                          + "\nUse ONLY the attribute names listed above, an op from "
                            "==,<=,>=,exists,absent, and a number (or null for exists/absent).\n")
              cons2, bad2, dropped2, dropped_full2 = check_spec(spec2, said, allow)
              if len(cons2) >= len(cons):
                  cons, bad, spec, dropped, dropped_full = cons2, bad2, spec2, dropped2, dropped_full2
          rejected[(c["task"], c["trial"])] = bad
          surv, why = [], []
          # ★표가 지지 못하는 술어는 **거르지 않고 판정에서 뺀다**(2026-08-20 밤·C553 수리).
          #   검산(G6)이 이미 막지만, 메뉴·검산·엔진이 갈리면 그 어긋남은 **조용한 전멸**로 나온다 —
          #   063 t0 이 그 모양이었다(카드표에 `absent` 표시가 0 인데 `absent` 제약 2건 → 후보 0).
          unsupported = [x for x in cons if x.get("op") not in (caps.get(x.get("attribute")) or ())]
          if unsupported:
              cons = [x for x in cons if x not in unsupported]
              print("        ⚠표가 지지 못하는 술어 %d건(판정 제외): %s"
                    % (len(unsupported), ", ".join("%s %s" % (u.get("attribute"), u.get("op"))
                                                   for u in unsupported[:4])))
          for cls, row in tbl.items():
              if not isinstance(row, dict):
                  continue
              if fams and (row.get("_family") not in fams):
                  continue                      # 계열 밖 후보는 애초에 안 센다(A2 선언)
              ok = all(passes(row, con) for con in cons)   # ★정본 술어 하나로([[67]] 사본 금지)
              if ok:
                  surv.append(cls)
          # ★제약이 하나도 안 남았으면 **목적함수를 돌리지 않는다**(2026-08-20 수리).
          #   직전 판에서 제약 0 인 사례 2건을 목적함수가 혼자 결정했고(`argmax waiver × −1` → `beige`),
          #   그것은 엔진이 나쁜 스펙 위에서 **최종 판단을 해 버리는** 자리다([[62]] 금지).
          obj = spec.get("objective") if isinstance(spec, dict) else None
          undecidable = not cons
          winners, score = (None, {}) if undecidable else objective_pick(obj, tbl, surv)
          # ★선호 채널 — 선언-기각된 것을 **순위**로 되쓴다(버리지 않는다).
          # ★`background` 는 순위 항이 아니다(2026-08-20 수리). 그것은 **손님에 대한 사실**이지 선호가 아니다.
          #   실물: 055 는 *"I'm 33, so not student/senior"* 가 `min_age` background 로 실렸고, purple 은
          #   `min_age` 가 미기재라 **순위 보류**로 빠졌다 — 표에 gold 의 근거 셋(해외수수료 0% · ATM 리베이트
          #   $30 · 30통화 보유)이 축자로 다 있었는데도. 선호는 손님이 **묻거나 바란 것**(question)만이다.
          prefs = [x for x in (dropped_full or [])
                   if x.get("attribute") in ATTRS and x.get("stated_as") == "question"]
          pbest, pscore, pheld = preference_rank(prefs, tbl, surv)
          # ★목적함수는 **필터 뒤 순위**일 뿐 생존 집합을 대체하지 않는다 — 둘을 따로 보고한다.
          surv_filter = list(surv)
          # ★채점은 **정확 일치**여야 한다(2026-08-20 수리): 표가 45 클래스로 넓어지면서 `silver_account`·
          #   `silver_plus_account`·`silver_saver_account` 가 공존한다. 옛 `startswith(첫 낱말)` 규칙은
          #   그 셋을 전부 `Silver Plus Account` 의 적중으로 셌다 — 채점기가 후해지면 결론이 무효다([[25]]).
          hit = [x for x in surv_filter if clsname(x) == clsname(c["gold"])]
          win_hit = bool(winners) and any(clsname(x) == clsname(c["gold"]) for x in winners)
          rows.append({"task": c["task"], "trial": c["trial"], "gold": c["gold"],
                       "objective": obj if winners else None,
                       "scores": {k: round(v, 4) for k, v in (score or {}).items()},
                       "n_constraints": len(cons), "n_rejected": len(rejected.get((c["task"], c["trial"])) or []),
                       "rejected": rejected.get((c["task"], c["trial"])) or [],
                       "n_surv": len(surv_filter), "surv": surv_filter,
                       "undecidable": undecidable, "reask": reask, "n_rejected_first": n0,
                       "unsupported": unsupported,
                       "declared_non_requirement": dropped,
                       "winner": winners, "winner_is_gold": win_hit,
                       "pref_best": pbest, "pref_held": len(pheld),
                       "pref_is_gold": bool(pbest) and any(clsname(x) == clsname(c["gold"]) for x in pbest),
                       "gold_in": (not undecidable) and bool(hit),
                       "unique": (not undecidable) and (len(surv_filter) == 1 and bool(hit)),
                       "spec": cons})
          if pbest:
              print("        선호 순위 → %s  (보류 %d · 상위: %s)"
                    % (", ".join(pbest[:3]), len(pheld),
                       ", ".join("%s=%.2f" % (k, v) for k, v in sorted(pscore.items(), key=lambda x: x[1])[:4])))
          if dropped:
              print("        선언-기각(모델이 요구가 아니라 함): %s"
                    % ", ".join("%s=%s" % (x[0], x[1]) for x in dropped[:4]))
          print("  %-9s t%s gold=%-22s 제약 %d개(거절 %d) → 생존 %2d %s%s"
                % (c["task"], c["trial"], c["gold"][:22], len(cons),
                   len(rejected.get((c["task"], c["trial"])) or []), len(surv),
                   "· gold 포함" if hit else "· ⛔gold 탈락",
                   " ✅유일" if (len(surv) == 1 and hit) else ""))
          if winners:
              print("        objective %s → %s" % ((obj or {}).get("mode"),
                    ", ".join("%s=%.2f" % (k, v) for k, v in sorted(score.items(), key=lambda x: x[1])[:4])))
              for t in ((obj or {}).get("terms") or [])[:3]:
                  print("           term %-30s ×%-4s of %s sign %s"
                        % (t.get("attribute"), t.get("times"), t.get("of_amount"), t.get("sign")))
          for con in cons[:4]:
              print("        %-34s %-6s %-8s ← %s" % (con.get("attribute"), con.get("op"),
                                                      con.get("value"), str(con.get("because"))[:60]))
      n = len(rows)
      if n:
          print(chr(10) + '  ★[%d회차] 스펙이 gold 를 남긴 사례 **%d/%d** · 유일하게 고른 사례 **%d/%d**'
                % (_rep + 1, sum(x['gold_in'] for x in rows), n, sum(x['unique'] for x in rows), n))
      all_rows.append(rows)
    rep = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
    outs = []
    for k, rws in enumerate(all_rows):
        # 1회면 정본 이름, 여러 회면 회차별 파일(확대 표본이 쓰던 이름 그대로).
        # ⚠재런은 **다른 이름**으로 — 커밋된 산출물을 조용히 덮어쓰면 인용의 근거가 사라진다([[30]]).
        name = ("x431_spec_selects.json" if len(all_rows) == 1
                else "x431_%s%d.json" % (a.tag, k + 1))
        q = os.path.abspath(os.path.join(rep, name))
        with io.open(q, "w", encoding="utf-8") as f:
            json.dump(rws, f, ensure_ascii=False, indent=1)
        outs.append(q)
    if len(all_rows) > 1:
        # ★판정선 — 회차 간 폭. 0 이 아니면 이 파이프의 어떤 비교도 그 폭 안에서는 무의미하다.
        gi = [sum(x["gold_in"] for x in r) for r in all_rows]
        un = [sum(x["unique"] for x in r) for r in all_rows]
        print(chr(10) + "  ★판정선: gold_in %s · unique %s (폭 %d/%d)"
              % (gi, un, max(gi) - min(gi), max(un) - min(un)))
    for q in outs:
        print("→ %s" % q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
