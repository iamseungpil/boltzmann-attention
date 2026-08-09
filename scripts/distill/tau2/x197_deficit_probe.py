# -*- coding: utf-8 -*-
r"""x197 — task_098·task_010 에서 **모델이 정확히 무엇을 못하는가** (격리 서브·유료 0).

## 왜 이것이 먼저인가 ([[62]] ⛔0)

099/100 이 유효했던 이유는 격리 프로브로 **결손을 먼저 재고**(자기-정박·순서·타입 제약) 그
자리에만 레버를 놓았기 때문이다. 098/010 은 그 측정을 **한 번도 하지 않은 채** 결정론 기구를
지었고, 그것은 *"결정론기 짜고 LLM 은 형식적으로 쓰는"* 것이라 **gold 프로그램과 구별되지
않는다**(C384·사용자 격노). 그래서 코드를 되돌리고 여기서 다시 시작한다.

## 무엇을 가르는가

한 태스크의 실패는 셋 중 하나다. 팔을 그 셋에 맞춰 끊는다:

  **전달**   재료가 문맥에 없었다        → 격리에서 주면 된다        → 레버 = 전달뿐
  **산수**   재료는 있는데 수를 못 만든다 → 계산만 따로 물으면 틀린다 → 레버 = **원시연산 하나**
  **선택**   수도 맞는데 못 고른다        → 수를 줘도 틀린다          → 레버 = 그 자리(099형)

⚠원시연산을 만들더라도 **피연산자는 LLM 이 정한다**([[62]]) — 축·행을 A2 에 박으면 태스크마다
  기구가 늘고 그건 일반화가 아니다.

## 팔 (098)

  A_iso      엔진이 실제로 만드는 통과 표 + 정제된 피연산자 + **손님의 실제 첫 발화** → 고르나
  B_sum      같은 것 + 옵션별 **합계를 미리 계산해** 붙임 → 산수를 빼면 고르나
  C_calc     선택은 묻지 않고 **합만** 물음("각 옵션의 두 값을 더하면?") → 산수 자체를 잴 수 있나
  D_null     **부정 통제** — 표를 주지 않고 질문만 → 표 없이도 맞히면 프로브가 새는 것이다

## 팔 (010)

  A_iso      원장 4행(전사 형태) + 손님의 실제 첫 발화 → 어느 것이 지급 안 됐는지·왜를 말하나
  B_win      같은 것 + **창 상수**(정책 축자: 9일에 2건) → 상수를 주면 이유를 짚나
  C_calc     선택 없이 **산수만** 물음("각 기록 앞 9일 안에 몇 건이 있었나") → 날짜 산수를 하나
  D_null     부정 통제 — 원장 없이 질문만

## 읽는 법

  · A 가 되면 → 결손은 **전달**이다. 라이브에 재료가 안 닿은 것이고 새 기구는 필요 없다.
  · A 실패·B 성공 → **산수**다. `sum` 같은 **원시연산 하나**만 만들고 피연산자는 LLM 이 채운다.
  · A·B 실패·C 성공 → **선택**이다(수는 만드는데 못 고른다) — 099 형 결손.
  · C 도 실패 → 산수 자체를 못한다. 그때만 계산 도구가 정당하다.
  · D 가 성공하면 → 프로브가 새고 있다. 다른 결론을 내기 전에 **그것부터** 고친다.

실행: python x197_deficit_probe.py [N]
"""
import collections
import gzip
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
import t2_factdag as FD                                         # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
# 손님 발화·원장 행은 **라이브 궤적에서 읽어 온다**(프로브가 저작하지 않는다).
RUNS = {"task_098": "bank_anchorslot_20260809v098", "task_010": "bank_anchorslot_20260809v010"}
GOLD = {"task_098": "Blue", "task_010": "Platinum Rewards Card"}   # 진단 전용([[23]])
CASE = {"task_098": {"days": 400, "stated": {"qualifying_deposit_usd": 600}}}


def ask(prompt, choices=None, temp=0.0, mx=24):
    body = {"model": MODEL, "temperature": temp, "max_tokens": mx,
            "messages": [{"role": "user", "content": prompt}]}
    if choices:
        body["guided_choice"] = list(choices)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def live(task):
    """그 태스크의 **실제** 손님 첫 발화와 도구 출력(원장 행)을 궤적에서 꺼낸다."""
    p = os.path.join(SIMS, RUNS[task] + ".json.gz")
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    s = d["simulations"][0]
    first = next((str(m.get("content") or "") for m in s["messages"]
                  if m.get("role") == "user"), "")
    ledger = ""
    for m in s["messages"]:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and "referral" in c.lower() and "Record ID" in c:
            ledger = c
            break
    return " ".join(first.split()), ledger


def run(task, arms, choices, n):
    print("\n" + "=" * 100)
    print("%s  gold=%r  (n=%d · %s)" % (task, GOLD[task], n, MODEL))
    print("=" * 100)
    out = {}
    for label, prompt, kind in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                if kind == "choice":
                    c[ask(prompt, choices, 0.0 if i == 0 else 0.7)] += 1
                else:
                    c[ask(prompt, None, 0.0 if i == 0 else 0.7, mx=180)] += 1
            except Exception as e:
                c["ERR %s" % type(e).__name__] += 1
        # ★채점은 **정확 일치** (2026-08-09 자기적발): 부분 문자열로 세면 gold `Blue` 가
        #   `Sky Blue`·`Light Blue`·`Navy Blue` 에 걸려 **부정 통제가 4/8 로 새는 것처럼**
        #   보였다. 통제가 새면 다른 결론을 못 낸다 — 자유 서술 팔만 포함 검사를 쓴다.
        if kind == "choice":
            hit = sum(v for k2, v in c.items() if str(k2).strip() == GOLD[task])
        else:
            hit = sum(v for k2, v in c.items() if GOLD[task].lower() in str(k2).lower())
        out[label] = (hit, n, c.most_common(3))
        print("  %-10s gold %d/%d   최빈: %s" % (label, hit, n, c.most_common(2)))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    sp = next(x for x in a2["ledger_metrics"] if x.get("eligible_text"))
    rows = (a2.get("policy_ontology") or {}).get("rows") or []
    axes = (sp.get("eligible") or {}).get("show_axes") or []
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    res = {}

    # ── task_098 ────────────────────────────────────────────────────────────
    q98, _ = live("task_098")
    cs = CASE["task_098"]
    tbl = (LG.eligible_text(cs["days"], None, maps, sp, cs["stated"]) or "").strip()
    body = [l for l in tbl.splitlines() if l.startswith("  ") and ":" in l]
    names = [l.strip().split(":")[0].strip() for l in body]
    m1, m2 = maps.get("referrer_bonus_usd") or {}, maps.get("referred_bonus_usd") or {}
    combo = "\n".join("  %s: total = %s" % (s0, LG._num(m1[s0]) + LG._num(m2[s0]))
                      for s0 in names if s0 in m1 and s0 in m2)
    facts = "The customer says the deposit will be about %d." % cs["stated"]["qualifying_deposit_usd"]
    Q = "\n\nThe customer asked:\n%s\n\nAnswer with one name copied exactly from the list above, and nothing else." % q98
    # ★필터 축 (사용자 지적·C337/x150): 099/100 은 **격리 단독으로는 안 됐고** 미달 행을 뺀
    #   표라야 됐다. 098 도 같은 자리인지 가른다. 1차 측정에서 최댓값을 만든 `Gold Years` 는
    #   예치 문턱이 A3 에 없어 우리 필터가 못 걸렀다.
    crit_axes = ["qualifying_deposit_usd", "referrer_tenure_days"]
    ver = set(names)
    for _a in crit_axes:
        ver &= set(maps.get(_a) or {})
    tbl_ver = (chr(10)).join([l for l in tbl.splitlines()
                              if not (l.startswith("  ") and ":" in l)
                              or l.strip().split(":")[0].strip() in ver])
    raw_names = sorted({s0 for a in axes for s0 in (maps.get(a) or {})})
    raw = "Policy constants on record:" + chr(10) + (chr(10)).join(
        LG._row_line(s0, ["%s=%s" % (a, LG._num((maps.get(a) or {})[s0]))
                          for a in axes if s0 in (maps.get(a) or {})])
        for s0 in raw_names)
    print("  [표] raw %d행 · 통과 %d행 · 기준실재 %d행 (뺀 것: %s)"
          % (len(raw_names), len(names), len(ver),
             ", ".join(sorted(set(names) - ver))[:90]))
    res["098"] = run("task_098", [
        ("A_raw", raw + "\n\n" + facts + Q, "choice"),
        ("A_ver", tbl_ver + "\n\n" + facts + Q, "choice"),
        ("A_iso", tbl + "\n\n" + facts + Q, "choice"),
        ("B_sum", tbl + "\n\nAdding the two bonus figures for each option:\n" + combo
         + "\n\n" + facts + Q, "choice"),
        ("C_calc", tbl + "\n\nFor each option above, add referrer_bonus_usd and "
         "referred_bonus_usd. Reply with 'name = total' one per line, nothing else.", "free"),
        ("D_null", facts + Q, "choice"),
    ], sorted(set(names) | set(raw_names)), n)

    # ── task_010 ────────────────────────────────────────────────────────────
    q10, led = live("task_010")
    if not led:
        print("\n⚠ task_010 원장 도구 출력을 궤적에서 못 찾았다 — 그 팔은 건너뛴다.")
        return 0
    win = ("The policy on record: at most %s referral bonuses in any rolling %s-day window."
           % (sp.get("window_max"), sp.get("window_days")))
    Q10 = ("\n\nThe customer asked:\n%s\n\nWhich account type's referral did NOT pay out, and "
           "why? Answer in one short sentence." % q10)
    res["010"] = run("task_010", [
        ("A_iso", led + Q10, "free"),
        ("B_win", led + "\n\n" + win + Q10, "free"),
        ("C_calc", led + "\n\n" + win + "\n\nFor each record above, count how many other "
         "records fall within the 9 days before it. Reply 'name = count' one per line.", "free"),
        ("D_null", Q10, "free"),
    ], None, n)

    json.dump({k: {a: v[:2] for a, v in d.items()} for k, d in res.items()},
              open(os.environ.get("T2_X197_OUT", "x197_out.json"), "w"), indent=1)
    print("\n※ A 성공 = 전달 결손 · A실패/B성공 = 산수 · A·B실패/C성공 = 선택 · D성공 = 프로브 누수")
    return 0


if __name__ == "__main__":
    sys.exit(main())
