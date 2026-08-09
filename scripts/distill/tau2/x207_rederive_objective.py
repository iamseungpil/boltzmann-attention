# -*- coding: utf-8 -*-
r"""x207 — 재도출이 **왜 NONE 인가**, 그리고 목적을 다시 실으면 어떻게 되나 (격리 A/B · 유료 0).

## 라이브가 세운 사실 (`bank_a3fill_20260810a` · 12 sim)

098 은 세 시행 모두 **정확히 옳은 표**를 받았다 — 예치 `600` 수용 · 종류 `checking_accounts` ·
카드/사업자 25 주어 제외. 격리에서 그 표에 손님 질문을 붙이면 `Blue` **8/8**(x201 F_kind·G_llm).
그런데 라이브는 **0/3** 이고, 그 사이에 있는 것은 한 칸뿐이다:

    [T2_REDERIVE] raw='NONE' → 목록 밖 = 침묵      (098: 3/3 침묵 · 099: `World Blue`=gold)

`decided_text` 가 선언돼 있으면 표는 메인에 안 실린다(C367/C370). 그래서 재도출이 침묵하면
**그 경로에서 메인으로 나가는 것이 하나도 없다.**

## 왜 NONE 인가 — 우리 문구가 그렇게 시켰다

호출부는 x158 근거로 `_obj = ""` 를 넘긴다(목적 구절을 실으면 099 가 10/10 → 0/10 이었다).
그런데 문구는 *"The customer says: {asked} … If the customer's words do not single out one,
reply NONE"* 이다. **빈 말은 아무것도 짚지 못하므로 NONE 이 문구상 옳은 답이다.**
099 는 그 지시를 무시하고 답했고(gold), 098 은 따랐다. 창-산수 꼬리말(C388)과 같은 계열이다.

⚠**x158 을 뒤집는 것이 아니다.** 그때의 표에는 A3 예치 문턱도 종류 필터도 없었다(카드가 표에
  남아 있었고 099 가 *"전부 카드"* 로 무너진 것이 바로 그 증상이다). **조건이 달라졌으므로
  다시 잰다** — 같은 조건에서 재론하는 것이 아니다([[03]]).

## 팔 (각 태스크 · gold 는 진단 전용·[[23]])

  CUR       현행 그대로 (`asked=""`)                        ← 라이브에서 도는 것
  NOCLAUSE  손님-말 블록과 NONE 조항을 **뺀다**(표+사실만)
  OBJ       `asked` = 손님의 **목적 구절 축자**              ← x158 이 해롭다고 한 구성

## 읽는 법

  · CUR 이 098 에서 NONE 이면 → 침묵의 원인은 모델이 아니라 **우리 문구**다.
  · NOCLAUSE 가 세 태스크 다 살면 → 조항만 빼면 된다(가장 싼 수정).
  · OBJ 가 099/100 을 다시 무너뜨리면 → x158 은 새 표에서도 유효하고 목적은 싣지 않는다.
  · 셋 다 098 을 못 살리면 → 병목은 이 칸이 아니다. 다른 데를 봐야 한다.

실행: python x207_rederive_objective.py [N]   (T2_PROBE_URL 로 포트 지정)
"""
import collections
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

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8141/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

# 케이스는 **`tasks.json` 의 시나리오 축자**에서 가져온다(gold 는 진단 전용·[[23]] — 표적
# 선정에만 쓴다). 예치액과 종류는 라이브 trace 와도 일치한다(`value: …` · `[T2_KIND] raw=…`).
#
# ⚠**자기적발 (1차 실행)**: 처음에 `days=400`(모든 문턱 통과)으로 고정하고 *"이 세 태스크를
#   가르는 축은 예치이지 재직기간이 아니다"* 라고 적었다. **100 에 대해 정확히 틀렸다** —
#   task_100 은 재직기간 함정이다(축자: *"Customer opened their first checking account only 65
#   days ago … World Blue requires 90 days … Hunter Green only requires 60 days"*). 그 가정
#   때문에 World Blue 가 표에 남아 세 팔이 전부 오답으로 보였다. 통제가 틀리면 다른 결론을
#   못 낸다([[08]]).
CASE = {
    # 체킹 문턱은 전부 ≤ 45일이라 재직기간이 가르지 않는다(쟁점은 예치·9일 창).
    "task_098": dict(days=400, tally={}, stated={"qualifying_deposit_usd": 600},
                     kind="checking_accounts", gold="Blue",
                     obj="the best combined referral bonus - the total of what I get plus what she gets"),
    # 099 는 재직기간이 쟁점이 아니고, Hunter Green 연간 9/10 사용이 **미끼**다(상한 10이라 여유 1).
    "task_099": dict(days=400, tally={"Hunter Green": 9}, stated={"qualifying_deposit_usd": 30000},
                     kind="business_checking_accounts", gold="World Blue",
                     obj="the biggest referral bonus for me"),
    # ★100 = 재직 65일. World Blue(90) 탈락 · Hunter Green(60) 통과 — A3 에 이미 있는 값이다.
    "task_100": dict(days=65, tally={}, stated={"qualifying_deposit_usd": 31000},
                     kind="business_checking_accounts", gold="Hunter Green",
                     obj="the biggest referral bonus for me"),
}


def ask(prompt, choices, temp):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 24,
            "messages": [{"role": "user", "content": prompt}], "guided_choice": list(choices)}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    a2 = load_domain_a2("banking_knowledge")
    sp = next(x for x in a2["ledger_metrics"] if x.get("eligible_text"))
    cfg = sp["eligible"]
    rows = (a2.get("policy_ontology") or {}).get("rows") or []
    kb = LG.subject_kinds(rows, cfg.get("kind_field") or "kind")
    maps0 = {ax: FD._a3_map(rows, {"axis": ax}) for ax in (cfg.get("show_axes") or [])}
    tpl = sp["rederive_prompt"]
    out = {}
    for task, c in CASE.items():
        maps, _d = LG.restrict_to_kind(maps0, kb, c["kind"])
        tbl = (LG.eligible_text(c["days"], c.get("tally") or {}, maps, sp, c["stated"]) or "").strip()
        names = [l.strip().split(":")[0].strip() for l in tbl.splitlines()
                 if l.startswith("  ") and ":" in l]
        facts = "\n".join(["days since the earliest account was opened = %d" % c["days"]]
                          + ["%s = %s" % (k, LG._num(v)) for k, v in sorted(c["stated"].items())])
        print("\n%s  표 %d행: %s   gold=%r" % (task, len(names), ", ".join(names), c["gold"]))
        if c["gold"] not in names:
            print("  ⚠gold 가 표에 없다 — 이 팔은 무의미하다. 케이스를 고쳐라.")
        # NOCLAUSE: 손님-말 블록과 NONE 조항을 지운 문구 (앞부분은 한 글자도 안 바꾼다)
        noclause = tpl.split("The customer says:")[0].rstrip() + (
            "\n\nAnswer with one name copied exactly from the list above, and nothing else.")
        for arm in ("CUR", "NOCLAUSE", "OBJ"):
            if arm == "NOCLAUSE":
                p0 = noclause.format(table=tbl, facts=facts)
            else:
                p0 = tpl.format(table=tbl, facts=facts,
                                asked=("" if arm == "CUR" else c["obj"]))
            cnt = collections.Counter()
            for i in range(n):
                try:
                    cnt[ask(p0, names + ["NONE"], 0.0 if i == 0 else 0.7)] += 1
                except Exception as e:
                    cnt["ERR %s" % type(e).__name__] += 1
            hit = sum(v for k, v in cnt.items() if str(k).strip() == c["gold"])
            out["%s/%s" % (arm, task)] = [hit, n]
            print("  %-9s gold %d/%d   %s" % (arm, hit, n, cnt.most_common(3)))
    json.dump(out, open(os.environ.get("T2_X207_OUT", "x207_out.json"), "w"), indent=1)
    print("\n※ CUR 이 098 에서 NONE 이면 침묵은 우리 문구 탓이다. NOCLAUSE 가 셋 다 살리면"
          "\n  조항만 빼면 되고, OBJ 가 099/100 을 무너뜨리면 목적은 계속 싣지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
