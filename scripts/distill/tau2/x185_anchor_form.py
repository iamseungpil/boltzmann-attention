# -*- coding: utf-8 -*-
r"""x185 — **정박의 세기는 형식이 정하는가**: 산문 대 JSON 인자 템플릿 대 거리 (유료 0).

## 왜

x184 가 오염 판정을 닫았다(32B·14B):

  · `+commit`(꼬리 자기-약속 **한 문장만**)이 `full` 을 재현한다 — 100 은 완전(0/8), 099 는 부분(5/8).
  · **`+commit→gold`(지목 이름만 gold 로) 는 8셀 중 7셀이 8/8** ⇒ **정박이 답을 결정한다.**
    표 증거와 거의 무관하다(x183 의 Δ 무효와 정합).
  · `+commit→alt` 의 오답이 **그 이름으로 따라간다**(32B/100 `Sky Blue` · 14B/099-desc `Cobalt Blue`)
    ⇒ 효과는 지목된 **이름**이다. 초안 §6.1 이 열어 둔 가름 프로브가 닫혔다.
  · **14B/100 은 능력 결손이 아니다** — `iso` 0/8 인데 `+commit→gold` 8/8(양 정렬).

남은 갈림: **같은 `d=0` 인데 099 는 5/8 이고 100 은 0/8** 이다. 거리로는 설명 안 된다.
두 꼬리 문장은 형식이 다르다 — 099 는 **산문 639자**, 100 은 **905자 + 숫자 근거($150)
+ ```json 인자 템플릿```**. 그리고 100 은 `+commit−name`(이름 제거)으로도 안 풀린다
(숫자가 지목자로 남았을 가능성).

## 축 (교차 오염 0 — 합성 정본 템플릿 하나를 형식만 바꿔 쓴다)

두 태스크의 실제 문장을 서로 이식하면 상대 태스크의 user_id·손님 사실이 딸려 온다.
그래서 **한 템플릿**을 쓰고 이름만 그 태스크의 정박으로 채운다.

  iso            주입 없음
  prose          "…the {NAME} Account, which will maximize your referral bonus."   (099 형식)
  prose+num      위 + "as it offers a higher referral bonus of ${V}"               (숫자 근거)
  json           prose+num + ```json {"account_type": "{NAME}"}``` 인자 템플릿      (100 형식)
  json+pad       json 과 표 사이에 **중립 패딩 ~1000자** (거리 축·도메인 어휘 0)

  sort  name_asc · name_desc      후보  표의 모든 주어

## 읽는 법

  · `json` 이 `prose` 보다 세면        → **정박 세기 = 형식**. 099/100 차이가 형식으로 환원된다.
  · `prose+num` 이 이미 세면          → 숫자 근거가 레버(이름 제거로 안 풀리던 것과 정합).
  · `json+pad` 가 풀리면              → 거리(최근성)가 세기를 깎는다 = RoPE-decay 정합.
  · 다 같으면                          → 형식도 거리도 아니다. 세기 축을 다시 잡아야 한다.

⚠주입 문장은 **합성**이다(궤적 축자 아님). 형식 축을 분리하기 위한 통제이고, 그래서
   `prose` 가 099 의 실제 결과를 재현하는지를 같은 표에서 함께 본다.

실행: python x185_anchor_form.py [N]
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

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TAG = os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i")
CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}
LEAD = "Here is a customer-service conversation so far.\n\n"
PAD_UNIT = ("ASSISTANT: Thank you for your patience. Please hold on for a moment while "
            "the details are reviewed. USER: Sure, take your time. ")


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def last_named(text, choices):
    best, pos, ln = None, -1, -1
    for c in choices:
        p = text.rfind(c)
        if p < 0:
            continue
        if p + len(c) > pos or (p + len(c) == pos and len(c) > ln):
            best, pos, ln = c, p + len(c), len(c)
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    bax = next(a for a in axes if "bonus" in a.lower() and "referrer" in a.lower())
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731

    def bval(nm):
        v = (maps.get(bax) or {}).get(nm)
        try:
            return int(float(str(v[0]).replace(",", "")))
        except Exception:
            return 0

    print("model=%s · n=%d" % (MODEL, n))
    out = []
    for task, case in CASE.items():
        gold = X.GOLD[task]
        lines = LG.eligible_text(case["days"], {}, maps, spec,
                                 {"qualifying_deposit_usd": case["deposit"]}).strip().splitlines()
        head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
        body = [l for l in lines if l.startswith("  ") and ":" in l]
        ALL = [name(l) for l in body]

        ms = Y.msgs_of(TAG, task)
        tail = [m for m in ms if m.get("role") == "assistant"][-1]
        tail_txt = "ASSISTANT: " + " ".join(str(tail.get("content") or "").split())
        anchor = last_named(tail_txt, ALL)
        v = bval(anchor)

        prose = ("ASSISTANT: Let's proceed with guiding you through the referral process for "
                 "the %s Account, which will maximise your referral bonus." % anchor)
        prose_num = prose + (" It offers a higher referral bonus of $%d." % v)
        js = prose_num + (
            "\nTo proceed, use the `submit_referral` tool with the following arguments:\n"
            "- `account_type`: The type of account you are referring them to open (`%s`)\n"
            "The tool call will look like this:\n```json\n"
            "{ \"name\": \"submit_referral\", \"arguments\": { \"account_type\": \"%s\" } }\n"
            "```" % (anchor, anchor))
        pad = PAD_UNIT * (1000 // len(PAD_UNIT) + 1)

        print("\n" + "=" * 100)
        print("%s  gold=%r · 정박=%r($%d) · 실제 꼬리 %d자"
              % (task, gold, anchor, v, len(tail_txt)))
        print("  주입 길이: prose %d · prose+num %d · json %d · pad %d"
              % (len(prose), len(prose_num), len(js), len(pad)))
        print("=" * 100)
        arms = [("iso      ", ""),
                ("prose    ", LEAD + prose + "\n\n"),
                ("prose+num", LEAD + prose_num + "\n\n"),
                ("json     ", LEAD + js + "\n\n"),
                ("json+pad ", LEAD + js + "\n\n" + pad + "\n\n"),
                ("real_tail", LEAD + tail_txt + "\n\n")]
        print("  %-9s | %-22s | %s" % ("arm", "name_asc", "name_desc"))
        for alabel, pre in arms:
            cells = []
            for rev in (False, True):
                order = sorted(body, key=name, reverse=rev)
                tbl = "\n".join(head[:1] + order + head[1:]).strip()
                prompt = pre + tbl + "\n\n" + X.FACTS[task] + "\n\n" + X.QUESTION
                c = collections.Counter()
                for i in range(n):
                    try:
                        c[guided_full(prompt, ALL, 0.0 if i == 0 else 0.7)] += 1
                    except Exception as e:
                        c["ERR %s" % type(e).__name__] += 1
                cells.append("%d/%d %-17s" % (c.get(gold, 0), n, c.most_common(1)[0][0][:17]))
                out.append({"task": task, "arm": alabel.strip(),
                            "sort": "desc" if rev else "asc", "anchor": anchor,
                            "gold_hit": c.get(gold, 0), "n": n, "dist": dict(c)})
            print("  %-9s | %s | %s" % (alabel, cells[0], cells[1]))

    json.dump(out, open(os.environ.get("T2_X185_OUT", "x185_out.json"), "w"), indent=1)
    print("\n  json > prose 면 정박 세기 = 형식 · pad 가 풀면 거리(RoPE-decay) · 다 같으면 축 재설정.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
