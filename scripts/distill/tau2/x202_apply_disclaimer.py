# -*- coding: utf-8 -*-
r"""x202 — 010: **우리 꼬리말이 답을 막던 것**을 고친다 (A2 문구만·엔진 0·유료 0).

## 근거 (x200·격리 A/B·n=8·32B)

같은 계산을 주고 **꼬리말만** 바꿨다:

  OLD     이유 **0/8**   *"제공된 정보엔 거절 사유가 없다"*
  NEW     이유 **6/8**
  OLDDOC  이유 8/8       (상태 정의 문서가 문맥에 있으면 옛 꼬리말도 안 막는다)
  NEWDOC  이유 **8/8**

y010 사이드카는 두 문장이 **실제로 나갔음**을 보인다(상태별세기×4·창산수×4). 즉 재료도 계산도
도달했는데, 우리가 그 뒤에 *"이것은 왜 그 상태인지 말하지 않는다"* 를 붙였고 모델은 그대로
따랐다 — 궤적에서 손님이 *"그건 왜를 답하지 않는다"* 고 되묻자 상담원 이관으로 갔다([[55]]).

## 무엇으로 바꾸나

**날조를 막으려던 의도는 유지한다** — 여전히 우리는 이유를 말하지 않는다. 대신 *"모른다"* 가
아니라 **어디에 있는지**를 말한다: 상태의 정의는 원장이 아니라 문서에 있고(코퍼스 축자:
`doc_credit_cards_credit_cards_(general)_001` = *"REJECTED - the user has too many referral
processes going on"*), 그 문서를 가져오라고 한다. 판단·연결은 여전히 모델 몫이다([[62]]·[[22]]).

⚠[[24]]: `ledger_metrics` 는 `settings.json` 과 `gate.json` 두 층에 있다. 둘 다 같은 값으로
  바꾸고, 바꾼 뒤 **두 층의 값이 같은지** 확인한다.
⚠`settings.json` 은 손으로 정렬된 포맷이라(일부 객체가 한 줄) JSON 재직렬화가 파일을 통째로
  다시 쓴다. 그래서 **문자열 치환**으로만 손대고, 치환 후 JSON 으로 다시 읽어 값이 정확히
  바뀌었는지·나머지가 그대로인지 검사한다.

실행: py -3 x202_apply_disclaimer.py [--apply]
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["a2/banking_knowledge.settings.json", "a2/banking_knowledge.gate.json"]

OLD_STATUS = ("\nOf the {total} record(s) above, grouped by the status each record carries: "
              "{breakdown}. This is a count of what the records say, not a judgement: the records "
              "state the status, they do not state why it is that status.")
NEW_STATUS = ("\nOf the {total} record(s) above, grouped by the status each record carries: "
              "{breakdown}. This is a count of what the records say, not a judgement: the records "
              "carry the status but not its definition. The knowledge base has a document that "
              "defines what each status value means - retrieve it before explaining a status.")

OLD_WINDOW = ("\nDate arithmetic on the records above, each one against the ones before it: "
              "{crowded}. The rolling-window allowance on record is {max} in {days} days. This "
              "says how many records already fell inside the window when each of these was made - "
              "it does not say why any record carries the status it carries.")
NEW_WINDOW = ("\nDate arithmetic on the records above, each one against the ones before it: "
              "{crowded}. The rolling-window allowance on record is {max} in {days} days. This is "
              "arithmetic on the dates, so a record that already had {max} inside its window is a "
              "fact you may state; what that fact implies for the record's status is in the "
              "document that defines the statuses - retrieve it and say which applies.")

NOTE = ("x200(격리 A/B·n=8·32B): 같은 계산에 꼬리말만 바꾸니 이유 진술이 0/8 → 6/8, 상태 정의 "
        "문서까지 닿으면 8/8. 옛 꼬리말(*'왜인지는 말하지 않는다'*)을 모델이 그대로 따라 "
        "*'모른다'* 로 갔다(y010 궤적). 이유는 여전히 **우리가 말하지 않는다** — 정의가 있는 "
        "문서로 보낼 뿐이다(코퍼스 축자: 'REJECTED - the user has too many referral processes "
        "going on').")

# ★**상태 문구는 건드리지 않는다** (2026-08-10·`test_status_breakdown` 이 못박아 둔 계약).
#   한 번 그 문구에 검색 지시를 넣었다가 v010 에서 에이전트가 **상태 낱말로 검색**해
#   (`referral status IN_PROGRESS REJECTED`) 이유를 못 찾은 실측이 있다. 나는 x200 만 보고
#   그 지시를 되살렸다가 회귀에 걸렸다 — 라이브 실측이 격리 A/B 를 이긴다([[56]]).
#   ⇒ 이유를 나르는 자리는 **창 산수 문장 하나**이고, 아래 두 상수는 기록으로만 남긴다.
PAIRS = [("window_history_text", OLD_WINDOW, NEW_WINDOW)]


def _esc(s):
    """파일에 실제로 적혀 있는 형태(= JSON 이스케이프된 본문)."""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    staged = {}
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        before = json.loads(txt)                       # 검사용 원본
        out, hit = txt, 0
        for key, old, new in PAIRS:
            eo, en = _esc(old), _esc(new)
            if eo in out:
                if out.count(eo) != 1:
                    print("  중단: %s 에서 %s 문구가 %d번 나온다" % (rel, key, out.count(eo)))
                    return 1
                out = out.replace(eo, en)
                hit += 1
            elif en in out:
                print("  이미 적용됨: %s / %s" % (rel, key))
            else:
                print("  중단: %s 에서 %s 의 현행 문구를 못 찾았다" % (rel, key))
                return 1

        # 치환 결과가 여전히 올바른 JSON 이고, **바뀐 것이 그 두 값뿐**인지 검사한다.
        try:
            after = json.loads(out)
        except Exception as e:
            print("  중단: %s 치환 후 JSON 파손 (%r)" % (rel, e))
            return 1
        b2, a2_ = json.loads(json.dumps(before)), json.loads(json.dumps(after))
        for ms in (b2.get("ledger_metrics") or []):
            for key, old, new in PAIRS:
                if ms.get(key) in (old, new):
                    ms[key] = "<TEXT>"
        for ms in (a2_.get("ledger_metrics") or []):
            for key, old, new in PAIRS:
                if ms.get(key) in (old, new):
                    ms[key] = "<TEXT>"
        if json.dumps(b2, sort_keys=True) != json.dumps(a2_, sort_keys=True):
            print("  중단: %s 에서 그 두 문구 말고 다른 것도 바뀌었다" % rel)
            return 1
        got = [ms.get(k) for ms in (after.get("ledger_metrics") or []) for k, _o, _n in PAIRS
               if ms.get(k)]
        ok = all(any(g == n for g in got) for _k, _o, n in PAIRS)
        print("  %-40s 교체 %d건 · 새 문구 실재 %s" % (rel, hit, "OK" if ok else "**아님**"))
        if not ok:
            return 1
        staged[rel] = out

    vals = []
    for rel, out in staged.items():
        d = json.loads(out)
        vals.append(sorted(str(ms.get(k)) for ms in (d.get("ledger_metrics") or [])
                           for k, _o, _n in PAIRS if ms.get(k)))
    print("  두 층 문구 동일: %s" % ("OK" if len(set(map(tuple, vals))) == 1 else "**불일치**"))
    if len(set(map(tuple, vals))) != 1:
        return 1
    if not a.apply:
        print("\n(미적용 — 쓰려면 --apply)")
        return 0
    for rel, out in staged.items():
        io.open(os.path.join(HERE, rel), "w", encoding="utf-8", newline="").write(out)
        print("  wrote %s" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
