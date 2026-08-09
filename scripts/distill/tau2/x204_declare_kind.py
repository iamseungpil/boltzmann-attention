# -*- coding: utf-8 -*-
r"""x204 — A2 `eligible` 에 **종류 선택 선언** 두 줄을 넣는다 (문자열 삽입·엔진 0·유료 0).

`kind_field` = A3 행에서 종류를 읽을 필드 이름(`x203` 이 붙였다) ·
`kind_prompt` = 손님의 말에서 **어느 종류인지 LLM 이 고르게** 하는 문구.

문구는 **x201 이 실제로 잰 것 그대로**다(G_llm 8/8 · 종류 선택 8/8 정확). 재보지 않은 문구로
바꾸지 않는다.

⚠`settings.json` 은 손으로 정렬된 포맷이라 JSON 재직렬화가 파일을 통째로 다시 쓴다. 그래서
  **문자열 삽입**만 하고, 삽입 후 JSON 으로 다시 읽어 *그 두 키 말고 바뀐 게 없는지* 검사한다.
⚠[[24]]: `settings.json` 과 `gate.json` 두 층에 같은 값을 넣는다.

실행: py -3 x204_declare_kind.py [--apply]
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

ANCHOR = '"tally_from": "get_referrals_by_user",'
KIND_FIELD = "kind"
KIND_PROMPT = ("These are the product groups on record:\n{kinds}\n\nConversation:\n{text}\n\n"
               "Which ONE group does the product the customer is asking about belong to? Reply "
               "with that group name exactly and nothing else.")
NOTE = ("x201(격리·n=8·32B): 통과 표에 개인 체킹 5 + 사업자 카드 6 + 카드 3 이 함께 실리면 "
        "모델이 카드의 단일 최대 수를 집는다(A_iso 0/8). **전달 팔을 먼저 쟀고**(E_hint = 표에 "
        "한 줄로 무엇을 묻는지 말해 주기) 그것도 0/8 이라 필터가 정당해졌다(⛔0 ②). 종류로 거른 "
        "표 8/8 · LLM 이 종류를 고르는 2단 구성 8/8(종류 선택 8/8 정확). 종류 값은 A3 행이 이미 "
        "인용하고 있는 출처 문서군에서 빌드 시점에 유도한다(x203) — 지어낸 어휘 0. 못 고르면 "
        "아무것도 거르지 않고, 종류를 모르는 주어도 남는다(모름 != 탈락).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    staged = {}
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        before = json.loads(txt)
        # ⚠파일 전체에서 `"kind_field"` 를 찾으면 오탐한다 — `gate.json` 에는 무관한 용도의
        #   `"kind_field": "exclusion_pin_kind"` 가 이미 있다. **이 선언 안에서만** 본다.
        if any((s.get("eligible") or {}).get("kind_field")
               for s in (before.get("ledger_metrics") or [])):
            print("  이미 적용됨: %s" % rel)
            staged[rel] = txt
            continue
        if txt.count(ANCHOR) != 1:
            print("  중단: %s 에서 삽입 지점이 %d 번 나온다" % (rel, txt.count(ANCHOR)))
            return 1
        i = txt.index(ANCHOR)
        pad = " " * (i - txt.rindex("\n", 0, i) - 1)      # 그 줄의 들여쓰기를 그대로 쓴다
        add = ("%s\n%s%s: %s,\n%s%s: %s,\n%s%s: %s,"
               % (ANCHOR,
                  pad, '"kind_field"', json.dumps(KIND_FIELD, ensure_ascii=False),
                  pad, '"kind_prompt"', json.dumps(KIND_PROMPT, ensure_ascii=False),
                  pad, '"_note_kind"', json.dumps(NOTE, ensure_ascii=False)))
        out = txt.replace(ANCHOR, add, 1)
        try:
            after = json.loads(out)
        except Exception as e:
            print("  중단: %s 삽입 후 JSON 파손 (%r)" % (rel, e))
            return 1
        # 바뀐 것이 **그 세 키뿐**인지 검사한다
        b2, a2_ = json.loads(json.dumps(before)), json.loads(json.dumps(after))
        for spec in (a2_.get("ledger_metrics") or []):
            for k in ("kind_field", "kind_prompt", "_note_kind"):
                (spec.get("eligible") or {}).pop(k, None)
        if json.dumps(b2, sort_keys=True) != json.dumps(a2_, sort_keys=True):
            print("  중단: %s 에서 그 세 키 말고 다른 것도 바뀌었다" % rel)
            return 1
        got = [(s.get("eligible") or {}) for s in (after.get("ledger_metrics") or [])
               if (s.get("eligible") or {}).get("kind_field")]
        ok = len(got) == 1 and got[0].get("kind_prompt") == KIND_PROMPT
        print("  %-40s 삽입 · 선언 실재 %s" % (rel, "OK" if ok else "**아님**"))
        if not ok:
            return 1
        staged[rel] = out

    vals = [json.dumps([(s.get("eligible") or {}).get("kind_prompt")
                        for s in (json.loads(v).get("ledger_metrics") or [])], ensure_ascii=False)
            for v in staged.values()]
    print("  두 층 선언 동일: %s" % ("OK" if len(set(vals)) == 1 else "**불일치**"))
    if len(set(vals)) != 1:
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
