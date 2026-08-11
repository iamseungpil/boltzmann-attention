# -*- coding: utf-8 -*-
r"""x249 — 검색 에이전트의 **문구 두 개**를 A2 에 선언한다 (빌드 시점 · 유료 0 · 모델 0).

## 왜 (배선의 마지막 조각 · [[16]])

엔진은 문구를 짓지 않는다. `t2_search.formalize_group` / `decide_from_docs` 는 A2 가 준 템플릿을
포맷할 뿐이고, 없으면 **침묵한다**. 그래서 두 키를 선언한다:

    group_prompt          손님의 말 → 문서군 하나 (닫힌 집합 = A3 `doc_index` 키)
    doc_decide_prompt     요청 + 재료(격리) → 이름 하나
    decided_by_docs_text  그 답을 메인에 싣는 문장 (099/100 의 `decided_text` 와 같은 규약)

## 문구가 왜 이 모양인가 (측정된 구성 그대로 · x248·n=8·두 축 8/8)

  · **요청이 머리**에 온다 — 뒤에 두면 형식화가 축을 잘못 골랐다(C417⒠: savings 요청에
    checking 군). 라이브에서도 그 턴의 요청이 문맥의 끝이다.
  · **재료는 축자**로만 싣고, 뺀 것은 이유와 함께 이미 재료 안에 있다(C327).
  · **이름만 답하라** — 엔진이 값을 계산하거나 고르지 않는다(⛔0 ④).

## 도메인 어휘 0

두 문구 어디에도 은행 어휘가 없다. 군 이름·상품 이름은 **A3 색인과 문서 축자**에서 런타임에
들어온다. 그래서 도메인이 바뀌어도 이 문구는 그대로다([[05]] 고정/가변 경계).

⚠두 층에 바이트 동일로 쓴다([[24]]).

실행: py -3 x249_a3_search_prompts.py [--apply]
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
LAYERS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json"]

GROUP_PROMPT = (
    "A customer service agent needs to look up policy documents for the request below.\n"
    "Which ONE of these document groups covers what the customer is asking about?\n"
    "Groups:\n{groups}\n\n"
    "Request and conversation:\n{text}\n\n"
    "Reply with the group name only, or 'none' if none of them covers it."
)

DECIDED_TEXT = (
    "\nA separate check was run on the policy documents on record, with this request and nothing "
    "else from this conversation in front of it. Documents whose stated period does not include "
    "today were excluded before it answered. It answers: {choice}."
)

DECIDE_PROMPT = (
    "{ask}\n\n"
    "Policy documents on record (verbatim; anything excluded is listed with the reason):\n"
    "{material}\n\n"
    "Which ONE option should the agent recommend? Answer with the name only."
)

NOTE = (
    "★출처 = 이 문구들은 **도메인 어휘가 0** 이다(은행 단어 없음). 군 이름·상품 이름은 A3 색인과 "
    "문서 축자에서 런타임에 들어오므로 도메인이 바뀌어도 그대로다([[05]] 고정/가변). 구성은 "
    "**측정된 그대로**다(x248·071 실물·n=8·두 축 8/8): 요청이 **머리**에 오고(뒤에 두면 형식화가 "
    "축을 잘못 골랐다·C417⒠), 재료는 축자이며 뺀 것은 이유와 함께 재료 안에 있고(C327), 답은 "
    "**이름 하나**다(엔진이 고르지 않는다·⛔0 ④). 소비자 = `t2_search.formalize_group` / "
    "`decide_from_docs` — 키가 없으면 두 함수는 **침묵한다**."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print("group_prompt %d자 · doc_decide_prompt %d자" % (len(GROUP_PROMPT), len(DECIDE_PROMPT)))
    for name, t in (("group_prompt", GROUP_PROMPT), ("doc_decide_prompt", DECIDE_PROMPT),
                    ("decided_by_docs_text", DECIDED_TEXT)):
        print("\n--- %s\n%s" % (name, t))
    if not a.apply:
        print("\n(--apply 없이는 쓰지 않는다)")
        return 0
    for rel in LAYERS:
        p = os.path.join(HERE, rel)
        txt = io.open(p, encoding="utf-8").read()
        doc = json.loads(txt)
        if json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "") \
                != txt:
            print("  중단: %s 재직렬화가 바이트 동일하지 않다" % rel)
            return 1
        po = doc["policy_ontology"]
        po["group_prompt"] = GROUP_PROMPT
        po["doc_decide_prompt"] = DECIDE_PROMPT
        po["decided_by_docs_text"] = DECIDED_TEXT
        po["_note_search_prompts"] = NOTE
        out = json.dumps(doc, ensure_ascii=False, indent=1) + ("\n" if txt.endswith("\n") else "")
        io.open(p, "w", encoding="utf-8", newline="").write(out)
        print("  기록: %s" % rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
