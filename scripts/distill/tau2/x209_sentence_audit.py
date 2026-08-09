# -*- coding: utf-8 -*-
r"""x209 — 우리 층이 내보내는 **문장 전수 목록**과 실발화 빈도 (유료 0 · 읽기 전용).

## 왜 (사용자 지시 2026-08-10)

> *"문구들이 사이드이펙트를 내는 거 같은데, 문구를 최소화하고 LLM 이 알아서 하게 하는 건
>  어떤가? 문구로 이것저것 지시하는 게 결국 프로그래밍과 같은 거 아닌가? 우리가 하는 것은
>  LLM 이 잘 못하는 부분을 보강하는 거지 LLM 을 일일이 지시하는 게 아니다."*

하루치 실측이 그 방향을 지지한다 — **사실을 더하면 좋아지고 지시를 더하면 나빠졌다**:

  사실 추가:  A3 예치 문턱(`B_sum` 0/8→8/8) · 종류 필터(0/8→8/8) · 창 날짜 산수(`C_calc` 0/8)
  지시 추가:  창-산수 부정 꼬리말(이유 0/8) · 상태 문구의 검색 지시(상태 낱말 검색 실패) ·
              재도출 NONE 조항(098·100 침묵 0/8) · 기권 옵션(098 6/8 붕괴) · 표 뒤 꼬리말(5/5→2/5)

그래서 **문장을 하나씩 빼서 재는 감사**가 필요하고, 이 파일은 그 1단계다 — 목록과 빈도.
판단(사실/지시)과 제거 A/B 는 이 목록 위에서 사람이 한다. 여기서 자동 분류하지 않는다:
문자열로 지시 여부를 판정하는 것 자체가 또 하나의 패턴매칭이다([[59]]).

## 무엇을 인쇄하나

  ⒜ A2 가 선언한 **모든 문장 키**(`*_text` · `*_prompt`)와 본문
  ⒞ 각 문장이 **최근 런에서 실제로 나간 횟수**(사이드카) — 안 나가는 문장은 감사 대상이 아니라
     **死배선 후보**다(그것대로 따로 다뤄야 한다)
  ⒟ 문장 안의 **명령형 신호** 개수 — 분류를 대신하지 않고 **읽는 순서를 정하는 데만** 쓴다

실행 (리모트): python x209_sentence_audit.py [tag ...]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from gate_interpreter import load_domain_a2                      # noqa: E402

LOGS = os.environ.get("T2_LOGS", "/home/woori/scratch/logs")
# 명령형 신호 — **분류가 아니라 정렬용**이다(뒤에 사람이 읽는다).
IMPER = re.compile(r"\b(retrieve|search|do not|don't|must|should|before |first |reply |answer "
                   r"with|determine|check |verify|say |state |ask )", re.I)


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + "/%d" % i)
    elif isinstance(o, str):
        yield path, o


def main():
    tags = sys.argv[1:] or ["bank_a3fill_20260810a"]
    a2 = load_domain_a2("banking_knowledge") or {}
    sents = []
    for path, val in walk(a2):
        key = path.rsplit("/", 1)[-1]
        if not (key.endswith("_text") or key.endswith("_prompt")):
            continue
        if key.startswith("_note"):
            continue
        sents.append((path, val))
    print("A2 가 선언한 문장 %d개\n" % len(sents))

    # 사이드카에서 실발화 — 문장의 **고정 접두**로 센다(자리표시자 앞까지)
    fired = collections.Counter()
    total = 0
    for tag in tags:
        p = os.path.join(LOGS, "fb_%s.jsonl" % tag)
        if not os.path.exists(p):
            print("  ⚠사이드카 없음: %s" % p)
            continue
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            body = str(o.get("text") or o.get("body") or "")
            if not body:
                continue
            total += 1
            for path, val in sents:
                head = re.split(r"\{", val, 1)[0].strip()
                if len(head) >= 25 and head[:60] in body:
                    fired[path] += 1
    print("사이드카 항목 %d개 · 태그 %s\n" % (total, ", ".join(tags)))

    rows = sorted(sents, key=lambda kv: (-fired[kv[0]], -len(IMPER.findall(kv[1]))))
    print("%-52s %6s %5s  %s" % ("키", "발화", "명령형", "머리말"))
    print("-" * 110)
    for path, val in rows:
        head = " ".join(val.split())[:52]
        print("%-52s %6d %5d  %s" % (path[-52:], fired[path], len(IMPER.findall(val)), head))

    print("\n\n=== 본문 (발화 많은 순) ===")
    for path, val in rows:
        if fired[path] == 0:
            continue
        print("\n### %s   (발화 %d · 명령형 %d)" % (path, fired[path], len(IMPER.findall(val))))
        print(" ".join(val.split()))

    print("\n※ 발화 0 = 감사 대상이 아니라 **死배선 후보**다 — 채널을 먼저 확인하라([[55]]).")
    print("※ 명령형 수는 **읽는 순서**일 뿐 분류가 아니다. 사실/지시 판정은 사람이 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
