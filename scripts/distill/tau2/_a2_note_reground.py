# -*- coding: utf-8 -*-
"""A2 노트 재근거 — 포맷 보존 표적 치환 (2026-08-19·[[23]] 소급·[[24]] 3사본 동시)."""
import io
import json

OLD_DT = u" = ${delta_total:.2f}"
ANCH = u'"_note_": "2026-08-13 x288'
ADD_DT = (
    u'"_note_delta_total_removed_2026_08_19": "REMOVED (2026-08-19 · 커밋 b220745d 와 같은 사유). '
    u'`= ${delta_total:.2f}` 를 return_template 에서 제거했다. 요율 출처는 정책 축자라 [[23]] 는 '
    u'통과하지만 delta_total 은 **채점되는 인자 그 자체**다(task_073 gold amount 9.50/9.00/1.50 = 계좌별 net). '
    u'엔진이 채점되는 값을 만들어 건네면 formalize->calc 아키텍처가 아니라 그 위조판을 재게 된다([[62]]·[[03b]]). '
    u'남긴 것 = {details}(라인별 중간 사실) + 정책 축자 문구. 엔진 코드 불변(kwarg 미사용=거동 0). '
    u'⚠효과는 **미측정**이다.", '
)
NEW = {
    u"_note_prohibits": (
        u"★출처(정책 축자·[[23]] 재근거 2026-08-19): 이 금지의 유일 근거는 선언 안의 _quote 축자다 — "
        u"'Do not collect sensitive card details; the tool uses the identifiers provided by the user.' "
        u"금지는 순서가 아니라 한 문장이고, 따옴표 없는 금지는 엔진이 무시한다. "
        u"⚠구판 문면은 'gold 20건 중 last4 요구 0건' 을 근거처럼 앞세웠다 — 그것은 **사후 확인**이지 "
        u"설계 출처가 아니므로 문면을 교체했다([[23]] 소급). 성적 계수(x81)도 근거가 아니라 관측이다."
    ),
    u"_note_choice_grounding": (
        u"★출처(정책 축자·[[23]] 재근거 2026-08-19). ⚠구판은 **gold 를 세서 레버 강도를 정했다** "
        u"('gold 52건 중 47 실패·오선택 27 … gold 3건 ⇒ deny는 오차단이라 넛지 1회로 둔다') — "
        u"compute_ops 임계 30과 같은 부류의 [[23]] 위반이라 **폐기**했다. 정책 축자로 같은 결론이 나온다: "
        u"'Internal: Opening Personal Savings Accounts' — 'Capture the exact account_class string. "
        u"It must be the full official name ending with Account' · "
        u"'Set account_class to the exact official name provided by the customer'. "
        u"즉 정책이 요구하는 것은 **공식 명칭 + 손님 확인**이지 *우리가 회수한 문서 안에 있음* 이 아니다 "
        u"⇒ 기존 술어가 **정책보다 좁았다**. account_class 는 열린 문자열이 아니라 형식이 닫혀 있다. "
        u"효과는 **미측정**이다."
    ),
}
FILES = ("a2/banking_knowledge.gate.json",
         "a2/banking_knowledge.specific.json",
         "a2/split/banking_knowledge.core.json")


def end_of_json_string(s, i):
    """s[i] == '\"' (여는 따옴표). 닫는 따옴표 index 반환."""
    j = i + 1
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == '"':
            return j
        j += 1
    raise ValueError("unterminated")


def replace_note(s, key, value):
    n = 0
    needle = '"%s"' % key
    pos = 0
    while True:
        k = s.find(needle, pos)
        if k < 0:
            break
        c = s.find(":", k + len(needle))
        q = s.find('"', c + 1)
        e = end_of_json_string(s, q)
        s = s[:q] + json.dumps(value, ensure_ascii=False) + s[e + 1:]
        pos = q + len(json.dumps(value, ensure_ascii=False))
        n += 1
    return s, n


def main():
    for f in FILES:
        s = io.open(f, encoding="utf-8").read()
        ndt = s.count(OLD_DT)
        s = s.replace(OLD_DT, "")
        nn = 0
        if ANCH in s:
            s = s.replace(ANCH, ADD_DT + ANCH, 1)
            nn = 1
        counts = {}
        for k, v in NEW.items():
            s, c = replace_note(s, k, v)
            counts[k] = c
        json.loads(s)
        io.open(f, "w", encoding="utf-8").write(s)
        print("%-46s dt=%d dtnote=%d %s  JSON OK" % (f, ndt, nn, counts))


main()
