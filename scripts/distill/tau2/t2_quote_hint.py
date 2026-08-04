"""P5 — 접지 가드가 인용을 반려할 때, **값이 실재하면** 원장의 표기를 지목한다.

task_046이 자기 두 trial로 원인을 확정했다. 같은 태스크·같은 원장·같은 참인 값인데:

  t0  balance_source="current_balance: $0.00"        (패러프레이즈)  → 드롭 → 동일 호출 7회 → 11턴 소실
  t1  balance_source="New Credit Card Balance: $0.00" (원장 축자)    → 통과

즉 **값의 진위가 아니라 인용 형태**가 갈랐다. 그런데 반려 문구는 "source quote not found"까지만
말하고 원장이 그 값을 **뭐라고 부르는지**는 말하지 않아, 모델은 같은 문자열로 재시도한다.

★지목은 무조건 하지 않는다 (리뷰 지적·C226 spoonfeed 계보). 엔진이 원장 표기를 그냥 알려주면
모델이 복사해 통과하고 가드가 형식으로 전락한다. **모델이 제시한 값이 원장에 실재할 때만** 지목한다:

  값이 원장에 있다  → 그 값을 담은 줄을 최대 2개 돌려준다 (고칠 것은 인용 형태뿐)
  값이 원장에 없다  → 지목 없이 종전대로 드롭 (날조는 여전히 못 뚫는다)

이러면 진위 검사는 손대지 않고 [[10]] 분담(생성=LLM·검증=엔진)도 유지된다.
"""

import os
import re

MAX_LINES = 2
_WS = re.compile(r"\s+")


def enabled():
    return os.environ.get("T2_QUOTE_HINT") == "1"


def _norm(s):
    return _WS.sub(" ", str(s or "")).strip()


def _needles(value):
    """값이 원장에 나타날 수 있는 표면형. 숫자는 소수점 표기가 흔들리므로 몇 가지를 본다."""
    v = _norm(value)
    if not v:
        return []
    out = {v}
    try:
        f = float(v.replace(",", "").replace("$", ""))
        out.add(("%f" % f).rstrip("0").rstrip("."))
        out.add("%.2f" % f)
        out.add("%,.2f" % f if False else format(f, ",.2f"))
    except Exception:
        pass
    return [x for x in out if len(x) >= 1]


def lines_for(value, corpus_texts):
    """값을 담은 원장 줄들(최대 MAX_LINES). 없으면 빈 리스트."""
    needles = _needles(value)
    if not needles:
        return []
    hits = []
    for t in corpus_texts or []:
        for ln in str(t or "").splitlines():
            l = _norm(ln)
            if not l or len(l) > 200:
                continue
            if any(n in l for n in needles):
                if l not in hits:
                    hits.append(l)
                if len(hits) >= MAX_LINES:
                    return hits
    return hits


def hint(value, corpus_texts):
    """반려 문구에 덧붙일 지목. 값이 원장에 없으면 빈 문자열(거동 불변)."""
    if not enabled():
        return ""
    hits = lines_for(value, corpus_texts)
    if not hits:
        return ""
    # 이 레버도 반려 문구에 덧붙이기만 해서 태그가 없다 — 발화 증명을 남긴다([[30]]·x43).
    try:
        from t2_lever_beat import beat as _beat
        _beat("T2_QUOTE_HINT", "%d line(s)" % len(hits))
    except Exception:
        pass
    return (" Your value IS present in the records; only the quoted string differs from how "
            "the records write it. The records say: %s"
            % " | ".join('"%s"' % h for h in hits))


def selftest():
    os.environ["T2_QUOTE_HINT"] = "1"
    ledger = ["Payment processed successfully! - Payment Amount: $125.00\n"
              "- New Credit Card Balance: $0.00 - Status: OK",
              "Found 2 record(s) in 'credit_card_accounts': 1. Record ID: cc_x_plat"]

    h = hint("0.00", ledger)
    assert "New Credit Card Balance: $0.00" in h, h
    print("  ok   값이 원장에 있으면 그 줄을 지목한다 (046 t0 해소)")

    assert hint("9999.99", ledger) == ""
    print("  ok   값이 원장에 없으면 지목 없음 = 날조 차단 유지")

    long_ledger = ["\n".join("balance is 0.00 line %d" % i for i in range(10))]
    assert len(hint("0.00", long_ledger).split(" | ")) <= MAX_LINES
    print("  ok   지목 줄 수 상한 %d" % MAX_LINES)

    os.environ["T2_QUOTE_HINT"] = "0"
    assert hint("0.00", ledger) == ""
    print("  ok   플래그 OFF면 문구 변화 0")

    os.environ["T2_QUOTE_HINT"] = "1"
    assert hint("", ledger) == "" and hint("0.00", []) == ""
    print("  ok   빈 값·빈 코퍼스 = 무발화")
    print("PASS (5/5)")


if __name__ == "__main__":
    selftest()
