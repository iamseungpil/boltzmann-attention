# -*- coding: utf-8 -*-
"""X4 E-FLIP — P2-b 프록시 술어의 변이-flip 실측 (배터리 P3·EXPERIMENT_PLAN §1-X4).

원 구현 축자 이식(출처 명시) + 의미-라벨 수작업 변이 스위트(합성 궤도·[[18]]).
측정: miss(참-조건인데 미발화)·false-fire(거짓-조건인데 발화). Wilson 하한>0 = 열림 [M].
"""
import math, re

def wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d)

results = []

# ── P-A: reask_signals (t2_gate_patch.py:895/963 · a2 banking_knowledge.gate.json) ──
SIGNALS = ['last 4', 'last four', 'last-4', 'four digits', '4 digits', 'correct last 4']
def pred_reask(text):
    t = text.lower()
    return any(s in t for s in SIGNALS)

reask_true_paraphrase = [  # 참 재요청·신호 회피 표현 (fire해야 정상)
    "Could you read me the ending digits on the card?",
    "What are the final four numbers printed on your card?",
    "Please give me the trailing digits of your card number.",
    "May I have the closing digits of the card?",
    "Tell me the digits at the end of the card, please.",
    "What does your card number end with?",
    "Could you share the tail end of the card number?",
    "I need the terminal digits of that card to proceed.",
]
reask_false_with_signal = [  # 재요청 아님(확인/부정/일반)·신호 포함 (fire하면 오탐)
    "Great - the last 4 digits you provided match our records.",
    "I have verified the last four already, no need to repeat them.",
    "You don't need to tell me the last 4 again.",
    "The last-4 on file is confirmed, thank you.",
    "Our secure system masks all but the last 4 digits automatically.",
    "Please never share the last 4 digits with anyone who calls you.",
]
reask_control = [
    "Could you tell me the last 4 digits of your card?",
    "Please provide the last four digits of the card.",
]
miss = sum(1 for t in reask_true_paraphrase if not pred_reask(t))
ff = sum(1 for t in reask_false_with_signal if pred_reask(t))
ctrl = sum(1 for t in reask_control if pred_reask(t))
results.append(("reask_signals(HAVE_VALUE/VALUE_ACQUIRE)",
                miss, len(reask_true_paraphrase), ff, len(reask_false_with_signal),
                ctrl, len(reask_control)))

# ── P-B: UNKNOWN_REPEAT (t2_gate_patch.py:5139 — `n in am.content`·대소문자 민감) ──
REJECTED = "check_rewards_program_9921"
def pred_unkrep(text):
    return REJECTED in text

unk_false_mention = [  # 이름 언급하지만 재지시 아님 (fire=오탐·016 기전)
    "I will stop asking you to run check_rewards_program_9921 - it does not exist.",
    "Apologies: check_rewards_program_9921 was invalid; let me search for the correct tool.",
    "Please ignore my earlier mention of check_rewards_program_9921.",
    "The name check_rewards_program_9921 was rejected, so we will use another path.",
]
unk_true_evade = [  # 참 재지시·표기 변이 (miss)
    "Please run check rewards program 9921 from your side now.",
    "Try running CHECK_REWARDS_PROGRAM_9921 on your end.",
    "Run the tool named Check_Rewards_Program_9921 again, please.",
]
unk_control = ["Please run check_rewards_program_9921 now."]
ff2 = sum(1 for t in unk_false_mention if pred_unkrep(t))
miss2 = sum(1 for t in unk_true_evade if not pred_unkrep(t))
ctrl2 = sum(1 for t in unk_control if pred_unkrep(t))
results.append(("UNKNOWN_REPEAT(name-substring)",
                miss2, len(unk_true_evade), ff2, len(unk_false_mention),
                ctrl2, len(unk_control)))

# ── P-C: EPLAN 수량 파서 (t2_eplan_patch.py:55-92 축자) ──
_QTY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "both": 2, "couple": 2, "pair": 2}
_QTY_DIGIT_RE = re.compile(r"\b([1-9]|10)\b")
_QTY_WORD_RE = re.compile(r"\b(%s)\b" % "|".join(_QTY_WORDS), re.I)
_TIME_UNIT = r"(?:second|minute|hour|day|week|month|year)s?"
_NUM_TOK = r"(?:[0-9]{1,3}|%s)" % "|".join(_QTY_WORDS)
_QTY_RANGE_RE = re.compile(r"\b%(n)s\s*(?:to|-|–|—)\s*%(n)s\b" % {"n": _NUM_TOK}, re.I)
_QTY_TIME_RE = re.compile(
    r"\b%(n)s(?:\s+of)?(?:\s+(?:business|working|calendar))?\s+%(u)s\b"
    % {"n": _NUM_TOK, "u": _TIME_UNIT}, re.I)

def pred_qty(text):
    """수량>=2 신호 검출 (원 구현 취지: 시간/범위 절제 후 수사 탐지)."""
    t = _QTY_RANGE_RE.sub(" ", text)
    t = _QTY_TIME_RE.sub(" ", t)
    for m in _QTY_WORD_RE.finditer(t):
        if _QTY_WORDS[m.group(1).lower()] >= 2:
            return True
    for m in _QTY_DIGIT_RE.finditer(t):
        if int(m.group(1)) >= 2:
            return True
    return False

qty_true_paraphrase = [  # 참 다중-수량 요청·소사전 회피 (miss)
    "Cancel the cards ending 1111 and the one ending 2222.",
    "Close my checking as well as my savings account.",
    "I'd like the remaining cards closed too, not just this one.",
    "Please cancel each and every card on my profile.",       # 'every'는 QTY 소사전 밖
    "Dispute the charge from Monday and also the one from Friday.",
]
qty_false_with_signal = [  # 다중-수량 아님·수사 포함 (fire=오탐)
    "Send 5 to my son from my checking account.",              # 금액
    "My apartment is unit 3 on Pine Street.",                  # 주소
    "I called you two times yesterday about this.",            # 횟수(대상 아님)
    "Rate it 9 out of 10, honestly.",
]
qty_control = ["Cancel both of my credit cards.", "Close two of my accounts."]
miss3 = sum(1 for t in qty_true_paraphrase if not pred_qty(t))
ff3 = sum(1 for t in qty_false_with_signal if pred_qty(t))
ctrl3 = sum(1 for t in qty_control if pred_qty(t))
results.append(("EPLAN qty-parser", miss3, len(qty_true_paraphrase),
                ff3, len(qty_false_with_signal), ctrl3, len(qty_control)))

print("%-38s %10s %12s %10s" % ("predicate", "miss", "false-fire", "control"))
for name, m, mn, f, fn, c, cn in results:
    flips = m + f
    n = mn + fn
    print("%-38s %4d/%-4d %6d/%-4d %8d/%-3d  flip=%d/%d WilsonLB=%.2f"
          % (name, m, mn, f, fn, c, cn, flips, n, wilson_lb(flips, n)))
