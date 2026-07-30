# -*- coding: utf-8 -*-
"""X2 v2 — 미접지 파생값의 재계산 오류율 + 종료사유 층화 (EXPERIMENT_PLAN §1-X2 잔여).

핵심: "미접지"(원장 부재)와 "오류"(값이 틀림)는 다르다. 미접지 금액 중
  ① 산술식 동반형("A - B = C"·"total of X + Y = Z")은 **식을 파싱해 재계산**하여 정오 판정
  ② 원장의 두 값의 합/차와 일치하는지 탐색(식 없는 경우·페어 탐색)
을 결정론으로 수행. 판정 불가분은 'undecided'로 남긴다(과대주장 금지).
+ 종료사유(termination_reason)별 미접지율 층화 = 절단 sim의 분모 왜곡 점검.

사용: py -3 x2v2_recompute_audit.py <examples.jsonl> <results_glob...>
"""
import gzip, json, re, sys, glob, os, io, collections, itertools

MONEY_NUM = r"\$?\s?(\d[\d,]*(?:\.\d{1,2})?)"
# ★2026-07-30 정정: 초판 2-항 EXPR이 3-항 합계를 마지막 두 항으로 오파싱 → wrong 5/5 전부
#   오탐(실측 확인: 155.33+54.85+269.16=479.34 등 모델이 옳았음). N-항 식으로 확장.
#   식: A op B (op C)* = Z  — 동일 연산자 체인만(혼합 우선순위 회피·보수적).
NTERM = re.compile(
    r"\$?\s?\d[\d,]*(?:\.\d{1,2})?(?:\s*[-+*x×]\s*\$?\\?\(?\s?\d[\d,]*(?:\.\d{1,2})?)+"
    r"\s*=\s*-?\s?\$?\s?\d[\d,]*(?:\.\d{1,2})?")
TOKEN = re.compile(r"[-+*x×]|\d[\d,]*(?:\.\d{1,2})?")


def f(x):
    return float(str(x).replace(",", "").replace("$", "").strip())


def norm_variants(v):
    s = ("%.2f" % v)
    out = {s, s.rstrip("0").rstrip("."), "%.1f" % v}
    if abs(v - round(v)) < 1e-9:
        out.add(str(int(round(v))))
    return out


def ledger_numbers(msgs, upto):
    """upto 이전 도구 출력·유저 발화의 통화 수치 집합(float)."""
    nums = set()
    for m in msgs[:upto]:
        if m.get("role") not in ("tool", "user"):
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        for mm in re.finditer(MONEY_NUM, c):
            try:
                nums.add(round(f(mm.group(1)), 2))
            except Exception:
                pass
    return nums


def _eval_chain(toks):
    """동일-연산자 체인 평가(혼합 시 None). toks = [num, op, num, op, num...]."""
    ops = {t for i, t in enumerate(toks) if i % 2 == 1}
    if len(ops) != 1:
        return None
    op = ops.pop()
    vals = [f(t) for i, t in enumerate(toks) if i % 2 == 0]
    acc = vals[0]
    for v in vals[1:]:
        if op == "+":
            acc += v
        elif op == "-":
            acc -= v
        else:
            acc *= v
    return acc


def verdict(amount, snippet, led):
    """미접지 금액 1건의 정오 판정: correct / wrong / undecided."""
    target = round(f(amount), 2)
    # ① 스니펫 내 명시 산술식 (N-항)
    for mm in NTERM.finditer(snippet):
        toks = TOKEN.findall(mm.group(0))
        if len(toks) < 4:
            continue
        rhs = f(toks[-1])
        if round(abs(rhs), 2) != target:
            continue
        got = _eval_chain(toks[:-1])
        if got is None:
            return ("undecided", "mixed-operators")
        operands = [round(f(t), 2) for i, t in enumerate(toks[:-1]) if i % 2 == 0]
        if all(o in led for o in operands):
            return ("correct" if round(abs(got), 2) == target else "wrong", "expr")
        return ("undecided", "expr-operands-not-in-ledger")
    # ② 원장 두 값의 합/차 탐색 (조합 폭발 방지: 원장 60개 이하일 때만)
    if 2 <= len(led) <= 60:
        for a, b in itertools.combinations(sorted(led), 2):
            for got in (a + b, abs(a - b)):
                if round(got, 2) == target:
                    return ("correct", "pair-search")
    return ("undecided", "no-derivation-found")


def main(ex_path, globs):
    ex = [json.loads(l) for l in io.open(ex_path, encoding="utf-8")]
    files = []
    for g in globs:
        files.extend(glob.glob(g))
    cache = {}

    def load(run):
        if run not in cache:
            path = next(p for p in files if os.path.basename(p) == run)
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                cache[run] = json.load(fh)
        return cache[run]

    agg = collections.Counter()
    rows = []
    for e in ex:
        try:
            d = load(e["run"])
        except StopIteration:
            continue
        sim = next((s for s in d["simulations"] if s["task_id"] == e["task"]), None)
        if sim is None:
            continue
        msgs = sim["messages"]
        led = ledger_numbers(msgs, e["turn_idx"])
        v, how = verdict(e["amount"], e["snippet"], led)
        agg[v] += 1
        agg["how_" + how] += 1
        rows.append((e["run"][:22], e["task"], e["amount"], v, how, e["snippet"][:70]))
    print("=== 재계산 판정 (%s·n=%d) ===" % (os.path.basename(ex_path), len(rows)))
    for k in ("correct", "wrong", "undecided"):
        print("  %-10s %4d (%.1f%%)" % (k, agg[k], 100.0 * agg[k] / max(1, len(rows))))
    print("  방법별:", {k[4:]: v for k, v in agg.items() if k.startswith("how_")})
    for r in [r for r in rows if r[3] == "wrong"][:12]:
        print("  WRONG:", r[0], r[1], "$" + r[2], "|", r[5].replace("\n", " "))

    # 종료사유 층화
    strat = collections.Counter()
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
        for s in d.get("simulations", []):
            tr = s.get("termination_reason") or "?"
            strat[tr] += 1
    print("=== 종료사유 분포(코퍼스 전체) ===", dict(strat))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
