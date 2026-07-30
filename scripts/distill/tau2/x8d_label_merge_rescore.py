"""규약 결함이 acts_exact 를 얼마나 깎았나 — 라벨 병합 하 재채점 (무료·기존 432행).

가설: `ASK`/`REQUEST` 경계가 결정론적이지 않아(사용자 지적: "transfer 가 ask 아닌가")
오류로 계상된 것이 있다. 그 경계를 **없앤 채** 재채점하면 그 몫이 드러난다.
같은 방식으로 `CLOSE`/`CONTROL`(종결 계열)도 시험한다.
"""
import json
from collections import Counter

D = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
rows = [json.loads(l) for l in open(D + r"\x8_triage_rows.jsonl", encoding="utf-8")
        if "error" not in l]
cell = {}
for r in rows:
    cell.setdefault((r["arm"], r["sample_id"]), r)


def merge(acts, mapping):
    return frozenset(mapping.get(a, a) for a in acts)


SCHEMES = {
    "원본(규약 그대로)": {},
    "ASK≡REQUEST 병합": {"ASK": "REQ", "REQUEST": "REQ"},
    "CLOSE≡CONTROL 병합": {"CLOSE": "END", "CONTROL": "END"},
    "둘 다 병합": {"ASK": "REQ", "REQUEST": "REQ", "CLOSE": "END", "CONTROL": "END"},
}

print("=" * 76)
print("라벨 병합 하 acts_exact 재채점 — 규약 모호성이 깎은 몫")
print("=" * 76)
print(f"{'scheme':22s} {'A':>6s} {'C':>6s} {'Actx':>6s}   (n=48 each)")
base = {}
for name, mp in SCHEMES.items():
    line = []
    for arm in ["A", "C", "Actx"]:
        rs = [v for (a, s), v in cell.items() if a == arm]
        ok = sum(1 for r in rs
                 if merge(r["gold_acts"], mp) == merge(r["pred_acts"], mp))
        line.append(ok / len(rs))
        if name == "원본(규약 그대로)":
            base[arm] = ok / len(rs)
    print(f"{name:22s} " + " ".join(f"{x:6.2f}" for x in line)
          + ("" if name == "원본(규약 그대로)"
             else "   Δ=" + " ".join(f"{x - base[a]:+.2f}"
                                    for x, a in zip(line, ['A', 'C', 'Actx']))))

print()
print("=" * 76)
print("다중-act 페널티는 병합 후에도 남나 (게이트 재확인)")
print("=" * 76)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


for name, mp in [("원본", {}), ("둘 다 병합", SCHEMES["둘 다 병합"])]:
    rs = [v for (a, s), v in cell.items() if a == "A"]
    tab = {1: [0, 0], 2: [0, 0]}
    for r in rs:
        g = merge(r["gold_acts"], mp)
        k = 1 if len(g) == 1 else 2
        tab[k][1] += 1
        tab[k][0] += (g == merge(r["pred_acts"], mp))
    (so, sn), (mo, mn) = tab[1], tab[2]
    sl, sh = wilson(so, sn)
    ml, mh = wilson(mo, mn)
    print(f"{name:12s} 단일 {so}/{sn}={so / sn if sn else 0:.2f}[{sl:.2f},{sh:.2f}]  "
          f"다중 {mo}/{mn}={mo / mn if mn else 0:.2f}[{ml:.2f},{mh:.2f}]  "
          f"CI중첩={'예' if not (sl > mh or ml > sh) else '아니오'}")

print()
print("=" * 76)
print("ESCALATE 를 별도 라벨로 뗄 때 gold 재분포 (입도 검토용·모델 재채점 아님)")
print("=" * 76)
import re
samp = {json.loads(l)["sample_id"]: json.loads(l)
        for l in open(D + r"\x8_sample_utterances.jsonl", encoding="utf-8")}
gold = {json.loads(l)["sample_id"]: json.loads(l)
        for l in open(D + r"\x8_gold_labels.jsonl", encoding="utf-8")}
P = re.compile(r"transfer|escalat|human agent", re.I)
req_total = sum(1 for v in gold.values() if "REQUEST" in v["acts"])
req_esc = sum(1 for k, v in gold.items()
              if "REQUEST" in v["acts"] and P.search(samp[k]["text"]))
print(f"REQUEST gold {req_total}건 중 이관성 {req_esc}건 = {100 * req_esc / req_total:.0f}%")
print(f"⇒ ESCALATE 분리 시 REQUEST {req_total}→{req_total - req_esc}, ESCALATE {req_esc} 신설")
mult_before = sum(1 for v in gold.values() if len(v["acts"]) > 1)
print(f"현 다중-act {mult_before}/48 = {100 * mult_before / 48:.0f}% "
      f"(ESCALATE 분리는 이 비율을 낮추지 않는다 — 라벨을 쪼개는 것이지 합치는 게 아님)")
