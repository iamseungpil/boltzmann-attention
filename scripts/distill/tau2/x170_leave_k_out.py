# -*- coding: utf-8 -*-
r"""x170 — 카드 효과는 **누적인가 문턱인가 부재인가** (유료 0·제거만·분포 기록).

## 왜 (세션 간 충돌·[[56]])

  · 다른 세션: 카드 **9행 일괄** 제거 → 정답 복귀
  · x169(C352): 카드 **1행씩** 제거 → **9행 전부 무효**

둘 다 참이려면 카드 효과는 **집합적**이어야 한다. 그리고 x169 에는 실제 계기 결함이 있다 —
**n=5·argmax 만** 기록해 5/5→3/5 같은 부분 이동을 못 봤다. 그게 누적의 흔적일 수 있다.

## 설계

k = 0·2·4·6·8·9 를 **누적 제거**하고 **분포 전체**(n=10)를 기록한다.
  · `cards_asc` / `cards_desc` : 카드 9행을 보너스 오름/내림 순으로 — 개수인가 특정 카드인가
  · `ctrl_asc`               : **비-카드** 행을 같은 개수만큼 — 행 수 효과 분리
⚠통제 집합에서 `Hunter Green`·`Lime Green`(C352 의 실효 쌍)과 `World Blue`(정답)는 **제외**한다.
  넣으면 통제가 자명하게 뒤집힌다. 설계 선택이므로 명시한다.

## 사전 등록 예측 (돌리기 전에 적는다·[[08]])

  · 다른 세션 읽기가 맞으면 → **카드 곡선만** 어느 k 에서 World Blue 로 넘어가고 통제는 안 넘어감
  · 두 곡선이 같이 넘어가면 → 카드가 아니라 **행 수**
  · asc 와 desc 가 다르면 → 개수가 아니라 **특정 카드**
  · 아무 것도 안 넘어가면 → 다른 세션의 9행 결과는 **두 표의 다른 차이**에서 온 것이다

실행: py -3 x170_leave_k_out.py [N]   (8140 = 32B 필요)
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
TASK = "task_099"
TRIGGER = 26
KEEP = ("Hunter Green", "Lime Green", "World Blue")   # 통제에서 제외(자명한 뒤집힘 방지)
KS = (0, 2, 4, 6, 8, 9)


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    lines = table.splitlines()
    ix = {}
    for i, l in enumerate(lines):
        if l.startswith("  ") and ":" in l:
            ix[l.strip().split(":")[0].strip()] = i
    # 계열·보너스는 A3 에서 (프로브가 분류·상수를 발명하지 않는다)
    cat = {}
    for r in rows:
        s, d = (r or {}).get("subject"), ((r or {}).get("source") or {}).get("doc")
        if s and d and s not in cat:
            cat[s] = "_".join(str(d).split("_")[1:3])
    bax = next((a for a in axes if "bonus" in a.lower()), axes[0])
    bon = {}
    for s, v in (maps.get(bax) or {}).items():
        try:
            bon[s] = float(str(v[0]).replace(",", ""))
        except Exception:
            bon[s] = 0.0

    cards = sorted([nm for nm in ix if "credit" in (cat.get(nm) or "")], key=lambda s: bon.get(s, 0))
    ctrl = sorted([nm for nm in ix if "credit" not in (cat.get(nm) or "") and nm not in KEEP],
                  key=lambda s: bon.get(s, 0))
    print("model=%s · 표 %d행 · 카드 %d · 통제후보 %d(제외 %s)"
          % (MODEL, len(ix), len(cards), len(ctrl), list(KEEP)))
    print("  카드(보너스 오름): %s" % [(c, bon.get(c)) for c in cards])

    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    pre = "Here is a customer-service conversation so far.\n\n" + Y.render(MS) + "\n\n"

    def run(drop):
        keep = [l for j, l in enumerate(lines) if j not in {ix[d] for d in drop}]
        ch = [l.strip().split(":")[0].strip() for l in keep if l.startswith("  ") and ":" in l]
        base = "\n".join(keep).strip() + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        got = [guided_full(pre + base, ch, 0.0 if i == 0 else 0.7) for i in range(n)]
        return collections.Counter(got)

    # ★leave-one-in (2026-08-09·C353⒠): 오름 k=8 은 카드가 `Business Platinum` **하나만 남은**
    #   상태에서 오답 10/10 인데, x169 에서 그 카드 **하나만 제거**하는 것은 무효였다.
    #   ⇒ 단독 **충분성**과 단독 **필요성**이 갈린다. 카드를 정확히 하나만 남기고 재면
    #     *어느 장이 혼자서 효과를 지탱하는가* 가 나온다. leave-one-out 의 정확한 짝이다.
    if os.environ.get("T2_LEAVE_ONE_IN") == "1":
        print("\n%-34s %-8s %s" % ("남긴 카드 1장", "행수", "분포 (n=%d)" % n))
        for keep_card in cards:
            drop = [c for c in cards if c != keep_card]
            c = run(drop)
            gold = c.get("World Blue", 0)
            print("%-34s %-8d %-42s gold=%d/%d %s"
                  % ("%s (%s)" % (keep_card, bon.get(keep_card)), len(ix) - len(drop),
                     c.most_common(3), gold, n, "★정답" if gold > n // 2 else "**오답 유지**"))
        c = run(cards)
        print("%-34s %-8d %-42s gold=%d/%d" % ("(카드 0장·통제)", len(ix) - len(cards),
                                               c.most_common(3), c.get("World Blue", 0), n))
        return 0

    sets = [("cards_asc", cards), ("cards_desc", cards[::-1]), ("ctrl_asc", ctrl)]
    print("\n%-12s %-4s %-8s %s" % ("집합", "k", "행수", "분포 (n=%d)" % n))
    for label, pool in sets:
        for k in KS:
            if k > len(pool):
                continue
            drop = pool[:k]
            c = run(drop)
            gold = c.get("World Blue", 0)
            print("%-12s %-4d %-8d %-42s gold=%d/%d %s"
                  % (label, k, len(ix) - k, c.most_common(3), gold, n,
                     "★정답 우세" if gold > n // 2 else ""))
    print("\n  카드 곡선만 넘어가면 '범위-밖이 조건' 지지 · 둘 다면 행 수 · asc≠desc 면 특정 카드.")
    print("  아무 것도 안 넘어가면 다른 세션의 9행 결과는 **두 표의 다른 차이**에서 온 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
