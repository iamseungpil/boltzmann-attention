# -*- coding: utf-8 -*-
r"""x169 — **실효 경쟁 집합**: 어느 행 하나를 빼야 답이 바뀌는가 (유료 0·제거만).

## 왜 이 이름인가 (이름 짓기 원칙)

"가족"과 "웅덩이"는 **추측을 이름으로 굳혀서** 무너졌다(C351 · 다른 세션 §6.4 정정). 그래서
다음 이름은 **측정 방식을 이름에 넣는다** — 이 파일이 정의하는 것은 딱 하나다:

    실효 경쟁 집합 = { 행 r : 표에서 r **하나만** 빼면 정박 하의 답이 바뀐다 }

의미도, 토큰도, 가족도 가정하지 않는다. **제거 개입으로만** 정의되므로 정책 상수를 지어내지
않는다([[03b]]) — 다른 세션이 세운 *"제거만으로 설계"* 원칙과 같은 자리다.

## 무엇을 대체하는가

다른 세션은 ⒜9행 묶음 제거(카드 대 최저보너스)로 *"범위-밖 항목이 있어야 정박이 문다"* 를,
⒝동점 상대 1행 제거로 *"최상위 동점은 지렛대가 아니다"* 를 얻었다. ⒜ 는 **두 표가 행 수만
같고 남은 내용이 통째로 다르다** — 카드 때문인지 다른 무엇 때문인지 못 가른다. 1행 단위
leave-one-out 은 그 질문을 **행마다** 답하고, ⒝ 를 특수 케이스로 포함한다.

## 사전 등록 예측 (돌리기 전에 적는다·[[08]])

  · 다른 세션의 읽기(범위-밖 항목이 조건)가 맞으면 → **답을 바꾸는 행은 카드 행에 몰린다**
  · 최상위 동점 가설은 이미 기각됐으므로 → `Business Platinum` 단독 제거는 답을 **안 바꾼다**
  · 아무 행도 단독으로 답을 못 바꾸면 → 효과는 **집합적**이고 leave-one-out 으로는 안 잡힌다
    (그 경우 다음은 leave-k-out 이지 새 서술이 아니다)

실행: py -3 x169_effective_set.py [N]   (8140 = 32B 필요)
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


def guided_full(prompt, choices, temp):
    body = json.dumps({"model": MODEL, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    axes = spec["eligible"]["show_axes"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    lines = table.splitlines()
    body_ix = [i for i, l in enumerate(lines) if l.startswith("  ") and ":" in l]
    names = [lines[i].strip().split(":")[0].strip() for i in body_ix]
    # 계열 라벨은 A3 에서 읽는다 (프로브가 분류를 발명하지 않는다)
    cat = {}
    for r in rows:
        s, d = (r or {}).get("subject"), ((r or {}).get("source") or {}).get("doc")
        if s and d and s not in cat:
            cat[s] = "_".join(str(d).split("_")[1:3])
    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)

    def ask(tbl_lines, msgs, k):
        tbl = "\n".join(tbl_lines).strip()
        ch = [l.strip().split(":")[0].strip() for l in tbl_lines if l.startswith("  ") and ":" in l]
        base = tbl + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        pre = ("Here is a customer-service conversation so far.\n\n" + Y.render(msgs) + "\n\n")
        return [guided_full(pre + base, ch, 0.0 if i == 0 else 0.7) for i in range(k)], ch

    print("model=%s · 표 %d행" % (MODEL, len(body_ix)))
    full, _ = ask(lines, MS, n)
    noanchor, _ = ask(lines, MS[:TRIGGER] + MS[TRIGGER + 1:], n)
    base_ans = collections.Counter(full).most_common(1)[0][0]
    clean_ans = collections.Counter(noanchor).most_common(1)[0][0]
    print("  전체표+정박   : %s" % collections.Counter(full).most_common(2))
    print("  전체표+정박없음: %s" % collections.Counter(noanchor).most_common(2))
    print("\n  ⇒ 기준 답(정박 하) = %r · 정박 없을 때 = %r" % (base_ans, clean_ans))
    if base_ans == clean_ans:
        print("  ⚠정박이 이 표에서 답을 안 바꾼다 — leave-one-out 의 전제가 없다. 중단.")
        return 1

    print("\n%-34s %-22s %-26s %s" % ("제거한 행", "계열(A3)", "답", "바뀜?"))
    changed = []
    for i, nm in zip(body_ix, names):
        got, _ = ask([l for j, l in enumerate(lines) if j != i], MS, n)
        top = collections.Counter(got).most_common(1)[0][0]
        flip = (top != base_ans)
        if flip:
            changed.append((nm, top))
        print("%-34s %-22s %-26s %s" % (nm, cat.get(nm, "?"), top, "★바뀜" if flip else ""))

    print("\n=== 실효 경쟁 집합 (이 행 하나를 빼면 답이 바뀐다) ===")
    if not changed:
        print("  **비어 있다** — 어떤 단일 행도 답을 못 바꾼다 ⇒ 효과는 집합적이다.")
        print("  다음은 leave-k-out 이지 새 서술이 아니다.")
    else:
        byc = collections.Counter(cat.get(nm, "?") for nm, _ in changed)
        for nm, top in changed:
            print("  %-32s → %s   (계열 %s)" % (nm, top, cat.get(nm, "?")))
        print("  계열 분포: %s" % dict(byc))
        print("  ⇒ 카드 행에 몰리면 '범위-밖 항목이 조건'이라는 읽기가 행 단위로 지지된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
