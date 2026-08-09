# -*- coding: utf-8 -*-
r"""x180 — **이름을 무의미 라벨로 바꿔도 방향 효과가 남는가** (유료 0·일반화 검정).

## 왜

C357: 오름차순은 두 모델 다 오답 · 내림차순은 두 모델 다 정답(6/6). **정답 자리는 원인이
아니다**(회전 k=24 는 정답을 1번째에 놓고도 실패하는데 `name_desc` 는 같은 자리에서 성공).
남는 변수는 **순서의 방향/관계 구조**다.

그런데 지금까지 전부 **실제 상품명**(Green/Blue/Card…)이 붙은 표였다. 방향 효과가

  · **순수 형식**(무엇이든 정렬된 목록의 방향)이면 → 라벨을 무의미하게 바꿔도 **남는다**
  · **도메인 어휘**에 걸린 것이면 → 라벨을 바꾸면 **사라진다**

전자면 이 결과는 도구 사용 일반으로 확장되고, 후자면 banking 표에 특유하다. 논문 일반화가
여기 걸려 있다.

## 설계

이름 25개를 `A01 B02 … Y25` 로 치환한다 — **첫 글자가 전부 다르고**(접두 공유 0),
알파벳 순서가 **원래 이름순과 일치**하도록 매핑한다 ⇒ `neutral_asc` 는 `name_asc` 와
**같은 행 배열**이고 이름만 무의미하다.

치환은 **일관되게** 적용한다: 표 · 궤적 메시지(정박 문장 포함) · FACTS · 후보 목록.
긴 이름부터 치환해 부분 겹침(`Green` ⊂ `Hunter Green`)을 막는다.

## arm

  domain_asc  / domain_desc    실제 이름 (같은 런 안의 통제)
  neutral_asc / neutral_desc   무의미 라벨

**예측**: 순수 형식이면 neutral_asc 오답 · neutral_desc 정답(도메인판과 같은 패턴).

실행: py -3 x180_neutral_names.py [N] [URL] [MODEL]
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

TASK = "task_099"
GOLD = "World Blue"


def guided_full(url, model, prompt, choices, temp):
    body = json.dumps({"model": model, "temperature": temp, "max_tokens": 12,
                       "guided_choice": list(choices),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8140/v1/chat/completions"
    model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"

    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    lines = LG.eligible_text(730, {}, maps, spec,
                             {"qualifying_deposit_usd": 30000}).strip().splitlines()
    head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
    body = [l for l in lines if l.startswith("  ") and ":" in l]
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731
    names_sorted = sorted(name(l) for l in body)
    # A01 B02 … Y25 — 첫 글자 전부 다름 · 알파벳 순서가 원래 이름순과 일치
    lab = {nm: "%s%02d" % (chr(65 + i), i + 1) for i, nm in enumerate(names_sorted)}
    print("model=%s · 라벨 매핑 예: %s" % (model, list(lab.items())[:3]))
    print("  gold %r → %r" % (GOLD, lab[GOLD]))

    def sub(text):
        """긴 이름부터 치환 — `Green` ⊂ `Hunter Green` 부분 겹침 방지."""
        for nm in sorted(lab, key=len, reverse=True):
            text = text.replace(nm, lab[nm])
        return text

    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    traj = Y.render(MS)
    facts, question = X.FACTS[TASK], X.QUESTION

    def build(neutral, desc):
        b = [sub(l) for l in body] if neutral else list(body)
        nmf = (lambda l: l.strip().split(":")[0].strip())
        order = sorted(b, key=nmf, reverse=desc)
        tbl = "\n".join(([sub(h) for h in head[:1]] if neutral else head[:1])
                        + order + ([sub(h) for h in head[1:]] if neutral else head[1:])).strip()
        t = sub(traj) if neutral else traj
        f = sub(facts) if neutral else facts
        q = sub(question) if neutral else question
        ch = [nmf(l) for l in order]
        gold = lab[GOLD] if neutral else GOLD
        prompt = ("Here is a customer-service conversation so far.\n\n" + t + "\n\n"
                  + tbl + "\n\n" + f + "\n\n" + q)
        return prompt, ch, gold, [nmf(l) for l in order]

    print("\n%-22s %-10s %-34s %s" % ("arm", "정답자리", "분포", "gold"))
    for neutral in (False, True):
        for desc in (False, True):
            prompt, ch, gold, nm = build(neutral, desc)
            c = collections.Counter(guided_full(url, model, prompt, ch, 0.0 if i == 0 else 0.7)
                                    for i in range(n))
            g = c.get(gold, 0)
            print("%-22s %-10s %-34s %d/%d %s"
                  % (("neutral" if neutral else "domain") + ("_desc" if desc else "_asc"),
                     "%d/%d" % (nm.index(gold) + 1, len(nm)), c.most_common(2), g, n,
                     "★정답" if g > n // 2 else ""))
    print("\n  neutral 이 domain 과 같은 패턴이면 → **순수 형식**(도구 사용 일반으로 확장).")
    print("  neutral 에서 사라지면 → 도메인 어휘에 걸린 것이고 이 표에 특유하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
