# -*- coding: utf-8 -*-
"""x135 — 발췌 arm 실험 (§7g). **종수가 아니라 정답 값 보존율로 판정한다.** 유료 0(로컬 vllm).

정본 = `FACT_DAG_DESIGN_2026_08_08.md` §7g · 근거 = 원장 C312.
C312가 남긴 것은 처방이 아니라 판정 불능이었다: `fill`이 종수를 늘리면서 **옳은 값을 하나 잃었다**
(`World Blue=90`). 종수로 고르면 `fill` 승리로 오독된다.

**정답을 gold 없이 정의한다**([[23]]):
    오라클 = **항목을 하나씩, 절단 없이** 물어 얻은 (키, 수, 인용) 합집합 · 인용 실재 검증 통과분만.
⚠오라클을 `split` **1회**로 두면 arm C가 자기 자신과 비교돼 재현율이 **1.0으로 고정**된다(설계
초안의 구멍). 그래서 오라클은 **split을 `--oracle-passes`회 돌린 합집합**이고, C는 그 중 1회분이라
모델 변동만큼 놓칠 수 있다 = 채점 가능해진다.

arm  A `per3000`  현행(항목당 3,000·예산 90,000)          호출 1
     B `fill`     예산 충전(항목당 몫=남은예산/남은항목)   호출 1
     C `split`    항목별 분할 질의                          호출 N

지표(전부 오라클 대비·per-case 목록 동반):
  1 재현율   |arm ∩ 오라클| / |오라클|          (키+값이 모두 같아야 일치)
  2 정밀도   |arm ∩ 오라클| / |arm|             (오라클 밖 = 날조 후보)
  3 ★보존율  다른 arm이 맞힌 (키,값)을 잃지 않았는가  ← C312의 소실이 여기서만 보인다
  4 비용     입력 문자 수 · 호출 수
  + 부정 통제([[57]]): 같은 arm을 같은 입력으로 `--repeats`회 → **모델 변동 폭**. 그 폭보다 작은
    arm 차이는 발췌 효과로 읽지 않는다.

usage: x135_excerpt_arms.py --specs dp:task_100:1,lim:task_100:0 --prompt threshold_prompt \
         --base http://localhost:8140/v1 --out x135_result.json
"""

import argparse
import collections
import gzip
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import t2_ledger as LG                        # noqa: E402
from gate_interpreter import load_domain_a2    # noqa: E402

FIELD = {"threshold_prompt": "min_days", "limit_prompt": "limit"}
DIRS = {"dp": "bank_stack_dp_20260808p", "lim": "bank_stack_lim_20260808n",
        "def": "bank_stack_def_20260808k", "name": "bank_stack_name_20260808m",
        "led": "bank_stack_led_20260807j", "win": "bank_stack_win_20260807i"}


# ── 발췌 규칙 3종 — 위치로만 고르고 내용은 보지 않는다([[59]]) ──────────────────
def a_per3000(items, per=3000, budget=90000):
    sel, used = [], 0
    for t in reversed(list(items)):
        s = str(t)[:per]
        if used + len(s) > budget:
            continue
        sel.append(s)
        used += len(s)
    sel.reverse()
    return [sel]                       # 호출 1회분


def b_fill(items, budget=90000, floor=3000):
    items = list(items)
    rest = list(reversed(items))
    sel, used = [], 0
    for i, t in enumerate(rest):
        cap = max(floor, (budget - used) // max(1, len(rest) - i))
        s = str(t)[:cap]
        if used + len(s) > budget:
            s = s[:max(0, budget - used)]
        if not s:
            if used >= budget:
                break
            continue                   # 빈 항목에서 끊지 않는다(x133 1차 실행의 버그)
        sel.append(s)
        used += len(s)
    sel.reverse()
    return [sel]


def c_split(items, budget=90000):
    return [[str(t)[:budget]] for t in items if str(t).strip()]   # 호출 N회분


ARMS = {"per3000": a_per3000, "fill": b_fill, "split": c_split}


class Agent(object):
    def __init__(self, model, base):
        self.llm = model if model.startswith("openai/") else "openai/" + model
        self.llm_args = {"temperature": 0.0, "api_base": base, "api_key": "dummy"}


def load_sim(dirname, task, idx):
    p = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", dirname + ".json.gz")
    data = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
    sims = [s for s in (data.get("simulations") or []) if s.get("task_id") == task]
    sim = sims[idx]
    texts = []
    for m in sim.get("messages") or []:
        if m.get("role") not in ("tool", "user"):
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(str(x) for x in c)
        texts.append(str(c or ""))
    return texts


def ask(agent, la, UM, template, chunk):
    try:
        prompt = template.format(text="\n---\n".join(chunk))
        try:
            um = UM(role="user", content=prompt)
        except TypeError:
            um = UM(content=prompt)
        kw = {k: v for k, v in dict(agent.llm_args).items() if "tool" not in k}
        sub = la.generate(model=agent.llm, tools=None, messages=[um], call_name="x135", **kw)
        return getattr(sub, "content", None) or "", len(prompt)
    except Exception as e:
        print("    ⚠호출 실패 %r" % (e,), file=sys.stderr)
        return "", 0


def run_arm(agent, la, UM, template, field, hay, chunks):
    """반환 `{(키, 값): 인용}` · 입력 문자 수 · 호출 수. **모으기만 하고 판정하지 않는다.**"""
    got, chars, calls = {}, 0, 0
    for ch in chunks:
        raw, n = ask(agent, la, UM, template, ch)
        chars += n
        calls += 1
        pairs, _rej, _given = LG.parse_pairs(raw, field, hay)
        for k, (num, quote) in pairs.items():
            got.setdefault((k, num), quote)
    return got, chars, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="dp:task_100:1,lim:task_100:0 …")
    ap.add_argument("--prompt", default="threshold_prompt")
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--repeats", type=int, default=2, help="부정 통제: 같은 arm 반복")
    ap.add_argument("--oracle-passes", type=int, default=2)
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--out", default=os.path.join(HERE, "x135_result.json"))
    a = ap.parse_args()

    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage as UM

    a2 = load_domain_a2(a.domain)
    spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get(a.prompt)), None)
    if spec is None:
        raise SystemExit("프롬프트 %r 를 가진 선언이 없다" % a.prompt)
    template, field = spec[a.prompt], FIELD[a.prompt]
    agent = Agent(a.model, a.base)
    out = {"prompt": a.prompt, "model": a.model, "sims": []}

    for token in [t.strip() for t in a.specs.split(",") if t.strip()]:
        short, task, idx = token.split(":")
        texts = load_sim(DIRS[short], task, int(idx))
        hay = " ".join("\n".join(texts).split())
        print("\n=== %s %s sim%s · 항목 %d · %d chars"
              % (short, task, idx, len(texts), sum(len(t) for t in texts)), flush=True)

        # ── 오라클 = split × N패스 합집합 (arm C가 자기 채점을 못 하게) ──────────
        oracle, ochars, ocalls = {}, 0, 0
        for p in range(a.oracle_passes):
            g, ch, cl = run_arm(agent, la, UM, template, field, hay, c_split(texts))
            oracle.update(g)
            ochars += ch
            ocalls += cl
            print("  오라클 패스 %d/%d → 누적 %d쌍" % (p + 1, a.oracle_passes, len(oracle)),
                  flush=True)

        rec = {"spec": token, "items": len(texts),
               "oracle": {"pairs": sorted("%s=%s" % k for k in oracle),
                          "chars": ochars, "calls": ocalls},
               "arms": {}}
        for name, fn in ARMS.items():
            reps = []
            for r in range(a.repeats):
                g, ch, cl = run_arm(agent, la, UM, template, field, hay, fn(texts))
                reps.append({"pairs": sorted("%s=%s" % k for k in g),
                             "chars": ch, "calls": cl,
                             "hit": sorted("%s=%s" % k for k in g if k in oracle),
                             "miss_oracle": sorted("%s=%s" % k for k in oracle if k not in g),
                             "outside": sorted("%s=%s" % k for k in g if k not in oracle)})
                print("  %-9s rep%d → %d쌍 (오라클 적중 %d/%d · 밖 %d)"
                      % (name, r + 1, len(g), len(reps[-1]["hit"]), len(oracle),
                         len(reps[-1]["outside"])), flush=True)
            rec["arms"][name] = reps
        out["sims"].append(rec)

    io.open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n저장: %s" % a.out)
    report(out)
    return 0


def report(out):
    print("\n" + "=" * 96)
    print("지표 — 재현율/정밀도는 (키,값) 일치 기준 · 보존율은 **다른 arm이 맞힌 것을 잃었는가**")
    tot = collections.defaultdict(lambda: [0, 0, 0, 0, 0])     # hit, oracle, arm, chars, calls
    lost_rows = []
    for s in out["sims"]:
        oracle = set(s["oracle"]["pairs"])
        best = {}
        for name, reps in s["arms"].items():
            for r in reps:
                best.setdefault(name, set()).update(r["hit"])
        union_hit = set().union(*best.values()) if best else set()
        for name, reps in s["arms"].items():
            for r in reps:
                t = tot[name]
                t[0] += len(r["hit"])
                t[1] += len(oracle)
                t[2] += len(r["pairs"])
                t[3] += r["chars"]
                t[4] += r["calls"]
            lost = sorted(union_hit - best[name])
            if lost:
                lost_rows.append((s["spec"], name, lost))
    print("\n| arm | 재현율 | 정밀도 | 입력 chars | 호출 |")
    print("|---|---|---|---|---|")
    for name in ("per3000", "fill", "split"):
        h, o, ar, ch, cl = tot[name]
        print("| %-8s | %5.1f%% | %5.1f%% | %9d | %4d |"
              % (name, 100.0 * h / o if o else 0, 100.0 * h / ar if ar else 0, ch, cl))
    print("\n★보존 실패 — **다른 arm은 맞혔는데 이 arm이 놓친 것**:")
    if not lost_rows:
        print("  (없음 — 세 arm이 같은 정답 집합을 잡았다)")
    for spec, name, lost in lost_rows:
        print("  · %-18s %-9s 잃음: %s" % (spec, name, ", ".join(lost)))
    print("\n⚠부정 통제: 같은 arm의 rep 간 차이가 arm 간 차이보다 크면 **발췌 효과로 읽지 않는다**.")


if __name__ == "__main__":
    sys.exit(main())
