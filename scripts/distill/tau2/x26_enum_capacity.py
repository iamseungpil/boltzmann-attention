# -*- coding: utf-8 -*-
"""X26 — 열거 결손이 **개수 상한**인가 **위치 효과**인가 **목록 길이 부하**인가 (GPU만·gold 미사용 프롬프트).

**왜**: 022는 두 런에서 **첫 열거 메시지가 9건을 확정하고 그 뒤 신규 0**이다(C270 후속 관측).
정답은 10건. 그래서 세 가설이 갈린다 —
  H-CAP  개수 상한: 몇 개를 요구하든 ~9에서 끊긴다(순서·길이 무관)
  H-POS  위치 효과: 목록 뒤쪽이 떨어진다(순서를 뒤집으면 **다른 것**이 떨어진다)
  H-LEN  길이 부하: 목록을 줄이면 전부 찾는다(같은 10건인데 방해 레코드만 줄여도 회복)
세 가설은 **한 프롬프트 안에서 분리 가능**하다. 실제 궤적의 77-레코드 출력을 그대로 쓴다.

**설계 규율(앞선 프로브들의 교훈 반영)**:
  · 프롬프트에 gold 없음 — 그림자 case 동일성으로 확인([[03b]])
  · 순서 조작은 **정보 동일·표시 순서만**(C265 A_reverse 방식)
  · 길이 축소는 **gold 10건 전량 보존** + 방해 레코드만 제거(정보-맞춤 유지 범위서 최대한)
  · 결정의 목적을 프롬프트에 명시(C265: 목적 없으면 미결정 질문이 된다)

**arm**(전부 같은 요청 문구·같은 정보량 원칙):
  full_orig  77건 원순서        | full_rev  77건 역순     | full_rot  77건 회전(gold 위치만 이동)
  trim20     gold 10 + 방해 10  | count_first  "먼저 몇 건인지 말하고 그 다음 나열"

**채점**(gold는 여기서만): 회수한 gold 수 · 놓친 gold의 **원본 위치** · 모델이 낸 총 개수.
  H-CAP → arm 무관하게 회수 ≈9 · H-POS → rev/rot에서 **놓치는 것이 바뀐다** · H-LEN → trim20서 10/10.

용법: py -3 x26_enum_capacity.py --sim <results.json> --task task_022 --base http://…/v1 --out r.jsonl
"""
import argparse
import json
import os
import re
import sys

TXN = re.compile(r"txn_[0-9a-f]+")
REC = re.compile(r"(\d+)\. Record ID: (txn_[0-9a-f]+)")


def load_case(path, task):
    """실제 궤적에서 ①레코드 목록 원문 ②사용자 발화 ③gold 집합을 뽑는다."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sim = next(s for s in d["simulations"] if str(s["task_id"]) == task)
    listing, blocks = None, []
    for m in sim["messages"]:
        if m.get("role") != "tool":
            continue
        c = str(m.get("content") or "")
        if len(REC.findall(c)) >= 10:
            listing = c
            break
    if listing is None:
        raise SystemExit("레코드 목록을 못 찾았다")
    # 레코드 블록 분해(번호 보존)
    parts = re.split(r"(?=\d+\. Record ID: txn_)", listing)
    for p in parts:
        m = REC.search(p)
        if m:
            blocks.append((int(m.group(1)), m.group(2), p.strip()))
    user = "\n".join(str(m.get("content") or "") for m in sim["messages"]
                     if m.get("role") == "user" and (m.get("content") or "").strip())[:1500]
    gold = set()
    for chk in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = chk.get("action") or {}
        args = a.get("arguments") or {}
        inner = args.get("arguments")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        if isinstance(inner, dict) and inner.get("transaction_id"):
            gold.add(inner["transaction_id"])
    return blocks, user, gold


def build(blocks, gold, arm):
    """arm별 레코드 순서/부분집합. **gold를 보고 고르는 것은 trim20의 보존에만** 쓰고
    순서 arm에는 일절 쓰지 않는다(선택 난이도를 바꾸지 않기 위해)."""
    if arm == "full_orig":
        seq = list(blocks)
    elif arm == "full_rev":
        seq = list(reversed(blocks))
    elif arm == "full_rot":
        k = len(blocks) // 2                       # 회전 = 순서 보존·위치만 이동
        seq = blocks[k:] + blocks[:k]
    elif arm == "trim20":
        g = [b for b in blocks if b[1] in gold]
        d = [b for b in blocks if b[1] not in gold][:10]
        seq = sorted(g + d, key=lambda x: x[0])    # 원 번호 순서 유지
    else:                                          # count_first = full_orig 와 같은 목록
        seq = list(blocks)
    return seq


def prompt_for(seq, user, arm, purpose):
    body = "\n\n".join(b[2] for b in seq)
    ask = ("First state how many transactions have a rewards discrepancy as COUNT=<n>, "
           "then list their transaction_ids, one per line."
           if arm == "count_first" else
           "List the transaction_id of EVERY transaction that has a rewards discrepancy, "
           "one per line. Do not explain.")
    return ("Customer request:\n%s\n\nTransaction records:\n%s\n\n%s%s"
            % (user, body, purpose, ask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--task", default="task_022")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--arms", default="full_orig,full_rev,full_rot,trim20,count_first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    blocks, user, gold = load_case(a.sim, a.task)
    pos = {t: n for n, t, _ in blocks}
    purpose = "You are deciding which transactions to file a cash back dispute for. "
    print("레코드 %d개 · gold %d개 · gold 위치 %s"
          % (len(blocks), len(gold), sorted(pos[t] for t in gold if t in pos)))

    import litellm
    rows = []
    for arm in [x for x in a.arms.split(",") if x]:
        seq = build(blocks, gold, arm)
        p = prompt_for(seq, user, arm, purpose)
        for g in gold:                      # 오염 가드: gold를 *지목*하는 문구가 없어야 한다
            assert ("gold" not in p.lower()) and ("correct answer" not in p.lower())
        try:
            r = litellm.completion(model="openai/" + a.model, api_base=a.base, api_key="x",
                                   temperature=0.0, messages=[{"role": "user", "content": p}])
            out = (r.choices[0].message.content or "").strip()
        except Exception as e:
            out = "ERROR: %r" % (e,)
        got = list(dict.fromkeys(TXN.findall(out)))
        listed_ids = {b[1] for b in seq}
        got = [t for t in got if t in listed_ids]
        hit = sorted(pos[t] for t in got if t in gold)
        miss = sorted(pos[t] for t in gold if t in listed_ids and t not in got)
        cnt = re.search(r"COUNT\s*=\s*(\d+)", out)
        rows.append({"arm": arm, "n_records": len(seq), "n_out": len(got),
                     "recall": len([t for t in got if t in gold]), "n_gold": len(gold & listed_ids),
                     "hit_pos": hit, "miss_pos": miss,
                     "claimed_count": int(cnt.group(1)) if cnt else None})
        print("  %-11s 레코드 %2d · 낸 개수 %2d · gold 회수 %d/%d · 놓친 위치 %s%s"
              % (arm, len(seq), len(got), rows[-1]["recall"], rows[-1]["n_gold"], miss,
                 (" · 자기주장 COUNT=%s" % rows[-1]["claimed_count"]) if rows[-1]["claimed_count"] else ""))

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n=== 판정 지침")
    print(" · 전 arm 회수 ≈9로 고정        → H-CAP(개수 상한)")
    print(" · rev/rot에서 **놓친 것이 바뀜** → H-POS(위치 효과)")
    print(" · trim20서 10/10               → H-LEN(길이 부하)")
    print(" · count_first가 회수를 올리면   → 자기-집계 요구가 완결을 산다(값싼 레버 후보)")


if __name__ == "__main__":
    main()
