#!/usr/bin/env python
"""S0 스모크 SFT 빌드: a2_s0_dataset_v3.jsonl(역렌더 쌍) -> chat SFT JSONL
(HANDOFF day4 §0.2-② — prompt=정책NL+카탈로그, target=spec JSON).

lora_train_chat_toolcall.py 입력 포맷({"messages":[...]})으로 변환.
시스템 프롬프트 = t2_a2_size_census.PROMPT_SYS 재사용 (학습/census 프롬프트 일치 —
재census가 곧 held-in 포맷 평가가 되도록). user 메시지도 census usr 구조를 미러
(단 retail 1-shot 예시는 제외 — zero-shot 컴파일을 학습 목표로).

Usage: t2_a2_s0_build_sft.py --dataset specs/a2_s0_dataset_v3.jsonl \
  --out /home/woori/scratch/a2_s0/sft_a2_s0.jsonl [--holdout 15]
holdout은 id 등간격 추출 — 스타일 로테이션 격자라 스타일 균형 유지.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_a2_size_census import PROMPT_SYS  # noqa: E402 — 학습/census 프롬프트 일치


def to_record(r):
    usr = (f"# Compile THIS policy ({r.get('domain_hint', 'domain')}):\n"
           f"{r['policy_nl']}\n\n"
           f"# tool catalog (name: type, required args):\n"
           + json.dumps(r["catalog"], indent=1) + "\n\nOutput the gate spec JSON:")
    return {
        "messages": [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user", "content": usr},
            {"role": "assistant", "content": json.dumps(r["spec"], indent=1)},
        ],
        "meta": {"id": r.get("id"), "style": r.get("style"),
                 "domain_hint": r.get("domain_hint"), "qc": r.get("qc")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=int, default=0,
                    help="N개를 held-out으로 분리 (등간격 — 스타일 균형)")
    ap.add_argument("--holdout_out", default=None)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.dataset, encoding="utf-8") if l.strip()]
    print(f"[build] {len(recs)} pairs from {a.dataset}")
    hold_idx = set()
    if a.holdout:
        step = max(1, len(recs) // a.holdout)
        hold_idx = set(list(range(0, len(recs), step))[:a.holdout])
    train, hold = [], []
    for i, r in enumerate(recs):
        (hold if i in hold_idx else train).append(to_record(r))
    with open(a.out, "w", encoding="utf-8") as f:
        for t in train:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[build] train={len(train)} -> {a.out}")
    if hold:
        out2 = a.holdout_out or a.out.replace(".jsonl", "_holdout.jsonl")
        with open(out2, "w", encoding="utf-8") as f:
            for t in hold:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        styles = {}
        for t in hold:
            styles[t["meta"]["style"]] = styles.get(t["meta"]["style"], 0) + 1
        print(f"[build] holdout={len(hold)} styles={styles} -> {out2}")


if __name__ == "__main__":
    main()
