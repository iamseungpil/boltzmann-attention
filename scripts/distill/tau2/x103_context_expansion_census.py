# -*- coding: utf-8 -*-
"""Is context expansion real, and is our own feedback what expands it?

Every cap in this stack is justified by one sentence — "the gate's own cost is context" —
and that sentence has never been measured. It was the stated reason for the follow-up
budget that, when read, turned out to have silenced the one lever whose predicate held
(028). Before any cap is re-introduced anywhere, the claim has to face the data:

  ① how conversations actually ended        context/step deaths vs the customer stopping
  ② what fills a conversation               tool output, model prose, customer text
  ③ what our layer added on top             the sidecar's bytes against the trajectory's
  ④ whether compaction had to intervene     VIEW_COMPACT firings per run

②/③ are different questions: the trajectory holds what was committed, while our feedback
goes to the generation buffer and is mostly replaced — so a large feedback volume is not by
itself expansion of the committed context. Both are reported.

Free: persisted trajectories, the sidecar, and run logs.

  usage: x103_context_expansion_census.py [arm] [tag]
"""

import collections
import glob
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x50_says_not_does import ARMS, SIM   # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
TAG = sys.argv[2] if len(sys.argv) > 2 else "20260805n"
LOGD = os.environ.get("T2_LOGD", "/home/woori/scratch/logs")
SIMD = os.environ.get("T2_SIMD", "/home/woori/scratch/tau2-bench/data/simulations")


def load_arm():
    out = []
    for p in sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        out.extend(d.get("simulations") if isinstance(d, dict) else d)
    return out


def load_tag():
    out = []
    for p in sorted(glob.glob(os.path.join(SIMD, "bank_smk_gpu*_%s" % TAG, "results.json"))):
        out.extend(json.load(io.open(p, encoding="utf-8")).get("simulations") or [])
    return out


def chars(sims):
    per = collections.Counter()
    sizes = []
    for s in sims:
        tot = collections.Counter()
        for m in s.get("messages") or []:
            n = len(str(m.get("content") or ""))
            n += sum(len(json.dumps(tc.get("arguments"), ensure_ascii=False))
                     for tc in (m.get("tool_calls") or []))
            tot[m.get("role") or "?"] += n
        sizes.append((sum(tot.values()), s.get("task_id"), dict(tot)))
        per.update(tot)
    return per, sizes


def report(name, sims):
    if not sims:
        print("  %s: 데이터 없음" % name)
        return
    term = collections.Counter(s.get("termination_reason") for s in sims)
    per, sizes = chars(sims)
    tot = sum(per.values()) or 1
    sizes.sort()
    med = sizes[len(sizes) // 2]
    print("\n[%s] sim %d" % (name, len(sims)))
    print("  ① 종료 사유: %s" % dict(term))
    print("  ② 문맥 구성(문자 기준): %s"
          % {k: "%.1f%%" % (100.0 * v / tot) for k, v in per.most_common()})
    print("     sim당 총 문자 — 중앙값 %d · 최대 %d(%s)" % (med[0], sizes[-1][0], sizes[-1][1]))


print("=" * 78)
print("맥락 팽창 실재 여부 — 전수 조사")
report("%s (지속 데이터)" % ARM, load_arm())
tag_sims = load_tag()
report("스모크 %s" % TAG, tag_sims)

# ③ 우리 층이 더한 양 — 사이드카가 있는 런에서만
p = os.path.join(LOGD, "fb_%s.jsonl" % TAG)
if os.path.exists(p) and tag_sims:
    import hashlib

    def fp(sim):
        for m in sim.get("messages") or []:
            if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
                return hashlib.sha1(m["content"].strip().encode("utf-8")).hexdigest()[:12]
        return None

    fb = collections.Counter()
    fbn = collections.Counter()
    for line in io.open(p, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        fb[r.get("sim")] += int(r.get("len") or len(str(r.get("text") or "")))
        fbn[r.get("sim")] += 1
    print("\n  ③ 우리 층이 보낸 양(사이드카) vs 궤적")
    print("     %-10s %-8s %-9s %-9s %s" % ("task", "결과", "궤적문자", "지시문자", "지시/궤적"))
    rows = []
    for s in tag_sims:
        k = fp(s)
        traj = sum(len(str(m.get("content") or "")) for m in (s.get("messages") or []))
        rows.append((fb.get(k, 0) / float(traj or 1), s.get("task_id"),
                     (s.get("reward_info") or {}).get("reward"), traj, fb.get(k, 0), fbn.get(k, 0)))
    for r, tid, rew, traj, f, n in sorted(rows, reverse=True):
        print("     %-10s %-8s %-9d %-9d %.0f%% (지시 %d건)"
              % (tid, "PASS" if rew == 1.0 else "fail", traj, f, 100 * r, n))

# ④ 압축 레버가 개입했나
print("\n  ④ 압축·문맥 레버 발화")
for f in sorted(glob.glob(os.path.join(LOGD, "bank_smk_gpu*_%s.log" % TAG))):
    txt = io.open(f, encoding="utf-8", errors="ignore").read()
    marks = {k: txt.count("[%s]" % k) for k in
             ("T2_VIEW_COMPACT", "T2_DYN_MT", "T2_REPEAT_CAP", "T2_REPEAT_GOV")}
    print("     %-40s %s · 'context' 언급 %d · 'max_steps' %d"
          % (os.path.basename(f), marks, txt.lower().count("context length"),
             txt.count("max_steps")))
