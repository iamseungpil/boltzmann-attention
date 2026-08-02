#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x46: ax32 중간-포렌식 증거 영속화 (2026-08-02 · 리뷰 G① — scratchpad-only 인용 금지 프로토콜).

라이브 sim(results.json)을 스테이징해 x38(태스크별 분해)을 돌리고, AX32_MIDRUN_PRESCRIPTIONS_DESIGN이
인용하는 per-step 축자 프로브(010 log_verification 창·012 KB 반환·018 give 창·019 인자/디스크레펀시)를
한 md로 영속한다. dbdiff(010)는 PYTHONPATH=tau2-bench/src 필요·--dbdiff 로 opt-in.

Run(remote):
  seka python x46_ax32_forensic.py \
    --tags bank_ax32p1_gpu0_20260802,bank_ax32p1_gpu1_20260802 \
    --out $REPO/reports/facet_rft_2026/AX32P1_FORENSIC_EVIDENCE_2026_08_02.md [--dbdiff task_010]
"""
import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
STAGE = "/home/woori/scratch/axis32run/x46stage"

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--simroot", default=SIMROOT)
ap.add_argument("--dbdiff", default="")
A = ap.parse_args()
TAGS = [t.strip() for t in A.tags.split(",") if t.strip()]

os.makedirs(STAGE, exist_ok=True)
sims = []
for tag in TAGS:
    src = os.path.join(A.simroot, tag, "results.json")
    if not os.path.exists(src):
        print("skip (no results):", tag); continue
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    with gzip.open(os.path.join(STAGE, tag + ".results.json.gz"), "wt", encoding="utf-8") as g:
        json.dump(d, g, ensure_ascii=False)
    for s in d.get("simulations", []):
        s["_tag"] = tag
        sims.append(s)

# ── x38 태스크별 분해(스테이징 디렉터리로) ──
x38_out = os.path.join(STAGE, "x38_section.md")
subprocess.run([sys.executable, os.path.join(HERE, "x38_pertask_trajectory.py"),
                "--simdir", STAGE, "--tags", ",".join(TAGS), "--out", x38_out], check=True)

def by_task(tid):
    return [s for s in sims if s.get("task_id") == tid]

def calls(s, name=None):
    out = []
    for i, m in enumerate(s.get("messages", [])):
        for tc in (m.get("tool_calls") or []):
            if name is None or tc.get("name") == name:
                out.append((i, m.get("role"), tc))
    return out

def tool_resp(s, idx, span=4):
    msgs = s.get("messages", [])
    for j in range(idx + 1, min(idx + span, len(msgs))):
        if msgs[j].get("role") == "tool":
            return j, str(msgs[j].get("content"))
    return None, ""

L = []
L.append("# ax32p1 중간-포렌식 증거 (x46 영속본 · 2026-08-02)\n")
L.append("> 생성기 = `x46_ax32_forensic.py` · 입력 = %s · **부분 런 스냅샷**(pass1 진행 중 채취)." % ",".join(TAGS))
L.append("> 소비처 = `AX32_MIDRUN_PRESCRIPTIONS_DESIGN_2026_08_02.md` §1. 등대 프로토콜: 인용은 이 영속본으로.\n")

# ── 프로브 1: 010 log_verification 창 ──
for s in by_task("task_010"):
    L.append("## 프로브 010 — log_verification 창 (중복-write 가설 기각 근거)\n")
    lv = calls(s, "log_verification")
    for i, role, tc in lv:
        L.append("- call@%d args: `%s`" % (i, json.dumps(tc.get("arguments"), ensure_ascii=False)))
        j, c = tool_resp(s, i)
        L.append("  - resp@%s: %s" % (j, " ".join(c.split())[:200]))
    L.append("")

# ── 프로브 2: 012 KB 반환(산문-변이 증거) ──
for s in by_task("task_012"):
    L.append("## 프로브 012 — KB_search 반환 전수 (무득점 신호 = LLM 산문 증거)\n")
    for i, role, tc in calls(s, "KB_search"):
        j, c = tool_resp(s, i)
        L.append("- q=`%s` → len=%d · head=%s" % (
            json.dumps(tc.get("arguments"), ensure_ascii=False)[:100], len(c), " ".join(c.split())[:120]))
    L.append("")

# ── 프로브 3: 018 give 창 + 직전 대화 ──
for s in by_task("task_018"):
    L.append("## 프로브 018 — give 배치·직전 대화 창 (채널-오설명→거부 사슬)\n")
    gv = calls(s, "give_discoverable_user_tool")
    idxs = sorted(set(i for i, _, _ in gv))
    L.append("- give 총 %d콜 · 메시지 위치 %s" % (len(gv), idxs))
    if idxs:
        first = idxs[0]
        msgs = s.get("messages", [])
        n_in_msg = len([1 for i, _, _ in gv if i == first])
        L.append("- 첫 give 메시지@%d 내 병렬 give 수 = %d" % (first, n_in_msg))
        for k in range(max(0, first - 5), min(first + 2, len(msgs))):
            m = msgs[k]
            if m.get("role") in ("assistant", "user") and isinstance(m.get("content"), str) and m["content"].strip():
                L.append("  - [%s@%d] %s" % (m["role"], k, " ".join(m["content"].split())[:200]))
    L.append("")

# ── 프로브 4: 019 call 인자 전수 + discrepancies 출력 ──
for s in by_task("task_019"):
    L.append("## 프로브 019 — dispute call 인자 전수 + get_reward_discrepancies 출력 (여분 2건 출처)\n")
    for i, role, tc in calls(s, "call_discoverable_user_tool"):
        L.append("- call@%d [%s] `%s`" % (i, role, json.dumps(tc.get("arguments"), ensure_ascii=False)[:180]))
    for i, role, tc in calls(s, "get_reward_discrepancies"):
        j, c = tool_resp(s, i)
        L.append("- discrepancies 출력(전문 1200자):\n\n```\n%s\n```" % c[:1200])
        ids = sorted(set(re.findall(r"txn_[0-9a-f]{12}", c)))
        L.append("- 반환 txn 집합: %s" % ids)
    L.append("")

# ── 프로브 팩 2 (r4 근거: 028 손-전사 사슬 · 040 give 인자 · 032/033 마커 · 020/027 coverage · 024 fit) ──
def probe_pack2():
    out = []
    # 028: discrepancy 5회 반환 + 의심 id 정확-일치 소스
    for s in by_task("task_028"):
        out.append("## 프로브 028 — get_reward_discrepancies 반환 전수 + 전사-id 출처\n")
        msgs = s.get("messages", [])
        n = 0
        for i, role, tc in calls(s, "get_reward_discrepancies"):
            n += 1
            j, c = tool_resp(s, i)
            out.append("- #%d(call@%d) head: %s" % (n, i, " ".join(c.split())[:180]))
        for sus in ("txn_d3b830f4a2a4", "txn_4c29a0f4a2a4", "txn_7d3b830f4a2a4"):
            src = None
            for i, m in enumerate(msgs):
                if m.get("role") == "tool":
                    for hit in re.finditer(r"txn_[0-9a-f]+", str(m.get("content"))):
                        if hit.group(0) == sus:
                            src = i; break
                if src is not None:
                    break
            out.append("- %s: tool-정확일치 idx=%s" % (sus, src))
        for i, role, tc in calls(s, "give_discoverable_user_tool"):
            out.append("- give@%d args: %s" % (i, json.dumps(tc.get("arguments"), ensure_ascii=False)[:150]))
        out.append("")
    # 040: give 인자(placeholder 실증) + 이관 요구 발화
    for s in by_task("task_040"):
        out.append("## 프로브 040 — give 인자 붕괴(placeholder)\n")
        for i, role, tc in calls(s, "give_discoverable_user_tool"):
            out.append("- give@%d args: %s" % (i, json.dumps(tc.get("arguments"), ensure_ascii=False)[:170]))
        out.append("")
    # 032/033: 마커·KB 무언급·transfer 직전 창
    for tid in ("task_032", "task_033"):
        for s in by_task(tid):
            out.append("## 프로브 %s — 마커·KB·직전 창(미시도 실증)\n" % tid)
            msgs = s.get("messages", [])
            mk = [w for m in msgs if m.get("role") == "tool"
                  for w in re.findall(r"\[T2_[A-Z_]+\]|\[GUIDANCE\]|\[coverage\]", str(m.get("content")))]
            hit = any("initial_transfer" in str(m.get("content")) for m in msgs if m.get("role") == "tool")
            out.append("- 엔진 마커: %s · KB 출력 initial_transfer 언급: %s" % (mk or "없음", hit))
            ti = [i for i, r, tc in calls(s, "transfer_to_human_agents")]
            if ti:
                for k in range(max(0, ti[0] - 3), ti[0] + 1):
                    m = msgs[k]
                    if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip():
                        out.append("- [%s@%d] %s" % (m["role"], k, " ".join(m["content"].split())[:160]))
            out.append("")
    # 020/027: coverage 마커 원문(검증-감사≠제출-완결 실증)
    for tid in ("task_020", "task_027"):
        for s in by_task(tid):
            out.append("## 프로브 %s — [coverage] 마커 원문\n" % tid)
            for m in s.get("messages", []):
                c = str(m.get("content"))
                if m.get("role") == "tool" and "[coverage]" in c:
                    k = c.find("[coverage]")
                    out.append("- %s" % " ".join(c[k:k + 160].split()))
            out.append("")
    # 024: fit 출력 head(FIT facts 표면화 실증)
    for s in by_task("task_024"):
        out.append("## 프로브 024 — fit 출력 head(FIT_DIFF facts 실림)\n")
        for i, role, tc in calls(s, "check_card_application_fit"):
            j, c = tool_resp(s, i)
            out.append("- fit@%d head: %s" % (i, " ".join(c.split())[:260]))
        out.append("")
    return out

L.extend(probe_pack2())

# ── dbdiff (opt-in) ──
if A.dbdiff:
    L.append("## dbdiff — %s (initial_state None-가드 변형)\n" % A.dbdiff)
    try:
        from pathlib import Path
        from tau2.registry import registry
        from tau2.data_model.simulation import Results
        for tid in A.dbdiff.split(","):
            tid = tid.strip()
            tgt = None
            for tag in TAGS:
                p = os.path.join(A.simroot, tag, "results.json")
                if not os.path.exists(p):
                    continue
                res = Results.load(Path(p))
                m = [x for x in res.simulations if x.task_id == tid]
                if m:
                    tgt = (tag, m[0]); break
            if not tgt:
                L.append("- %s: sim 없음" % tid); continue
            tag, sim = tgt
            env_ctor = registry.get_env_constructor("banking_knowledge")
            task = [t for t in registry.get_tasks_loader("banking_knowledge")() if t.id == tid][0]
            ist = task.initial_state
            idata = ist.initialization_data if ist else None
            iacts = ist.initialization_actions if ist else None
            hist = list(ist.message_history or []) if ist else []
            gold = env_ctor(retrieval_variant="no_knowledge"); gold.set_state(idata, iacts, hist)
            for a in (task.evaluation_criteria.actions or []):
                try:
                    gold.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
                except Exception as e:
                    L.append("- gold ERR %s: %r" % (a.name, e))
            pred = env_ctor(retrieval_variant="no_knowledge"); pred.set_state(idata, iacts, list(sim.messages))
            L.append("- %s agent_db_match=%s user_db_match=%s" % (
                tid, gold.tools.get_db_hash() == pred.tools.get_db_hash(),
                (gold.user_tools.get_db_hash() == pred.user_tools.get_db_hash()) if gold.user_tools else "n/a"))
            def diff(g, p, path, out):
                if isinstance(g, dict) and isinstance(p, dict):
                    for k in set(g) | set(p):
                        if k not in g: out.append("  - ONLY-PRED %s.%s = %s" % (path, k, str(p[k])[:140]))
                        elif k not in p: out.append("  - ONLY-GOLD %s.%s = %s" % (path, k, str(g[k])[:140]))
                        else: diff(g[k], p[k], "%s.%s" % (path, k), out)
                elif isinstance(g, list) and isinstance(p, list):
                    if len(g) != len(p): out.append("  - LEN %s: gold=%d pred=%d" % (path, len(g), len(p)))
                    for i in range(min(len(g), len(p))): diff(g[i], p[i], "%s[%d]" % (path, i), out)
                elif g != p:
                    out.append("  - DIFF %s: gold=%s / pred=%s" % (path, str(g)[:100], str(p)[:100]))
            dd = []
            diff(gold.tools.db.model_dump(), pred.tools.db.model_dump(), "", dd)
            L.extend(dd or ["  - (agent DB diff 없음)"])
    except Exception as e:
        L.append("- dbdiff 불가(PYTHONPATH=tau2-bench/src 필요): %r" % e)

# ── x38 본문 병합 ──
L.append("\n---\n\n# 부록: x38 태스크별 분해 (동일 스냅샷)\n")
with open(x38_out, encoding="utf-8") as f:
    L.append(f.read())

with open(A.out, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("WROTE", A.out)
