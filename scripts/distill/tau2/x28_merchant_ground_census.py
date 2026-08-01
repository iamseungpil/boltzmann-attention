# -*- coding: utf-8 -*-
"""x28: merchant×KB 접지 전수 계수 (2026-08-01·C275 후속·handoff §6b-7 선행 무료 작업).

목적: quote-ground 술어 후보 {현행(전체 merchant 축자 ∈ quote) / 양측-핀+등가(부분문자열 선택 등가)}의
구조적 false-abstain / false-apply 표면과, 비-부분문자열 별칭(열린 채널 필요) 후보를 gold-free로 계수.

전부 결정론(LLM 0·gold 0). 입력 = 도메인 db.json(거래 레코드 merchant_name 전집) + documents/*.json(KB).
출력 = stdout 요약 + JSON(지정 시).

판정 항목:
  A. 현행 규칙 구조적-abstain 후보: head가 제외-문맥 라인에 등장하는데 전체 이름은 코퍼스 어디에도 없음
     → 정책이 그 브랜드를 제외해도 어떤 quote도 술어를 충족 못 함 (Target·Microsoft 365형).
  B. 핀+등가 feasible: merchant의 선행 토큰 n-gram이 코퍼스에 등장 → 부분문자열 다리 존재.
  C. 충돌쌍(false-apply 위험): head 토큰을 공유하는 서로 다른 merchant 그룹 (Delta형) — 정책이
     bare head로만 부를 때 등가가 복수 행에 성립. 그룹 나열=검토 대상.
  D. 별칭 후보(비-부분문자열): 제외-문맥 라인의 대문자 개체 중 어느 merchant와도 전체/부분 매칭이
     안 되는 것 — 의미 확인은 열린 판단이므로 후보 나열까지만.
"""
import json, glob, os, re, sys, io
from collections import defaultdict

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DOM = sys.argv[1] if len(sys.argv) > 1 else "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
OUT = sys.argv[2] if len(sys.argv) > 2 else None

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", s.lower())).strip()

# ---------- 1. merchant 전집 (거래 레코드) ----------
merchants = set()
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "merchant_name" and isinstance(v, str):
                merchants.add(v.strip())
            else:
                walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(json.load(open(os.path.join(DOM, "db.json"), encoding="utf-8")))
merchants = sorted(merchants)

# ---------- 2. KB 코퍼스 (라인 단위 원문 보존) ----------
lines_raw = []           # (doc_id, line)
for f in sorted(glob.glob(os.path.join(DOM, "documents", "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    body = d.get("content") or d.get("text") or json.dumps(d, ensure_ascii=False)
    did = os.path.basename(f)
    for ln in str(body).split("\n"):
        ln = ln.strip()
        if ln:
            lines_raw.append((did, ln))
corpus_n = " ".join(norm(ln) for _, ln in lines_raw)

EXCL_RE = re.compile(r"exclu|not eligible|does not (earn|qualify)|excluded|standard rate|"
                     r"do(es)? not count|no (points|rewards)|only qualifies|not qualify", re.I)
excl_lines = [(d, ln) for d, ln in lines_raw if EXCL_RE.search(ln)]
excl_n = " ".join(norm(ln) for _, ln in excl_lines)

# ---------- 3. merchant별 접지 분류 ----------
def leading_ngrams(mn):
    toks = norm(mn).split()
    return [" ".join(toks[:k]) for k in range(len(toks), 0, -1)]  # 긴 것부터

head_of = {}
rows = []
for m in merchants:
    toks = norm(m).split()
    head = toks[0] if toks else ""
    head_of[m] = head
    full_in = norm(m) in corpus_n
    # 부분문자열 다리: 가장 긴 선행 n-gram이 코퍼스에 등장 (head 자체는 길이 4+만 인정)
    bridge = None
    for g in leading_ngrams(m):
        if g == norm(m):
            if full_in:
                bridge = g
                break
            continue
        if (len(g.replace(" ", "")) >= 4) and (g in corpus_n):
            bridge = g
            break
    head_excl = bool(head) and len(head) >= 4 and head in excl_n
    full_excl = norm(m) in excl_n
    rows.append({"merchant": m, "head": head, "full_in_corpus": full_in,
                 "bridge_ngram": bridge, "head_in_exclusion": head_excl,
                 "full_in_exclusion": full_excl})

# ---------- 4. 집계 ----------
A = [r for r in rows if r["head_in_exclusion"] and not r["full_in_corpus"]]      # 현행 구조적-abstain 후보
B_feasible = [r for r in rows if r["bridge_ngram"]]
B_absent = [r for r in rows if not r["bridge_ngram"]]                            # KB가 아예 안 부름(매핑 불요)

groups = defaultdict(list)
for m in merchants:
    h = head_of[m]
    if h and len(h) >= 4:
        groups[h].append(m)
C = {h: ms for h, ms in groups.items() if len(ms) >= 2 and h in corpus_n}
C_excl = {h: ms for h, ms in C.items() if h in excl_n}

# D. 제외-문맥 라인의 개체 후보(대문자 시퀀스) 중 무매칭
ENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9&'\.]+(?:\s+[A-Z][A-Za-z0-9&'\.]+){0,3})\b")
cand = defaultdict(set)
for did, ln in excl_lines:
    for mm in ENT_RE.finditer(ln):
        ent = mm.group(1).strip()
        if len(norm(ent).replace(" ", "")) >= 4:
            cand[ent].add(did)
def matches_some_merchant(ent):
    en = norm(ent)
    for m in merchants:
        mn = norm(m)
        if en == mn or en in mn or mn in en:
            return True
    return False
D = sorted((e, sorted(ds)) for e, ds in cand.items() if not matches_some_merchant(e))

# ---------- 5. 출력 ----------
print(f"merchant 전집: {len(merchants)}종 · KB 라인 {len(lines_raw)} · 제외-문맥 라인 {len(excl_lines)}")
print(f"\n[A] 현행 규칙 구조적-abstain 후보 (head∈제외문맥 ∧ 전체이름∉코퍼스): {len(A)}종")
for r in A:
    print(f"    {r['merchant']}  (head='{r['head']}' 다리={r['bridge_ngram']})")
print(f"\n[B] 핀+등가 다리 존재(선행 n-gram∈코퍼스): {len(B_feasible)}종 / 다리 없음(KB 무언급→매핑 불요): {len(B_absent)}종")
bridged_not_full = [r for r in B_feasible if not r["full_in_corpus"]]
print(f"    그중 전체이름은 없고 다리만 있음(=핀+등가가 현행 대비 순회수하는 표면): {len(bridged_not_full)}종")
for r in bridged_not_full:
    print(f"    {r['merchant']}  ← 다리 '{r['bridge_ngram']}'" + ("  [제외문맥]" if r["head_in_exclusion"] else ""))
print(f"\n[C] head 공유 충돌 그룹(코퍼스 등장 head): {len(C)}그룹 / 그중 제외-문맥 등장: {len(C_excl)}그룹")
for h, ms in sorted(C.items()):
    tag = "  ★제외문맥" if h in C_excl else ""
    print(f"    '{h}': {ms}{tag}")
print(f"\n[D] 별칭 후보(제외-문맥 개체 중 어떤 merchant와도 전체/부분 무매칭): {len(D)}종 — 의미 확인은 열린 판단(나열까지만)")
for e, ds in D:
    print(f"    '{e}'  ({', '.join(d[:40] for d in ds[:2])})")

# 검증: 실측 3사례
print("\n[검증] 실측 사례 분류:")
for probe in ("Target - Eco Collection", "Microsoft 365", "Thrive Market", "ThredUp"):
    r = next((x for x in rows if x["merchant"] == probe), None)
    print(f"    {probe}: {json.dumps(r, ensure_ascii=False) if r else '(레코드에 없음)'}")

if OUT:
    json.dump({"rows": rows, "A": [r["merchant"] for r in A],
               "bridged_not_full": [r["merchant"] for r in bridged_not_full],
               "collision_groups": C, "collision_excl": C_excl,
               "alias_candidates": [e for e, _ in D]},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nJSON → {OUT}")
