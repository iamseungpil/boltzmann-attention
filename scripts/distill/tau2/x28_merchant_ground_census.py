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
HEAD_RE = re.compile(r"^\s*(#{1,6})\s")

# ★섹션-스코프 + 깊이 상속(2026-08-01 자기수정 2회):
#   (1차) 라인 단위로 잡으면 `## What is excluded…` 제목 밑 불릿(`- Target`)이 누락된다.
#   (2차) 섹션을 잡아도 `### General Retailers`처럼 **키워드 없는 하위 제목**이 제외 섹션을 껐다
#         (실측: Target이 [A]서 계속 빠짐 — ecocard_004 구조가 정확히 이 형태).
#   ⇒ 제외-제목의 깊이를 기억하고, 더 깊은 제목은 상속·같거나 얕은 제목에서만 해제.
excl_lines, in_excl, excl_depth, cur_doc = [], False, 0, None
for d, ln in lines_raw:
    if d != cur_doc:
        cur_doc, in_excl, excl_depth = d, False, 0
    hm = HEAD_RE.match(ln)
    if hm:
        lvl = len(hm.group(1))
        if EXCL_RE.search(ln):
            in_excl, excl_depth = True, lvl
        elif in_excl and lvl > excl_depth:
            pass                      # 하위 제목 = 제외 섹션 상속
        else:
            in_excl, excl_depth = False, 0
    if in_excl or EXCL_RE.search(ln):
        excl_lines.append((d, ln))
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
# 문두·일반명사 잡음 제거: 불릿 항목만 + 상품/일반어 스톱리스트(도메인 리터럴 아님·문법 범주).
STOP = set("""account accounts additional additionally after always approved balance bank before bronze
buying cards cash claims combined common confirm coverage credits debit deposit each eligible example
exceptions excluded exclusions exclusive gift gold hardware important invitations items keep marketplaces
merchant merchants mixed note only once other partial payments pending procedure purchases qualify
qualifying reasons referrals retention savings self situations some these this tips unlimited unsupported
utilization what will your silver platinum diamond green ecocard rewards card""".split())
BULLET_RE = re.compile(r"^\s*[-*•]\s+")
cand = defaultdict(set)
for did, ln in excl_lines:
    if not BULLET_RE.match(ln):
        continue                      # 산문 문장은 개체 추출 잡음이 지배 → 불릿 항목만
    body = BULLET_RE.sub("", ln)
    for mm in ENT_RE.finditer(body):
        ent = mm.group(1).strip()
        toks = norm(ent).split()
        if len(norm(ent).replace(" ", "")) >= 4 and not all(t in STOP for t in toks):
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

# ---------- 5b. C278 술어 정적 시뮬 (설계서 §6-2) ----------
# 가정: sub가 정책이 쓰는 이름을 핀한다 = 각 merchant의 "다리" n-gram(코퍼스 등장 최장 선행) 또는
#       비선행 매칭 토큰. 엔진 판정만 결정론 재현(LLM 없음).
print("\n[C278 정적 시뮬] named 경로(선행 앵커) 통과/기각")
try:
    import importlib.util as _ilu
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_scaffold_get.py")
    _sp = _ilu.spec_from_file_location("_sg_for_census", _p)
    _sg = _ilu.module_from_spec(_sp)
    import types as _t, sys as _s
    for _m, _a in (("tau2", {}), ("tau2.data_model", {}), ("tau2.data_model.message", {"UserMessage": object, "ToolMessage": object, "MultiToolMessage": object})):
        if _m not in _s.modules:
            _mod = _t.ModuleType(_m)
            for _k, _v in _a.items():
                setattr(_mod, _k, _v)
            _s.modules[_m] = _mod
    _sp.loader.exec_module(_sg)
    QP = {"policy_field": "p", "kind_field": "k", "row_field": "m", "pin_anchor": "leading"}

    def _verdict(pin, merchant, quote):
        return _sg._quote_pin_check(QP, {"exclusion_quote": quote, "p": pin, "k": "named_merchant"},
                                    {"m": merchant}, "exclusion_quote", 0, norm(quote))[0]
    a_pass = [r["merchant"] for r in rows
              if r["head_in_exclusion"] and not r["full_in_corpus"] and r["bridge_ngram"]
              and _verdict(r["bridge_ngram"], r["merchant"], r["bridge_ngram"]) == "pass"]
    print(f"  현행 구조적-abstain 후보 중 named+앵커로 회수: {len(a_pass)}/{len(A)}  {a_pass}")
    # 비선행 범주어(위 [B] 무-다리 중 임의 토큰 매칭)가 named 경로에서 기각되는지
    gen_rej = gen_pass = []
    gen_rej, gen_pass = [], []
    for m in merchants:
        if any(r["merchant"] == m and r["bridge_ngram"] for r in rows):
            continue
        toks = norm(m).split()
        for t in toks[1:]:                      # 비선행 토큰만
            if len(t) >= 4 and t in corpus_n:
                (gen_pass if _verdict(t, m, t) == "pass" else gen_rej).append((m, t))
                break
    print(f"  비선행 토큰 핀(범주어 축): 기각 {len(gen_rej)} / 통과 {len(gen_pass)}"
          f"  {'✓ R5 목표 달성(전부 기각)' if not gen_pass else '⚠통과 사례: %s' % gen_pass[:5]}")
except Exception as _e:
    print(f"  (시뮬 생략: {_e!r})")

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
