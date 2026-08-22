# -*- coding: utf-8 -*-
r"""x468 — **닫힌 산술 미유도(actual_apy) · 수식 격자 A/B/C/N** (2026-08-22·[[62]] 정보-맞춘·C562 형)

## 왜 (정본 `t7336_tasks/T7336_TASK_093.md`·`T7336_TASK_094.md`·`T7335_NT1_FORENSIC_HALFA`)
종단 write 로 흘러가는 `actual_apy` 가 **유도되지 않는다**(첫 커밋 자리 = 수식 도구 호출):
    093 t0 [41]  actual=2.5  ← ctx [38] KB 문서 축자 *"Balances below the tier 2 threshold earn
                 2.5% APY"* 의 복제 후보(리뷰 MINOR 정정: expected−checking 역산 서술은 추정이었다)
                 · 원장에는 `MONTHLY INTEREST CREDIT 480.0`·잔액 144000 이 [34]/[28] 실재 →
                 480×12/144000=4.0
    093 t1 [37]  actual=4.00 ← **옳게 유도**(대조 결정점 — A 가 여기서도 옳아야 계기가 산다)
    094 t0/t1 [20]  actual=5.0 ← **손님 stated rate 복제**(손님 [1] 축자 *"The base rate is 5.0%"*
                 — 리뷰 MINOR 정정: 고객 산술 408×12/96000=5.1 의 복제가 아니다·5.0≠5.1)
A2 `get_interest_correction.params.actual_apy` 는 그 유도를 문면으로 지시한다(*"Derive it from the
latest MONTHLY INTEREST CREDIT … monthly credit amount x 12 / principal x 100"*) — 도구 스키마에
이미 실려 있는데도 결정점에서 적용되지 않았다. 값 수식 계열(`T2_VALUE_FORMULA`)은 카드 리턴 한
자리만 선언돼 있어 이 필드에는 레버가 없다.

## ⚠가장 위험한 축 — 무엇을 재고 무엇을 재지 않나
2026-08-19 에 **엔진측 값 치환·gold-fit 상수를 걷어냈다**(`_note_compute_ops_removed_2026_08_19`·
t7328 재기준선): 엔진이 채점 인자를 직접 채우면 측정 대상이 사라진다([[62]]). 여기서 재는 것은
*"입력(이자 금액·잔액·기간)을 **모델이 원장에서 형식화**하면 **엔진이 수식으로 계산**"* 하는 C562 형
분담이 **모델 단독 유도보다 나은가**이지, 엔진이 값을 **고르는** 것이 아니다 — 어느 거래가 최근
이자 크레딧인지·어느 숫자가 원금인지는 끝까지 서브(LLM)가 정하고, 엔진은 ⑴인용의 원문 실재
⑵값의 인용 내 실재 ⑶곱셈·나눗셈만 한다(C45 동형·[[59]] ⓐ 허용역). 수식·입력 필드명의 출처는
**A2 선언 축자**(위 param 문장 + `op` 의 월할 상수 + 반환문 "MONTHLY at APY/12")와 **원장 필드명**
(A2 `isolate.instructions` 축자 *"principal = the account's <field>"*)이고, 정책 2편(W 를 이름으로
담은 문서 — `_docs_naming(W)` 도출·"find the interest credit"/"confirm the interest amount credited")
이 그 자리를 뒷받침한다. **gold 수치는 코드·프롬프트 어디에도 없다**([[23]]).

## 팔 (한 변수만 다르다 · 문맥은 t7336 093/094 양 trial 궤적 축자 복원 · 4 결정점)
    A_asis        현행 그대로 — 결정점 직전까지 축자. **A 가 라이브 값(복제값)을 재현해야 측정 유효**
    B_formalize   같은 문맥 + 결정점 직전 도구 출력 끝에 **[VALUE] 블록** 배달:
                  서브(격리·기능 하나=입력 2종 형식화) ← 원장 도구 출력(선언 read 도구의 출력만)
                  + 손님 발화 축자 → {credit, principal} 각각 값+축자 인용 → 엔진이 인용 실재·
                  값-인용 실재 검산 후 `credit × periods / principal × 100` 계산 → 식·피연산자·
                  출처(레코드/손님)를 **함께** 실어 돌려준다. 거부 시 무엇이 없고 **무엇을 읽으면
                  되는지**(선언 read 도구 이름)를 적는다([[64]]).
    C_hint        같은 문맥 + 같은 자리에 **A2 param 문장(수식)만** 문면으로 — 계산은 모델
    N_neg         같은 문맥 + 같은 자리에 **무내용 재촉 한 줄**([[57]] 부정통제)
⚠재생 인터페이스 = **실물 도구 스키마(env + 우리 scaffold_get 선언 → `_build_tool`) + 실제 메시지
  객체 + `la.generate`**(x459/x465 관용구·C584: 렌더-텍스트 인터페이스는 라이브를 재현 못 한다).
⚠결정점 = 각 sim 에서 `actual_apy` 를 인자로 실은 **첫** 호출(= 유도가 처음 커밋되는 수식 도구 호출
  자리 — 리뷰 BLOCK 반영). 이전 판(마지막 호출·W 우선)은 ctx 에 수식 도구 반환문(*"applied(actual)
  APY=… use these APY values …"*)이 이미 실려 A 의 '복제'가 우리 return_template 지시의 이행이 되고
  B/C 배달이 **우리 자신의 문구와 모순**됐으며([[55]] '우리 문구' 층), 유일한 대조점(093 t1)만
  T-호출이라 부호표가 이종 결정점 비교였다. 첫 호출 통일로 4 결정점 전부 같은 유형·ctx 오염 0
  (실측: 093 t0 ctx[:41] 에 'applied(actual)' 0건·[:45] 에는 [42] 1건). 닫힌 술어(역할·인자 키
  존재)만 쓴다 — 발화의 뜻은 판정하지 않는다([[59]]).
⚠system 복원(리뷰 MAJOR·정보-맞춤) = 저장 sim 의 messages 에 system 이 0건(실측)이라 sim['policy']
  (7541자 실재)를 **x466 관용구 `system_dicts`(라이브 빌더 LLMAgent)로 복원해 기본 켬**(--no-system
  은 진단용). 정책 유/무 대조 = A_asis det 1발을 system 없이 한 번 더(`A_nosys` 행·부호표 밖).
⚠정책 문서 2편(*"find the interest credit"*/*"confirm the interest amount credited"*)은 ④에서
  `_docs_naming(W)` 도출·출력만 한다 — **의도적 미배달**(이 프로브의 조작 변수는 수식 문면 전달과
  형식화 분담 둘뿐·문서 배달 축은 x465 계열이 이미 잰다).

## 채점 (닫힌 술어만·[[59]]ⓐ — 다음 tool-call 의 인자 수치)
    ref_match   다음 호출의 `actual_apy` 가 **원장 수치로 계산한 참조값** ±0.01  (참조 = 이 스크립트가
                원장(db.json 구조 레코드 · 없으면 궤적의 원장 도구 출력)에서 읽어 같은 식으로 계산·gold 0)
    off_ref     `actual_apy` 를 실었는데 참조와 다르다   (+ `eq_live` 플래그 = 라이브 복제값과 같다)
    read_first  `actual_apy` 없이 다음 호출(래퍼 안쪽 포함)이 **선언 read 도구** — 실패가 아니라
                **선언-순응**이다(리뷰 MAJOR: A3 param 문장·B 의 REJECT/NOTE([[64]])가 지목한 read 이행)
    no_field    호출은 냈는데 `actual_apy` 도 read 도 아니다 / no_tool 호출 없음 / EXC 예외
    ref_unavail 참조를 원장에서 못 만들었다(094 로컬: 거래 read 가 어느 sim 에도 없다 → 리모트 db.json)
    stale_amount(행 병기 플래그) W 대상 호출이 `actual_apy` 만 갱신하고 금액 필드(compute ∩ W.args −
                T.params 로 도출)가 라이브 구값 그대로면 내부 모순 보고서다([[70]] 판매 항목·리뷰 MINOR)
n = det 1발 + temp 6발 = **7/팔/결정점**(≥7). 성적 확정은 본런 reward 재실행에서만([[69]]).

## [[71]] 격리 서브에이전트 계약 — 4문 답
  1) 기능 하나인가 — 둘 다 하나다. B 의 서브는 **입력 2종(credit·principal)의 형식화** 하나만 하고,
     각 팔의 메인 재생은 **다음 발화/호출 생성** 하나만 한다. 채점은 엔진 밖 닫힌 산술 대조.
  2) 재료가 A2/A3 선언에서 읽혀 나왔나 — 그렇다. 수식 문장·월할 상수·필드 키·대상 도구·read 도구
     ·KB 도구 이름 전부 런타임에 A3(`scaffold_get_tools`·`field_ops`·`arg_source_reads`·
     `function_agents`)와 env 레지스트리(`env_surface.json`)에서 도출한다. id 계열 키(계좌·거래·유저)
     도 리터럴이 아니라 `field_ops.id_ref` 멤버십 + 공급-read 인자 사슬 + param 문장 축자로 도출한다
     (리뷰 MINOR 반영 — 이전 판의 `endswith('account_id')`·`'transaction_id'`·`'user_id'` 3종 제거).
     서브가 받는 원장 = **선언 read 도구의 출력**만(KB 검색 출력·우리 층 피드백 제외) + 손님 발화.
     gold·`reward_info` 는 채점 기준으로 열지 않는다([[23]]·[[69]]).
  3) 전달이 선언된 id → 정확 집기인가 — 그렇다. 서브에 가는 것은 문맥에 **이미 있는** 원장 출력의
     축자(도구명 멤버십으로 집는다). bm25·embedding 0.
  4) 엔진이 해석·선택·순위를 하지 않는가 — 안 한다. 어느 줄이 이자 크레딧이고 어느 수가 원금인지는
     서브가 정한다. 엔진은 인용 실재 검산 + 산술 하나. argmax·최댓값·"정답은 X" 0 — [VALUE] 블록은
     식·피연산자·인용·출처를 병기해 **검산 가능한 산출**이지 지목이 아니다(C562 자리와 동형).

## [[62]] 4문 답
  ① 결손을 격리로 쟀나 — 정본 포렌식이 쟀다(093 61+73msg·094 42+42msg 전수·[S]): 결손 = 결정점에서
     **닫힌 산술을 안 한다**(복제). 이 프로브는 그 결손의 **경계**를 가른다: C(수식만)로 닫히면 전달
     부하, B 에서만 닫히면 산술 분담, B 도 안 닫히면 다른 층.
  ② 격리에서 되면 레버는 전달뿐인가 — C 가 이기면 처방은 **문면 전달**(A2 param 문장의 결정점 표면화)
     뿐이다. B 가 이기고 C 가 지면 분담(형식화→계산)이 정당화되고 그때도 엔진은 **주어진 수의 곱셈**
     한 칸만 늘어난다(`T2_VALUE_FORMULA` 와 같은 크기·값 선택 0).
  ③ 사라지는 모델 판단 0 — 어느 거래·어느 잔액·어느 출처를 쓸지, 그 값을 write 에 실을지는 모델이 한다.
  ④ 엔진 출력에 argmax·최댓값·"정답은 X" 0 — [VALUE] 는 식+피연산자+인용+출처 병기. 거부문은 이름만 댄다.

## [[70]] 병기 (무엇을 파나)
B 가 판 것 = 서브 1호출(지연) + 문맥 +N자 + **엔진이 계산한 수가 문맥에 생긴다** — 서브가 틀린 줄
(이자 아닌 거래·다른 계좌 잔액·손님 주장)을 집으면 그 위에서 우리가 **자신 있게 곱셈**을 해 준다.
완화 = 인용·출처 병기(손님 발화 출처는 **레코드가 아니라고 명시** + 확인할 read 도구 이름). C 가 판
것 = 문맥 +1문장. 추가 판매 항목(리뷰 MINOR): W 대상 재생 호출이 `actual_apy` 만 갱신하면 같은
호출의 금액 필드(라이브 30.0/40.0 — 옛 actual 로 계산된 값)와 **내부 모순**인 보고서가 된다 —
`stale_amount` 닫힌 플래그로 행마다 병기·집계에 노출. 부호표는 (태스크 2 × trial 2) 4행이며
**같은 태스크 반복 = 독립 표본 아님**.
격리 결과는 승격 근거가 아니라 경계 판정이다 — 라이브 효과는 본런 A/B([[69]] reward)가 판정.

## 계기 함수 주의 ([[67]])
`kv_blocks`·`field_calls`·`ledger_texts`·`user_texts` 는 이 프로브의 **계기 전용** 신규 함수다
(레버 경로 아님·[[59]] 무관) — 다음 프로브가 필요로 하면 복사하지 말고 `t2_forensic` 편입이 정본
경로다(사본 분기 실물 2건 전례).

## 실행 (⚠지금 금지 — 유료 t7336 종료·GPU 유휴 후에만 · 리모트 · cwd=tau2-bench 루트 · vLLM @8141)
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    python scripts/distill/tau2/x468_actual_apy_formula_grid.py --port 8141
    # 배선만(LLM 0·GPU 불요): ... x468_actual_apy_formula_grid.py --wiring-only
    #   — 리모트(tau2 有)는 ④(실물 도구 빌드·_docs_naming·system 복원)까지 검증하고 끝난다(리뷰
    #     MAJOR: ④ 가드가 wiring-only 사각이던 유일 fatal 배선이었다). tau2 없는 로컬은 ③까지
    #     검증 후 ④ 생략을 명시하고 EXIT 0.
"""
import argparse
import io
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                  # noqa: E402  정본 로더·인자 평탄화(사본 금지·[[67]])
import t2_subcall as SC                  # noqa: E402  단발 격리 서브·계약 파싱·값 실재 검산 정본
import t2_compute as TC                  # noqa: E402  날짜 파싱 정본(_parse_date)
import x465_transfer_doc_iso as X465     # noqa: E402  관용구 재사용: load_a3·inject·last_tool_idx·replay·NUDGE
from t2_scaffold_get import _norm_ground  # noqa: E402  grounding 정규화 정본

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)
DOMAIN = "banking_knowledge"

# t7336 halfA 실물 (포렌식 정본과 동일·읽기 전용). 리모트 라이브 경로가 있으면 `F.path_for` 가 먼저 본다.
D_RESULTS = "bank_t7336_halfA_20260821b"
# 구조 원장(참조 계산용·리모트). 없으면 궤적의 원장 도구 출력으로 떨어진다(로컬).
D_DB = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/db.json"
TOL = 0.01

# 배달 헤더 — 지시가 재료 **앞**(C578). 어느 값을 쓰라고 말하지 않는다(열린 술어=모델 몫·[[22]]).
VALUE_HEAD = "[VALUE] %s = %s = %s x %d / %s x 100"
VALUE_SRC = "   %s %s <- \"%s\" [%s]"
VALUE_CUSTOMER_NOTE = ("   NOTE: an input above comes from the customer's statement, not from an account "
                       "record; confirm it by reading the records (%s) before relying on it.")
VALUE_REJECT = ("[VALUE] %s could not be derived: %s. Read the records with %s and try again; the "
                "declared derivation is: %s")
# 출처 귀속(리뷰 MINOR): 배달 문장의 주인은 수식 도구(T)의 params 선언이다 — W 의 것으로 제시 금지([[23]]).
HINT_HEAD = "[NOTICE] Before filling %s — declared derivation (verbatim from %s.params.%s): %s"

SYS_FORM = ("You are a records-extraction sub-task for a bank support agent. You do ONE thing: fill "
            "the skeleton below from the material given — nothing else. Copy each quote verbatim "
            "(one line from the material, character for character). If neither the records nor the "
            "customer's messages state an item, omit that item instead of guessing. Reply with "
            "exactly one JSON object and nothing else.")


# ── 선언 읽기 (A3 + env 레지스트리) ─────────────────────────────────────────────
def registry(domain=DOMAIN):
    with io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8") as f:
        return json.load(f)[domain]["tools"]


def _consts(op, out=None):
    """op 트리의 `const` 잎 값들(닫힌 구조 탐색·값 해석 0)."""
    out = [] if out is None else out
    if isinstance(op, dict):
        if op.get("op") == "const":
            out.append(op.get("value"))
        for v in op.values():
            _consts(v, out)
    elif isinstance(op, list):
        for v in op:
            _consts(v, out)
    return out


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")


def derive_decls(a3, reg):
    """이 프로브가 쓰는 이름 전부를 선언에서 **도출**한다(리터럴 0·도출 실패는 멈춘다·[[55]])."""
    compute = set((a3.get("field_ops") or {}).get("compute") or [])
    sg = [d for d in (a3.get("scaffold_get_tools") or []) if isinstance(d, dict)]
    cands = []
    for d in sg:
        ok = set(((d.get("isolate") or {}).get("operand_keys") or [])) & compute
        if ok:
            cands.append((d, ok))
    if len(cands) != 1 or len(cands[0][1]) != 1:
        raise SystemExit("수식 도구 도출 실패: scaffold_get_tools 중 isolate.operand_keys∩field_ops.compute "
                         "가 정확히 1개여야 한다 — %r" % [(d["name"], sorted(k)) for d, k in cands])
    T, f = cands[0][0], sorted(cands[0][1])[0]
    p_set = set(T["isolate"]["operand_keys"]) - compute
    if len(p_set) != 1:
        raise SystemExit("원금 키 도출 실패(operand_keys − compute ≠ 1): %r" % sorted(p_set))
    p = sorted(p_set)[0]
    cs = [c for c in _consts(T.get("op")) if isinstance(c, (int, float)) and 0 < c < 1]
    if len(cs) != 1:
        raise SystemExit("월할 상수 도출 실패(op 의 (0,1) const 가 1개여야): %r" % cs)
    periods = int(round(1.0 / cs[0]))
    tpl = str(T.get("return_template") or "")
    if ("/%d" % periods) not in tpl:
        raise SystemExit("op 상수(1/%d)와 return_template 문면이 어긋난다: %s" % (periods, tpl[:120]))
    sentence = str((T.get("params") or {}).get(f) or "").strip()
    if not sentence:
        raise SystemExit("A3 %s.params.%s 문장이 비어 있다" % (T["name"], f))
    m = re.search(r":\s*([a-z][a-z ]+?)\s+x\s+%d\s*/\s*%s\b" % (periods, re.escape(p)), sentence)
    if not m:
        raise SystemExit("param 문장에서 피제수 이름을 못 읽었다(': <name> x %d / %s' 형태 기대): %s"
                         % (periods, p, sentence[:160]))
    credit_key = _slug(m.group(1))
    caps = re.findall(r"\b(?:[A-Z]{2,}\s+){1,}[A-Z]{2,}\b", sentence)
    if not caps:
        raise SystemExit("param 문장에 대문자 구(거래 설명 축자)가 없다: %s" % sentence[:160])
    phrase = max(caps, key=len).strip()
    ins = str((T.get("isolate") or {}).get("instructions") or "")
    m2 = re.search(r"%s = the account's (\w+)" % re.escape(p), ins)
    if not m2:
        raise SystemExit("isolate.instructions 에서 원금 레코드 필드를 못 읽었다('%s = the account's <field>')" % p)
    principal_field = m2.group(1)
    W = [n for n, t in reg.items() if t.get("mutates") and f in (t.get("args") or [])]
    if len(W) != 1:
        raise SystemExit("종단 write 도출 실패(mutates & %s ∈ args 가 1개여야): %r" % (f, W))
    W = W[0]
    asr = {k: v for k, v in (a3.get("arg_source_reads") or {}).items()
           if not str(k).startswith("_") and isinstance(v, list)}
    reads = sorted({t for lst in asr.values() for t in lst if t in reg and not reg[t].get("mutates")})
    if not reads:
        raise SystemExit("arg_source_reads 에서 read 도구를 0개 도출 — 층 정합부터([[24]])")
    # id 계열 키 = field_ops.id_ref 멤버십(리뷰 MINOR: `endswith('account_id')`·`'transaction_id'`
    # 리터럴 제거 — 이름의 출처를 전부 선언으로).
    id_ref = [k for k in ((a3.get("field_ops") or {}).get("id_ref") or [])]
    w_args = list(reg[W].get("args") or [])
    # ① 종단 write 의 id 계열 인자를 공급하는 read 중, 그 read 자신이 write 시점에 이미 쥔
    #    인자(w_args 부분집합)로 호출 가능한 것(= [[64]] 지목용·선언 순서 그대로)
    fix_reads = []
    for k in w_args:
        if k not in id_ref:
            continue
        for t in (asr.get(k) or []):
            if t in reads and t not in fix_reads and set(reg[t].get("args") or []) <= set(w_args):
                fix_reads.append(t)
    # ② 유도 문장(sentence)이 어간으로 가리키는 id_ref 키(예: transaction)의 공급 read 중
    #    인자가 w_args 부분집합인 것 — 문장 축자에 없는 계열(card 등)은 자연 배제
    low = sentence.lower()
    for k in id_ref:
        stem = str(k).rsplit("_id", 1)[0].replace("_", " ")
        if not stem or stem not in low:
            continue
        for t in (asr.get(k) or []):
            if t in reads and t not in fix_reads and set(reg[t].get("args") or []) <= set(w_args):
                fix_reads.append(t)
    # 계좌/유저 키 구분(리터럴 0): 부모(uid) = 다른 id 키의 공급 read 의 **인자**로 나타나는 id 키
    idks = [k for k in w_args if k in id_ref]
    sup = {k: [t for t in (asr.get(k) or []) if t in reg] for k in idks}
    uidks = [k for k in idks
             if any(k in (reg[t].get("args") or []) for k2 in idks if k2 != k for t in sup[k2])]
    accks = [k for k in idks if k not in uidks]
    # 금액 필드(내부-모순 stale 검사용·리뷰 MINOR): W 인자 중 compute 계열인데 T 의 param 이 아닌 것
    amount_keys = [k for k in w_args if k in compute and k not in (T.get("params") or {})]
    kb = set()
    for fa in (a3.get("function_agents") or []):
        kb |= set(fa.get("wraps") or []) | set(fa.get("getter_tools") or [])
    return {"T": T["name"], "f": f, "p": p, "periods": periods, "sentence": sentence,
            "credit_key": credit_key, "phrase": phrase, "principal_field": principal_field,
            "W": W, "reads": reads, "fix_reads": fix_reads, "kb_tools": sorted(kb),
            "id_ref": id_ref, "uid_key": (uidks[0] if len(uidks) == 1 else None),
            "acc_key": (accks[0] if len(accks) == 1 else None), "amount_keys": amount_keys,
            "scaffold_names": sorted(d["name"] for d in sg), "sg_decls": sg}


# ── 궤적 복원·결정점 ───────────────────────────────────────────────────────────
def load_task_sims(results, suffixes):
    ss = F.sims(results, ".results.json.gz")
    out = [s for s in ss if str(s.get("task_id")).split("_")[-1] in suffixes]
    out.sort(key=lambda s: (str(s.get("task_id")), s.get("trial") or 0))
    if not out:
        raise SystemExit("task %s 의 sim 이 없다 (%s)" % (sorted(suffixes), results))
    return out


def field_calls(msgs, f):
    """`f` 를 인자로 실은 assistant 호출 전부 → [(i, target, value)] (래퍼 안쪽까지 평탄화·정본)."""
    out = []
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            flat = F.flat_args(a)
            if f in flat:
                out.append((i, str(F.inner_name(a) or F.nameof(tc)), flat.get(f), flat))
    return out


def find_dp(msgs, f):
    """결정점 = `f` 를 인자로 실은 **첫** 호출(= 유도가 처음 커밋되는 자리·리뷰 BLOCK 반영).

    이전 판(마지막 호출·W 우선)은 ctx 에 수식 도구 반환문("applied(actual) APY=… use these APY
    values …")이 들어가 측정을 오염시켰다(4개 중 3개) — docstring ⚠결정점 참조. 첫 호출로 통일하면
    4 결정점 전부 같은 유형(수식 도구 호출 자리)이고 ctx 에 그 반환문이 없다(실측 0건)."""
    fc = field_calls(msgs, f)
    if not fc:
        raise SystemExit("`%s` 를 인자로 실은 호출이 궤적에 없다 — 결정점 없음" % f)
    return fc[0], fc


def id2call(msgs):
    """tool_call id → (호출 도구명, 래퍼 안쪽 대상) — 역할·이름만 본다([[03b]])."""
    out = {}
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            d = F._as_dict(tc)
            out[d.get("id")] = (F.nameof(tc), str(F.inner_name(F.argsof(tc)) or ""))
    return out


def ledger_texts(ctx, reads):
    """원장 = **선언 read 도구**의 성공 출력 축자(KB 검색·우리 층 피드백은 도구명 멤버십으로 자연 제외)."""
    names = id2call(ctx)
    out = []
    for m in ctx:
        if str(m.get("role") or "") != "tool":
            continue
        nm, inner = names.get(m.get("id"), ("", ""))
        if nm in F.GRANTS:
            continue                      # unlock/give 는 부여 응답이지 원장이 아니다
        if nm in reads or inner in reads:
            c = str(m.get("content") or "")
            if c.strip() and not c.lstrip().startswith("Error"):
                out.append(c)
    return out


def user_texts(ctx):
    return [str(m.get("content") or "") for m in ctx
            if str(m.get("role") or "") == "user" and str(m.get("content") or "").strip()
            and not (m.get("tool_calls") or [])]


# ── 참조값 (계기·엔진 아님) ─────────────────────────────────────────────────────
def kv_blocks(text):
    """env 가 찍는 `key: value` 레코드 렌더링을 블록 dict 로(계기 전용·레버 경로가 아니다)."""
    blocks, cur = [], None
    for ln in str(text or "").splitlines():
        s = ln.strip()
        m = re.match(r"^(?:\d+\.\s*)?([A-Za-z_][A-Za-z0-9_ ]*?):\s*(.*)$", s)
        if not m:
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        if re.match(r"^\d+\.\s", s) or cur is None:
            cur = {}
            blocks.append(cur)
        cur[k] = v
    return [b for b in blocks if b]


def _date_of(row):
    for k, v in row.items():
        d = TC._parse_date(v)
        if d is not None and re.match(r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", str(v)):
            return d
    return None


def _nums_of(row):
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(str(v).replace("$", "").replace(",", ""))
        except Exception:
            pass
    return out


def reference_from_rows(accounts, txrows, dec, want_account=None):
    """참조 = 선언 구(`phrase`) 를 담은 거래 중 **가장 최근** 것의 유일 수치 × periods / 계좌 원금 × 100.

    accounts: [{...}] 계좌 행 · txrows: [{...}] 거래 행. 판단 0 — 구 포함(축자)·날짜 최대·수치 유일성만.
    여러 계좌가 후보면 결정점 호출의 계좌 id 와 값이 같은 행을 고르고, 그래도 여럿이면 None(모호).
    """
    hits = [r for r in txrows if any(dec["phrase"] in str(v) for v in r.values())]
    if not hits:
        return None, "거래 행에 '%s' 가 0건" % dec["phrase"]
    dated = [(d, r) for r in hits for d in [_date_of(r)] if d is not None]
    latest = max(dated, key=lambda x: x[0])[1] if dated else hits[0]
    nums = {k: v for k, v in _nums_of(latest).items() if _date_of({k: latest[k]}) is None}
    if len(nums) != 1:
        return None, "최근 거래 행의 수치 필드가 1개가 아니다: %r" % nums
    credit = list(nums.values())[0]
    acc = [a for a in accounts if dec["principal_field"] in a
           and any(str(v) == str(x) for v in a.values() for x in latest.values())]
    if want_account is not None:
        acc2 = [a for a in acc if any(str(v) == str(want_account) for v in a.values())]
        acc = acc2 or acc
    if len(acc) != 1:
        return None, "거래 행과 잇닿는 계좌 행이 %d개(모호)" % len(acc)
    try:
        principal = float(str(acc[0][dec["principal_field"]]).replace("$", "").replace(",", ""))
    except Exception:
        return None, "원금 필드 %s 를 수치로 못 읽음" % dec["principal_field"]
    if not principal:
        return None, "원금 0"
    return ({"ref": credit * dec["periods"] / principal * 100.0, "credit": credit,
             "principal": principal, "tx": latest, "account": acc[0]}, "ok")


def reference(task_sims, dec, db_path, want_account, want_user):
    """구조 원장(db.json·리모트) 우선 → 없으면 그 태스크 **모든 sim** 의 원장 도구 출력(로컬)."""
    if db_path and os.path.exists(db_path):
        import x394_calc_formalize_iso as X394
        db = X394.load_db() if db_path == os.path.join(X394.BK, "db.json") else \
            json.load(io.open(db_path, encoding="utf-8", errors="replace"))
        uid = want_user
        if not uid:
            return None, "db.json 참조에는 user_id 가 필요하다(결정점 호출에 없음)"
        accts, tx = X394.records_for(db, uid)
        rows = []
        for aid, v in tx.items():
            items = v if isinstance(v, list) else (list(v.values()) if isinstance(v, dict) else [])
            for r in items:
                if isinstance(r, dict):
                    rows.append(dict(r, **{"__account": aid}))
        arows = [dict(a, **{"__account": aid}) for aid, a in accts.items()]
        got, why = reference_from_rows(arows, rows, dec, want_account)
        return got, "db.json · " + why
    texts = []
    for s in task_sims:
        texts += ledger_texts(s.get("messages") or [], dec["reads"])
    blocks, seen = [], set()
    for t in texts:
        for b in kv_blocks(t):
            key = tuple(sorted(b.items()))
            if key not in seen:           # 같은 레코드가 여러 sim 에 찍힌다 — 중복은 모호가 아니다
                seen.add(key)
                blocks.append(b)
    accounts = [b for b in blocks if dec["principal_field"] in b]
    txrows = [b for b in blocks if dec["principal_field"] not in b]
    got, why = reference_from_rows(accounts, txrows, dec, want_account)
    return got, "궤적 원장(%d sim·%d출력·%d블록) · %s" % (len(task_sims), len(texts), len(blocks), why)


# ── B 팔: 서브 형식화 → 엔진 계산 ───────────────────────────────────────────────
def sub_prompt(dec, ledger, users):
    skel = ('{"%s": {"value": <number>, "quote": "<exact line>"}, '
            '"%s": {"value": <number>, "quote": "<exact line>"}}' % (dec["credit_key"], dec["p"]))
    return ("# Declared derivation of %s (verbatim from the tool declaration)%s%s%s%s"
            "# Account records and transactions read in this conversation (verbatim tool outputs)%s%s%s%s"
            "# What the customer said (verbatim)%s%s%s%s"
            "# Fill this exact skeleton (omit an item only if neither source states it)%s%s%s"
            % (dec["f"], NLC, dec["sentence"], NLC, NLC,
               NLC, (NLC + "---" + NLC).join(ledger) if ledger else "(none read yet)", NLC, NLC,
               NLC, (NLC + "---" + NLC).join(users) if users else "(nothing)", NLC, NLC,
               NLC, skel, NLC))


def _provenance(quote, ledger, users):
    q = _norm_ground(quote)
    if not q:
        return None
    if any(q in _norm_ground(t) for t in ledger):
        return "account record"
    if any(q in _norm_ground(t) for t in users):
        return "customer's statement"
    return None


def engine_value(form, dec, ledger, users):
    """엔진 = 닫힌 검산 둘 + 산술 하나. 반환 (배달 텍스트, ok, 상세). 값 선택·순위 0."""
    det = {}
    missing = []
    for key in (dec["credit_key"], dec["p"]):
        it = (form or {}).get(key) if isinstance(form, dict) else None
        if not isinstance(it, dict):
            missing.append("%s (not provided)" % key)
            continue
        v, q = it.get("value"), str(it.get("quote") or "")
        try:
            fv = float(str(v).replace("$", "").replace(",", ""))
        except Exception:
            missing.append("%s (value is not a number)" % key)
            continue
        src = _provenance(q, ledger, users)
        if src is None:
            missing.append("%s (its quote is not found verbatim in the records or the customer's messages)" % key)
            continue
        if not SC.val_grounded(fv, [q]):
            missing.append("%s (the number is not inside its own quote)" % key)
            continue
        det[key] = {"value": fv, "quote": " ".join(q.split()), "source": src}
    reads = ", ".join(dec["fix_reads"] or dec["reads"])
    if missing:
        return (VALUE_REJECT % (dec["f"], "; ".join(missing), reads, dec["sentence"]), False,
                {"accepted": det, "missing": missing})
    c, p = det[dec["credit_key"]]["value"], det[dec["p"]]["value"]
    if not p:
        return (VALUE_REJECT % (dec["f"], "%s is 0" % dec["p"], reads, dec["sentence"]), False,
                {"accepted": det, "missing": ["principal 0"]})
    apy = c * dec["periods"] / p * 100.0
    lines = [VALUE_HEAD % (dec["f"], ("%.4f" % apy).rstrip("0").rstrip("."), c, dec["periods"], p)]
    for key in (dec["credit_key"], dec["p"]):
        d0 = det[key]
        lines.append(VALUE_SRC % (key, d0["value"], d0["quote"][:160], d0["source"]))
    if any(d0["source"] != "account record" for d0 in det.values()):
        lines.append(VALUE_CUSTOMER_NOTE % reads)
    return NLC.join(lines), True, {"accepted": det, "apy": apy}


def engine_selftest(dec):
    """엔진 경로 자체검정(합성 재료·도메인 값 0): 출처 라벨 2종·거부 2종·산술 1종. 실패 = 배선 중단."""
    led = ["row: one" + NLC + "cred: 7.0" + NLC + "held: 1400.00"]
    usr = ["I got 7 and I hold 1400 with you"]
    ck, pk = dec["credit_key"], dec["p"]
    want = 7.0 * dec["periods"] / 1400.0 * 100.0
    t1, ok1, i1 = engine_value({ck: {"value": 7.0, "quote": "cred: 7.0"},
                                pk: {"value": 1400, "quote": "held: 1400.00"}}, dec, led, usr)
    t2, ok2, i2 = engine_value({ck: {"value": 7, "quote": "I got 7 and I hold 1400"},
                                pk: {"value": 1400, "quote": "held: 1400.00"}}, dec, led, usr)
    t3, ok3, i3 = engine_value({ck: {"value": 7.0, "quote": "cred: 9.0"},
                                pk: {"value": 1400, "quote": "held: 1400.00"}}, dec, led, usr)
    t4, ok4, i4 = engine_value({ck: {"value": 8.0, "quote": "cred: 7.0"},
                                pk: {"value": 1400, "quote": "held: 1400.00"}}, dec, led, usr)
    checks = [("record·arith", ok1 and abs(i1.get("apy", 0) - want) < 1e-9 and "NOTE" not in t1),
              ("customer·NOTE", ok2 and "NOTE" in t2 and "customer" in t2),
              ("quote 미실재→거부+read 지목", (not ok3) and all(r in t3 for r in dec["fix_reads"])),
              ("값≠인용→거부", (not ok4) and "not inside its own quote" in t4)]
    for name, ok in checks:
        print("   [selftest] %s %s" % ("PASS" if ok else "**FAIL**", name))
    return all(ok for _n, ok in checks)


def sub_formalize(agent, la, UserMessage, dec, ledger, users):
    raw = SC.sub_generate(agent, la, UserMessage, SYS_FORM + NLC + NLC + sub_prompt(dec, ledger, users),
                          "x468_formalize", temperature=0.0)
    return SC.parse_contract(raw), raw


# ── 채점 ──────────────────────────────────────────────────────────────────────
def _num(v):
    try:
        return float(str(v).replace("%", "").replace("$", "").replace(",", ""))
    except Exception:
        return None


def classify(calls, dec, ref, live, live_amt=None):
    """다음 tool-call 의 `f` 수치 ↔ 참조(±TOL). 닫힌 산술 대조뿐.

    반환 (cat, value, target, eq_live, stale_amount).
    read_first(리뷰 MAJOR): `f` 없이 다음 호출(래퍼 안쪽 포함)이 선언 read 도구 멤버십이면
    실패(no_field)가 아니라 **선언-순응**으로 따로 센다 — B/C 의 REJECT/NOTE([[64]])가 지목한
    read 이행을 실패로 접으면 부호표가 역방향으로 왜곡된다.
    stale_amount(리뷰 MINOR): W 대상 호출이 `f` 만 갱신(eq_live 아님)하고 금액 필드가 라이브
    구값 그대로면 내부 모순 보고서 — [[70]] 판매 항목의 닫힌 플래그."""
    got = []
    for nm, ag in calls:
        flat = F.flat_args(ag or {})
        if dec["f"] in flat:
            got.append((str(F.inner_name(ag or {}) or nm), flat.get(dec["f"]), flat))
    if not calls:
        return "no_tool", None, None, False, False
    if not got:
        eff = [str(F.inner_name(ag or {}) or nm) for nm, ag in calls]
        cat = "read_first" if any(e in set(dec["reads"]) for e in eff) else "no_field"
        return cat, None, ",".join(eff), False, False
    pref = [g for g in got if g[0] == dec["W"]] or got
    target, v, flat = pref[0]
    fv = _num(v)
    if fv is None:
        return "no_field", v, target, False, False
    lv = _num(live)
    eq_live = lv is not None and abs(fv - lv) <= 1e-3
    stale = False
    if target == dec["W"] and dec.get("amount_keys") and live_amt is not None and not eq_live:
        av = _num(flat.get(dec["amount_keys"][0]))
        stale = av is not None and abs(av - live_amt) <= 1e-3
    if ref is None:
        return "ref_unavail", fv, target, eq_live, stale
    return ("ref_match" if abs(fv - ref) <= TOL else "off_ref"), fv, target, eq_live, stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--results", default=D_RESULTS, help="런 태그 또는 results 경로(F.path_for)")
    ap.add_argument("--tasks", default="093,094")
    ap.add_argument("--db", default=D_DB, help="구조 원장(참조 계산·없으면 궤적 원장)")
    ap.add_argument("--n", type=int, default=6, help="temp 표본 수 — det 1발이 늘 앞선다(합 n+1≥7)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--arms", default="A_asis,B_formalize,C_hint,N_neg")
    ap.add_argument("--no-system", action="store_true",
                    help="라이브 system(sim['policy']·x466 관용구) 복원 끄기 — 진단용일 뿐 기본 유효 조건 아님")
    ap.add_argument("--wiring-only", action="store_true",
                    help="배선 검증만(LLM 0·GPU 불요·[[55]]) — 리모트는 ④ 도구 빌드까지 검증")
    ap.add_argument("--out", default="x468_actual_apy_formula_grid.json")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    # ── ① 선언 도출 (리터럴 0) ──────────────────────────────────────────────────
    a3 = X465.load_a3()
    reg = registry()
    dec = derive_decls(a3, reg)
    print("=" * 100)
    print("x468 · 수식 도구=%s · 대상 필드=%s · 원금 키=%s · periods=%d · 종단 write=%s"
          % (dec["T"], dec["f"], dec["p"], dec["periods"], dec["W"]))
    print("  수식 문장(A3 param 축자): %s" % dec["sentence"][:200])
    print("  피제수 키=%s · 거래 구(축자)='%s' · 원금 레코드 필드=%s"
          % (dec["credit_key"], dec["phrase"], dec["principal_field"]))
    print("  read 도구(선언·%d): %s" % (len(dec["reads"]), ", ".join(dec["reads"])))
    print("  [[64]] 지목 read: %s · KB 도구(제외): %s" % (", ".join(dec["fix_reads"]), ", ".join(dec["kb_tools"])))
    print("  id 키 도출(id_ref 멤버십·리터럴 0): 계좌=%s · 유저=%s · 금액 필드(stale 검사)=%s"
          % (dec["acc_key"], dec["uid_key"], ", ".join(dec["amount_keys"]) or "(없음)"))

    # ── ② 궤적 복원·결정점·참조 ────────────────────────────────────────────────
    tasks = {x.strip() for x in a.tasks.split(",") if x.strip()}
    sims = load_task_sims(a.results, tasks)
    cases = []
    for s in sims:
        msgs = s.get("messages") or []
        (i_dp, target, live, flat), fc = find_dp(msgs, dec["f"])
        ctx = msgs[:i_dp]

        def _val_of(key, _msgs=msgs, _flat=flat):
            """id 값 집기(결정점 호출 → 없으면 그 궤적 첫 등장) — 키는 선언 도출(acc_key/uid_key)."""
            if key is None:
                return None
            v = _flat.get(key)
            if v:
                return v
            for _i, _t, _v, _fl in field_calls(_msgs, key):
                return _v
            return None
        want_acc = _val_of(dec["acc_key"])
        want_uid = _val_of(dec["uid_key"])
        # 라이브 종단 write 의 금액 구값(stale_amount 대조용·리뷰 MINOR) — 궤적 전체의 마지막 W 호출
        live_amt = None
        w_live = [c0 for c0 in fc if c0[1] == dec["W"]]
        if w_live and dec["amount_keys"]:
            live_amt = _num(w_live[-1][3].get(dec["amount_keys"][0]))
        same_task = [x for x in sims if x.get("task_id") == s.get("task_id")]
        ref, how = reference(same_task, dec, a.db, want_acc, want_uid)
        led = ledger_texts(ctx, dec["reads"])
        usr = user_texts(ctx)
        i_tool = X465.last_tool_idx(ctx)
        cases.append({"sim": s, "task": s.get("task_id"), "trial": s.get("trial"), "i_dp": i_dp,
                      "target": target, "live": live, "live_amt": live_amt, "ctx": ctx, "ref": ref,
                      "ref_how": how, "ledger": led, "users": usr, "i_tool": i_tool,
                      "n_field_calls": len(fc)})
        print(NLC + "── %s t%s · msgs=%d · 결정점=[%d] %s(%s=%r) · 필드호출 %d · 배달 자리=[%s]"
              % (s.get("task_id"), s.get("trial"), len(msgs), i_dp, target, dec["f"], live, len(fc), i_tool))
        print("   원장 출력 %d건(%d자) · 손님 발화 %d건 · 참조: %s"
              % (len(led), sum(len(t) for t in led), len(usr),
                 ("%.4f (= %s x %d / %s x 100)" % (ref["ref"], ref["credit"], dec["periods"], ref["principal"])
                  if ref else "UNAVAILABLE")))
        print("   참조 경로: %s" % how)
        if ref is not None:
            try:
                lv = float(str(live).replace("%", ""))
                print("   라이브 값 %s ↔ 참조 %.4f → %s" % (live, ref["ref"],
                      "일치(대조 결정점)" if abs(lv - ref["ref"]) <= TOL else "불일치(실패 재현 표적)"))
            except Exception:
                pass

    # 라이브 system 복원 전제(리뷰 MAJOR·정보-맞춤): 저장 messages 에 system 0건 → sim['policy'] 필수
    pol = str(cases[0]["sim"].get("policy") or "")
    n_sys_msgs = sum(1 for c in cases for m in c["sim"].get("messages") or []
                     if str(m.get("role") or "") == "system")
    print(NLC + "[배선] system 복원(x466 관용구·기본 켬): 저장 system 메시지 %d건 · sim['policy'] %d자 · %s"
          % (n_sys_msgs, len(pol), "OFF(--no-system·진단용)" if a.no_system else "ON"))
    if not pol and not a.no_system:
        raise SystemExit("sim['policy'] 가 비어 있어 라이브 system 을 복원할 수 없다 — "
                         "--no-system 은 진단용일 뿐 기본 유효 조건이 아니다")

    # ── ③ 배선 선검증 (LLM 0) — 엔진 dry-run·프롬프트 조립 ───────────────────────
    print(NLC + "[배선] 팔 프롬프트 조립 (LLM 0):")
    for c in cases:
        if c["i_tool"] is None:
            print("   ⚠%s t%s: 배달 자리 없음 — B/C/N 불가" % (c["task"], c["trial"]))
            continue
        hint = HINT_HEAD % (dec["f"], dec["T"], dec["f"], dec["sentence"])
        c["ctx_C"] = X465.inject(c["ctx"], hint)
        c["ctx_N"] = X465.inject(c["ctx"], X465.NUDGE)
        sp = sub_prompt(dec, c["ledger"], c["users"])
        c["sub_prompt_chars"] = len(sp)
        # dry-run: 참조 추출값을 서브 산출 **모양**으로 넣어 검산·산술 경로만 확인(모델 산출 아님)
        dry = None
        if c["ref"] is not None:
            tx, acc = c["ref"]["tx"], c["ref"]["account"]
            tline = next((ln for t in c["ledger"] for ln in t.splitlines()
                          if str(c["ref"]["credit"]) in ln or ("%.1f" % c["ref"]["credit"]) in ln), "")
            pline = next((ln for t in c["ledger"] for ln in t.splitlines()
                          if dec["principal_field"] in ln and str(acc.get(dec["principal_field"])) in ln), "")
            if tline and pline:
                form = {dec["credit_key"]: {"value": c["ref"]["credit"], "quote": tline.strip()},
                        dec["p"]: {"value": c["ref"]["principal"], "quote": pline.strip()}}
                dry_txt, ok, info = engine_value(form, dec, c["ledger"], c["users"])
                dry = {"ok": ok, "apy": info.get("apy"), "text": dry_txt}
        c["dry"] = dry
        print("   %s t%s: C(+%d자) N(+%d자) B 서브 프롬프트 %d자 · 엔진 dry-run %s"
              % (c["task"], c["trial"], len(hint), len(X465.NUDGE), len(sp),
                 ("PASS apy=%.4f" % dry["apy"] if dry and dry["ok"] else
                  ("FAIL: " + dry["text"][:100]) if dry else "(참조/원장 없음 → 생략)")))
        if dry and dry["ok"]:
            print("      " + dry["text"].replace(NLC, NLC + "      "))
    n_ref = sum(1 for c in cases if c["ref"] is not None)
    print("[배선] 결정점 %d · 참조 확보 %d · 참조 없음 %d(리모트 db.json 으로 확보) · 팔 %s"
          % (len(cases), n_ref, len(cases) - n_ref, ",".join(arms)))
    print("[배선] 엔진 자체검정(합성 재료):")
    if not engine_selftest(dec):
        print("⛔엔진 자체검정 실패 — B 팔이 레버 실물과 다르다. 여기서 멈춘다([[55]]).")
        return 1

    # ── ④ 실물 도구 + system 복원 (tau2 필요 — wiring-only 도 여기까지 검증·리뷰 MAJOR) ──
    try:
        import x448_index_vs_all_iso as IVA
        import tau2.agent.llm_agent as la
        from tau2.data_model.message import UserMessage
        from tau2.environment.tool import Tool
        import t2_scaffold_get as SG
    except ImportError as e:
        if a.wiring_only:
            print(NLC + "[배선] wiring-only 종료 — LLM 0·GPU 0 (⚠로컬: tau2 미설치라 ④ 도구 빌드·"
                  "_docs_naming·system 복원은 리모트 wiring-only 에서 검증: %r)" % (e,))
            return 0
        raise
    sb = IVA.Sandbox()
    tools = list(sb.env.get_tools() or [])
    names = {t.name for t in tools}
    # 가드 완화(리뷰 MAJOR): 실물 궤적에서 W·read 는 unlock/call 래퍼 경유였다('Tool unlocked:' 축자)
    # — 직접 노출을 요구하면 discoverable 지형의 리모트 첫 실행이 죽는다. 필요 도구는 이미 env
    # 레지스트리에서 **도출**됐으므로(멤버십 자명), 직접 노출 ∨ 래퍼 가용이면 통과.
    need = sorted({dec["W"]} | set(dec["reads"]))
    indirect = [t for t in need if t not in names]
    wrappers = sorted(n for n in names if n in set(F.GRANTS) | set(F.EXECS))
    if indirect and not wrappers:
        raise SystemExit("직접 노출도 래퍼(unlock/call)도 없는 도구: %r (래퍼 0)" % indirect)
    if indirect:
        print("  래퍼 경유 도구 %d종(직접 미노출·래퍼 %s 가용): %s"
              % (len(indirect), ", ".join(wrappers) or "-", ", ".join(indirect)))
    for d in dec["sg_decls"]:                     # 라이브와 같은 우리 scaffold_get 스키마
        try:
            tools.append(SG._build_tool(Tool, d))
        except Exception as e:
            print("  ⚠scaffold 도구 빌드 실패 %s: %r" % (d.get("name"), e))
    base = "http://localhost:%d/v1" % a.port
    agent = types.SimpleNamespace(llm="openai/%s" % a.model,
                                  llm_args={"api_base": base, "api_key": "dummy"})
    try:
        import t2_gate_patch as GP
        import t2_search as TS
        corpus = TS.corpus_from_env(sb.env)
        dids = sorted(GP._docs_naming(dec["W"], os.environ.get("T2_KB_DOCS_DIR") or "", corpus=corpus) or ())
        print("  정책 문서(W 를 이름으로 담은 문서·`_docs_naming`): %s — 도출·출력만, 어느 팔에도 "
              "배달하지 않는다(의도적 미배달·docstring ⚠)" % (", ".join(dids) or "(0편)"))
    except Exception as e:
        print("  ⚠정책 문서 도출 생략: %r" % (e,))
    print("  재생 인터페이스: 실물 도구 %d종(env %d + scaffold %d) + 실제 메시지 객체 + la.generate (C584)"
          % (len(tools), len(names), len(tools) - len(names)))
    # 라이브 system 복원(리뷰 MAJOR — x466 관용구 재사용·[[67]] 사본 금지)
    sys_msgs, sys_src = [], "없음(--no-system)"
    if not a.no_system:
        import x466_id_resolution_iso as X466
        sys_msgs, sys_src = X466.system_dicts(pol, tools, a.model)
    print("  system=%s · %d메시지 · %d바이트" % (sys_src, len(sys_msgs),
          sum(len(m["content"].encode("utf-8")) for m in sys_msgs)))
    if a.wiring_only:
        print(NLC + "[배선] wiring-only 종료 — LLM 0·GPU 0 (④ 도구 빌드·_docs_naming·system 복원까지 검증)")
        return 0

    # ── ⑤ 재생 — 결정점 × 팔 × (det 1 + temp n) ─────────────────────────────────
    rows = []
    for c in cases:
        print(NLC + "=" * 100)
        print("%s t%s · 결정점 [%d] · 참조 %s" % (c["task"], c["trial"], c["i_dp"],
              ("%.4f" % c["ref"]["ref"]) if c["ref"] else "UNAVAILABLE"))
        if "B_formalize" in arms and c["i_tool"] is not None:
            form, raw = sub_formalize(agent, la, UserMessage, dec, c["ledger"], c["users"])
            vtxt, ok, info = engine_value(form, dec, c["ledger"], c["users"])
            c["ctx_B"] = X465.inject(c["ctx"], vtxt)
            c["sub"] = {"form": form, "raw": raw[:1500], "ok": ok, "info": info, "text": vtxt}
            print("  [B 서브] %s%s" % ("OK" if ok else "REJECT", NLC + "  " + vtxt.replace(NLC, NLC + "  ")))
        for arm in arms:
            cx = {"A_asis": c["ctx"], "B_formalize": c.get("ctx_B"), "C_hint": c.get("ctx_C"),
                  "N_neg": c.get("ctx_N")}.get(arm)
            if cx is None:
                print("  ── %s 생략(배달 자리 없음)" % arm)
                continue
            delta = sum(len(str(m.get("content") or "")) for m in cx) - \
                sum(len(str(m.get("content") or "")) for m in c["ctx"])
            print("  ── %s (델타 +%d자)" % (arm, delta))
            for k, t in enumerate([0.0] + [a.temperature] * a.n):
                try:
                    # system 은 dict 로 앞에 끼운다 — x465.replay 의 CLS(role=system) 가 복원(리뷰 MAJOR)
                    # ⛔이름으로 읽는다 — 언팩 개수 하드코딩이 x466 을 전 표본 EXC 로 죽였다(2026-08-22).
                    _r = X465.replay(list(sys_msgs) + cx, tools, a.model, base, t)
                    calls, text, dropped = _r.calls, _r.text, _r.dropped
                except Exception as e:
                    print("    #%d t=%.1f EXC %r" % (k, t, e))
                    rows.append({"task": c["task"], "trial": c["trial"], "arm": arm, "k": k, "temp": t,
                                 "cat": "EXC", "err": repr(e)[:200]})
                    continue
                cat, v, target, eq_live, stale = classify(
                    calls, dec, c["ref"]["ref"] if c["ref"] else None, c["live"], c["live_amt"])
                rows.append({"task": c["task"], "trial": c["trial"], "arm": arm, "k": k, "temp": t,
                             "cat": cat, "value": v, "target": target, "eq_live": eq_live,
                             "stale_amount": stale,
                             "calls": [[nm, json.dumps(ag, ensure_ascii=False, default=str)[:500]]
                                       for nm, ag in calls],
                             "dropped_msgs": dropped, "delta_chars": delta,
                             "text": " ".join(text.split())[:300]})
                print("    #%d t=%.1f  %-12s %s=%-8s eq_live=%-5s target=%s%s%s"
                      % (k, t, cat, dec["f"], v, eq_live, target or "-",
                         "  ⚠stale_amount(금액 구값·내부모순)" if stale else "",
                         ("  ⚠복원 누락 %d" % dropped) if dropped else ""))
        # 정책 유/무 대조(리뷰 MAJOR — det 1발·A 문맥·system 없이): 부호표 밖 `A_nosys` 행
        if sys_msgs and "A_asis" in arms:
            try:
                _r = X465.replay(c["ctx"], tools, a.model, base, 0.0)
                calls, text, dropped = _r.calls, _r.text, _r.dropped
                cat, v, target, eq_live, stale = classify(
                    calls, dec, c["ref"]["ref"] if c["ref"] else None, c["live"], c["live_amt"])
                rows.append({"task": c["task"], "trial": c["trial"], "arm": "A_nosys", "k": 0,
                             "temp": 0.0, "cat": cat, "value": v, "target": target,
                             "eq_live": eq_live, "stale_amount": stale, "dropped_msgs": dropped,
                             "delta_chars": 0, "text": " ".join(text.split())[:300]})
                print("  ── A_nosys(정책 유/무 대조·det 1발): %-12s %s=%s eq_live=%s"
                      % (cat, dec["f"], v, eq_live))
            except Exception as e:
                print("  ── A_nosys EXC %r" % (e,))
                rows.append({"task": c["task"], "trial": c["trial"], "arm": "A_nosys", "k": 0,
                             "temp": 0.0, "cat": "EXC", "err": repr(e)[:200]})

    # ── ⑥ 집계 — 팔 × 닫힌 분류 + 태스크별 부호표 + [[70]] 병기 ───────────────────
    cats = ["ref_match", "off_ref", "read_first", "no_field", "no_tool", "ref_unavail", "EXC"]
    arms_out = arms + (["A_nosys"] if any(r["arm"] == "A_nosys" for r in rows) else [])
    print(NLC + "=" * 100)
    print("%-12s %s  eq_live  stale  (n/팔)" % ("팔", " ".join("%-12s" % c for c in cats)))
    for arm in arms_out:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        print("%-12s %s  %-7s %-6s %d" % (arm, " ".join(
            "%-12s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == c), len(rs))) for c in cats),
            "%d/%d" % (sum(1 for r in rs if r.get("eq_live")), len(rs)),
            "%d" % sum(1 for r in rs if r.get("stale_amount")), len(rs)))
    keys = sorted({(r["task"], r["trial"]) for r in rows})
    print(NLC + "[결정점별 ref_match] (같은 태스크 반복 = 독립 표본 아님·[[70]] 부호표 · 4점 모두 "
          "같은 유형=첫 수식-도구 호출 자리)")
    print("%-12s %s" % ("팔", " ".join("%-14s" % ("%s#%s" % (t[-3:], tr)) for t, tr in keys)))
    for arm in arms_out:
        cells = []
        for t, tr in keys:
            rs = [r for r in rows if r["arm"] == arm and r["task"] == t and r["trial"] == tr]
            cells.append("%-14s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == "ref_match"), len(rs))
                                    if rs else "-"))
        print("%-12s %s" % (arm, " ".join(cells)))
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"decls": {k: v for k, v in dec.items() if k != "sg_decls"},
                   "system": {"src": sys_src, "n": len(sys_msgs)},
                   "cases": [{"task": c["task"], "trial": c["trial"], "i_dp": c["i_dp"],
                              "target": c["target"], "live": c["live"], "live_amt": c["live_amt"],
                              "ref": (c["ref"]["ref"] if c["ref"] else None), "ref_how": c["ref_how"],
                              "sub": c.get("sub"), "ledger_chars": sum(len(t) for t in c["ledger"])}
                             for c in cases],
                   "rows": rows}, f, ensure_ascii=False, indent=1, default=str)
    print(NLC + "판정: A 가 라이브 복제값(eq_live)을 재현해야 격리가 산다(093 t1 은 대조점·A 도 참조 일치 기대).")
    print("      C 에서 ref_match 가 오르고 N 이 A 와 같으면 **원인은 문면 미전달**(레버=전달만).")
    print("      B 만 오르면 **형식화→계산 분담**이 경계를 넘는다 · B 도 안 오르면 다른 층([[62]]②).")
    print("      B/C 의 read_first 증가는 **선언-순응**(A3 문장·REJECT/NOTE 가 지목한 read 이행)이지 "
          "실패가 아니다([[64]]·리뷰 MAJOR).")
    print("      A_nosys(det 1발·부호표 밖)가 A_asis det 와 갈리면 결과가 정책 문면 존재에 민감하다는 "
          "뜻 — C '문면 미전달' 결론을 그만큼 좁혀 읽는다.")
    print("[[70]] 병기: 부호표 4행(태스크 2×trial 2·독립 아님). B 가 판 것 = 서브 1호출 + 문맥 +N자 + "
          "엔진 계산값이 문맥에 생김(서브가 틀린 줄을 집으면 그 위에서 곱셈). W 대상 호출의 "
          "stale_amount(금액 구값 유지=내부 모순 보고서)도 판매 항목으로 병기. 성적 확정은 본런 "
          "reward([[69]]).")
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
