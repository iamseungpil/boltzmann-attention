# -*- coding: utf-8 -*-
r"""x467 — **정책 미독 불리언/범주 — 정의 문서 전달이 판정을 바꾸나** 격리 A/B/N (2026-08-22·[[62]] 정보-맞춘)

## 왜 (정본 `t7336_tasks/T7336_TASK_040.md`·`T7336_FORENSIC_HALFB_2026_08_22.md` §4.6)
t7336 halfB 의 write 태스크 두 건이 **같은 결손 부류**로 죽었다 — 정책/서명 read 를 생략한 채 인자값을
임의 결정(아래 문서 id·수치는 전부 **provenance 인용**이다 — 코드 경로 리터럴 0·§리터럴 참조):
    040 t0  `eligible_for_provisional_credit` 불리언 ×6 전부 틀림 — Provisional Credit Guidelines(015)·
            dispute history(7291) 미독. ⚠뷰 사실(리뷰 BLOCK②): 015 전문은 [22] bm25 출력(29,854자)에
            커밋 히스토리로는 축자 실재했지만, 그 자리는 결정점 [49]의 27 메시지 앞·keep_recent 6 밖이라
            라이브 **생성 뷰**에서는 head300+tail150 다이제스트로 접혀 있었다 — 라이브 모델은 [49]에서
            015 전문을 **보지 못했다**. "이미 실재"는 커밋 히스토리 사실이지 뷰 사실이 아니다.
    085 t1  `dispute_category:"General"` 날조 — 정본 031 본문 미열람([3]·[6] 의 id 목록에는 실려 있었음).
            서명이 선언한 enum(6281 unlock 출력 [10])에 "General" 은 없다.
⛔016 t0(referral·diagnose 형)은 이 프로브에서 **뺐다**(리뷰 BLOCK①): 라이브 [22] 는 메인 문맥 생성이
    아니라 `T2_ACTION_SUB` **서브 격리** 발화다 — `usage.prompt_tokens` = 908(앞 메인 턴들은
    10,786~14,299)·재료 = 손님 발화 3건 + T2_DIAG `diagnosed_text` 뿐. 메인-문맥 재생으로는 인터페이스·
    재료가 모두 달라 정보-맞춤이 성립하지 않는다. referral 축은 x464(SEARCH_AGENT 재무장) 계열로 보낸다.
    이 파일은 kind=write 만 받는다(다른 kind 는 SystemExit).
x465 가 **이관 사슬**에서 "문서 전달이 산다"(A 7/7 일반 ↔ B 6/7 사슬)를 실증했지만, 그 배선
`require_doc_before.tools` 는 **이관 도구 4종만** 선언돼 있다. 질문: **dispute write 에서도**
정의 문서 전문 배달이 불리언/범주 판정을 바꾸는가 —
    "안 읽어서"   → B 에서 판정·행동이 달라진다 ⇒ 레버는 **전달(부하 축소)뿐**으로 산다
    "읽어도 실패" → B 도 A 와 같다 ⇒ 경계 — 전달로는 안 닫히고 처방은 다른 층이다
N_neg 이 A 와 같아야 "결정점에 뭐라도 찔러서"가 아니라 **문서 내용**이 원인이라 말할 수 있다([[57]]).

## 팔 (한 변수만 다르다 · 문맥 = 라이브 **생성 뷰** 재구성·리뷰 BLOCK②)
원시 커밋 히스토리는 라이브 생성 뷰가 아니다 — 라이브 프롬프트는 system + 압축 뷰다. 팔 문맥 조립:
    ① 궤적 dict → tau2 메시지 객체(정본 `x465.to_objs` — replay 와 같은 변환)
    ② `GP._compact_view(objs)` — 라이브 call-site 와 **같은 env·기본값**(keep_recent 6·min_len 800·
       min_total 60,000·msg_cap 8,000). t7336 에서 켜져 있었음을 런 스크립트로 확인:
       `run_t7336_repaired_stage1_20260821.sh:91` 이 `go_stack.sh` 를 source 하고 go_stack.sh:100 이
       `T2_VIEW_COMPACT=1`, :136 `T2_VIEW_COMPACT_MINTOTAL=60000`, :137 `T2_VIEW_MSG_CAP=8000`.
    ③ system 선두 삽입 — tau2 `LLMAgent` 의 조립식 `SYSTEM_PROMPT.format(domain_policy=sim['policy'],
       agent_instruction=AGENT_INSTRUCTION)` (텍스트 사본 0). 궤적 role 에 system 이 없으므로 필수.
    ④ inject(배달)는 이 **압축 뷰 위**에 한다.
검산(닫힌): 라이브 결정점 `usage.prompt_tokens` = 040 [49] 23,023 · 085 [63] 16,608 (궤적 축자) ↔ 원시
커밋 히스토리 82,885자/97,155자 — 원시 재생이면 ≈35k/40k tok 라 라이브와 다른 팔이 된다. **A_asis det
발의 재생 `prompt_tokens` 가 라이브 ±10% 밖이면 SystemExit**([[55]] — 결과 사용 금지).
    A_asis     현행 그대로(생성 뷰) — **A 가 라이브 실패를 재현해야 측정 유효**(040 write+true ·
               085 "General"). 재현 확인은 **라이브 결정점 호출 자체를 같은 채점기로 먼저** 찍는다.
    B_docfull  같은 뷰 + 마지막 도구 출력 끝에 **정의 문서 전문** 배달(헤더 = x465 `DELIVER_HEAD`
               **축자** — 헤더 델타 0·리뷰 MAJOR). 문서 집합은 **도출**이지 지목이 아니다(§도출).
               집합 **전부**·순위 0·절단만 표시.
    N_neg      같은 뷰 + 같은 채널·같은 자리에 **무내용 재촉 한 줄**([[57]] 부정통제·x465 `NUDGE` 축자)
    (선택) B_d1  1홉 집합만(=현행 `_docs_naming` 이 주는 것·도구명을 **대는** 문서뿐). `--arms` 로 켠다.
               B_d1 ≈ B_docfull 이면 정본 도출 함수 변경 불요 — 선언 범위 확장만으로 닫힌다.
⚠재생 인터페이스 = **실물 도구 스키마 + 실제 메시지 객체 + `la.generate`**(x459·x465 관용구·C584:
  렌더-텍스트로 묻는 것은 더 약한 인터페이스라 라이브를 재현 못 한다). 사본 0 — `x465.replay` 그대로
  (`usage.prompt_tokens` 반환은 이 리뷰로 **정본 x465 에** 추가했다·[[67]] 사본 금지).

## 결정점 (닫힌 술어로 특정 · 태스크 번호는 좌표이지 규칙이 아니다)
    write 형(040 t0·085 t1)   첫 **도메인 write** 호출 직전 = env 레지스트리 `mutates=True` 이고 래퍼
        (unlock/give)가 아니며 A3 `relations.by_tool[*].gate_satisfiers`(=log_verification)가 아닌 첫 호출.
        040 실물 [49] `call→file_credit_card_transaction_dispute_4829` · 085 실물 [63] `call→…_6281`.
    배달 자리 = 결정점 직전 **마지막 도구 출력 꼬리**(x465 `inject`·도구-결과 채널·역할 구조 불변).
        그 자리는 최신 배치라 압축 뷰에서도 전문 유지(=`_compact_view` 의 batch 면제)다.

## 도출 (코드 경로에 문서 id·도구명·필드명·정책 상수 리터럴 0 — 전부 선언·레지스트리·코퍼스 제목에서.
##        docstring 의 id·수치는 provenance 인용이다 — 리뷰 MINOR 반영)
    D1 = 정본 `t2_gate_patch._docs_naming(tool)`(도구명을 **대는** 문서·x465 와 같은 함수)
         ∪ A3 `policy_ontology.action_index` 에서 `tools ∋ tool` 인 문서(같은 사실의 선언판).
    D2 = **제목 참조 1홉**: 코퍼스 문서 제목의 핵(괄호 구간·선두 라벨 제거·형태만 정규화 —
         `_note_` 출처는 `title_core` docstring)이 ⓐ도구 서명 전문(unlock 출력 축자) 또는
         ⓑ D1 본문 안에 축자로 등장하는 문서.
         040 실물: 014 본문 *"determine this based on the Provisional Credit Eligibility Guidelines
         article"* → 015 · 085 실물: 6281 서명 *"based on Debit Card Provisional Credit Guidelines"*
         → 032. (`_docs_naming` 의 반대 방향 — 도구/문서가 **문서를 대는** 경우. 뜻 해석 0.)
         ★B_d1 ≠ B_docfull 로 갈리면 이 홉(`title_core`·`docs_named_in`)은 `_docs_naming` 정본에
         이식하고 여기 사본은 지운다([[67]]).
    B_docfull = D1 ∪ D2 (전부) · B_d1 = D1.
    본문 = `x448.Sandbox.cat`(shell cat·실물) → DOCDIR json → **로컬 폴백**: 로컬 결과 gz 의 KB 도구
    출력에서 복원한 코퍼스(env 가 인쇄한 축자·`--wiring-only` 전용·LLM 팔은 `--allow-traj-corpus`
    없이는 거부). ⚠샌드박스 코퍼스는 제목이 비므로 docs_dir 로 채운다 — 제목 결손이 남으면
    **SystemExit**(D2 가 조용히 ∅ 가 되는 경로 봉쇄·리뷰 MAJOR).

## 채점 (닫힌 술어만·[[59]]ⓐ — ⛔gold/`reward_info` 로 채점하지 않는다 [[23]][[69]])
  ★불리언 `eligible_for_provisional_credit` 의 **정책값 기계 도출은 열린 판단이다** — 015 의 5기준 중 4(이력
    상한)는 미독 read 에 달렸고 reason/금액 조건의 상수는 어느 선언에도 닫힌 형태로 없다(`policy_facts` 는
    인용 문자열). 지시대로 **그 항목은 채점 제외**하고 값 분포만 진단용으로 인쇄한다. 닫힌 것만 센다:
    cat            write_same(결정점 write 그대로) / write_other / read:<name>(unlock·call 포함·
                   레지스트리 `mutates=False`) / **kb_read:<name>**(env 도구 − 레지스트리 차집합 =
                   shell·KB_search_* — 이 프로브가 묻는 '선행 read'와 뷰 다이제스트 탈출구·리뷰 MAJOR)
                   / unlock_same / other_tool / no_tool / EXC(예외·복원누락 포함)
    enum_ok        서명이 `Must be one of:` 로 선언한 인자 전부가 그 enum 안(085 "General" 의 축)
    required_ok    서명 `(required)` 인자가 전부 실림(env 의 *missing N required* 와 같은 술어)
    placeholder0   `<…>` 꼴·`GP.DEFAULT_PLACEHOLDERS` 값 0
    grounded       enum 밖 문자열 인자가 근거 텍스트 어딘가에 실재(`t2_subcall.val_grounded` 정본·
                   날조만 잡는다·틀린-but-실재 값은 못 잡는다). ⚠**팔-상대 술어다**(리뷰 MINOR 명기):
                   B 팔에서만 배달 blob 이 근거에 들어간다 — 의도된 상대 실재이고 표에 그렇게 적는다.
    bool_vals      (진단) 불리언 인자 값 분포 — 판정 아님. bool 원형과 문자열 'true'/'false' 를 함께
                   잡고 **repr 로 집계**해 원형이 분포에서 사라지지 않게 한다(리뷰 MINOR).
  n = det 1발 + temp 6발 = **7/팔 (≥7)** · 태스크별 부호표([[70]]) 를 함께 낸다.
  shot 별 `secs`(재생 지연)·`prompt_tokens` 를 행에 기록한다(리뷰 MINOR — B 가 판 것의 실측).

## [[71]] 격리 서브에이전트 계약 — 4문 답
  1) 기능 하나인가 — 하나다. 각 팔의 서브(=결정점 재생성)는 **다음 발화/호출 생성** 하나만 한다. 채점은
     엔진의 닫힌 이름·멤버십·실재 대조로 바깥에서 한다.
  2) 재료가 A2/A3 선언에서 읽혀 나왔나 — 그렇다. 도구 집합·결정점 술어 = env 레지스트리(`env_surface`)·A3
     `relations`·`policy_ontology`(`action_index`); 문서 집합 = 정본 `_docs_naming` + 코퍼스 **제목** 참조.
     코드 경로에 문서 id·도구명·필드명·태스크별 떠먹이기 리터럴 0([[63]]). gold 0.
  3) 전달이 선언된 id → 정확 집기인가 — 그렇다. `x448.Sandbox.cat`(shell cat) 이 도출 id 를 정확히 집는다.
     bm25·embedding 0 — 유사도 검색은 baseline 이지 우리 방식이 아니다.
  4) 엔진이 해석·선택·순위를 하지 않는가 — 안 한다. 도출 집합 **전부** 배달(순위 0·절단만 표시), 어느 값이
     맞는지·무엇을 부를지는 끝까지 모델 몫(argmax·정답 문장 0·헤더는 어느 문서도 지목하지 않는다).

## [[62]] 4문 답
  ① 결손을 격리로 쟀나 — 정본 포렌식이 쟀다(040 73msg·085 125msg + 레버 로그 전수·[S]).
     이 프로브는 그 결손의 **경계**(미전달 ↔ 읽어도 실패)를 가른다.
  ② 격리에서 되면 레버는 전달뿐인가 — 그렇다. 단 라이브 전이에는 **알려진 간극**이 있다(리뷰 MAJOR·[[70]]):
     `_require_doc_deliver` 의 `_ctx_fits` 가 **커밋 히스토리 자수** 기준이라 두 결정점(82,885/97,155자)
     모두 배달 0자여도 이미 초과 → 라이브 레버는 skipped·0회 발화. 배선 제안 = `require_doc_before.tools`
     선언 확장 **+ `_ctx_fits` 가 압축 뷰 자수를 쓰도록 수정**(별건 수리·여기서 구현 0). 케이스별 판정은
     `--wiring-only` 가 인쇄한다.
  ③ 사라지는 모델 판단 0 — 불리언·범주를 읽고 정하는 것은 모델이 한다(엔진 도출값 없음).
  ④ 엔진 출력에 argmax·최댓값·"정답은 X" 0 — 배달 blob 은 코퍼스 축자 연결 + x465 축자 헤더뿐.

## [[70]] 병기 — 무엇을 파나
  B 가 판 것 = 문맥 +N자/결정점(배달 부피) + 재생 지연(shot 별 `secs` 실측). 태스크 2개 × 결정점 1 —
  부호표는 **태스크별 행**으로 인쇄하고(합으로 null 이어도 태스크별 부호가 갈릴 수 있다) **라이브 가드
  통과 여부(`_ctx_fits`) 열**을 병기한다. 격리 결과는 승격 근거가 아니라 경계 판정이고, 라이브 효과는
  본런 reward A/B 가 판정한다([[69]]).

## 실행 (리모트·8141·GPU 유휴 시 · cwd=tau2-bench 루트)
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    python scripts/distill/tau2/x467_policy_boolean_doc_iso.py --port 8141
    # 배선만(LLM 0·GPU 불요·로컬도 가능): ... x467_policy_boolean_doc_iso.py --wiring-only
    # 채점 경로만(LLM 0·재생 stub=라이브 결정점 호출/발화): ... x467_policy_boolean_doc_iso.py --selftest --n 0
    # 1홉 집합 팔 추가: ... --arms A_asis,B_docfull,B_d1,N_neg
"""
import argparse
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                 # noqa: E402  정본 로더·래퍼 해제(사본 금지·[[67]])
import t2_gate_patch as GP              # noqa: E402  `_docs_naming`·`_compact_view`·`_ctx_fits` 정본
import t2_subcall as SC                 # noqa: E402  `val_grounded` 정본
import x465_transfer_doc_iso as X465    # noqa: E402  문서 배달 관용구(inject·replay·to_objs·NUDGE·헤더)
import x448_index_vs_all_iso as IVA     # noqa: E402  Sandbox(shell cat)·form_norm

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)

# 결정점 좌표(provenance) — 태그:태스크:trial:종류. 규칙이 아니라 **어느 궤적을 복원하나**다(x464 D_* 동형).
# 016(diagnose)은 리뷰 BLOCK① 로 제외 — T2_ACTION_SUB 서브 격리 발화(pt 908)라 메인-문맥 재생 밖.
D_CASES = ("bank_t7336_halfB_20260821b:040:0:write,"
           "bank_t7336_halfB_20260821b:085:1:write")

# 배달 헤더 = x465 **축자**(헤더 델타 0·리뷰 MAJOR — 문구 변주는 [[57]] 인자 변화 오염).
HEAD = X465.DELIVER_HEAD
NUDGE = X465.NUDGE

_PAREN = re.compile(r"\([^)]*\)")
_ANGLE = re.compile(r"<[^<>]{1,60}>")
# 서명 형식(env unlock 출력 축자 `  - name: type (required) - desc … Must be one of: 'a', 'b'`) — 구조 파싱
_SIG_LINE = re.compile(r"^\s*-\s*(\w+):\s*(\w+)\s*\((required|optional)\)\s*-\s*(.*)$")
_QUOTED = re.compile(r"'([^']+)'")
_KB_ENTRY = re.compile(r"^\s*\d+\.\s+(.+?)\n\s+ID:\s+(doc_\S+)\n(?:\s+Score:.*\n)?\s+Content:\s?", re.M)


def form_norm(t):
    return IVA.form_norm(t)


# ── 코퍼스 (실물 → DOCDIR → 로컬 폴백) ───────────────────────────────────────────
class Corpus(object):
    """`{id: {"title", "content"}}` + 집기 채널. **읽기만** 한다([[25]]·변이 0).

    origin = "sandbox"(shell cat·실물) | "docdir"(json 파일) | "trajectory"(로컬 폴백)
    ★로컬 폴백은 **배선 검증 전용**이다: 로컬 결과 gz 의 KB 도구 출력(env 가 인쇄한 축자)에서
      `N. Title / ID: / Content:` 형식으로 복원한다 — 형식 파싱이지 도메인 해석이 아니다. LLM 팔은
      `--allow-traj-corpus` 없이는 이 코퍼스로 돌지 않는다(라이브와 같은 재료원이어야 한다).
    ★샌드박스 origin 은 제목이 전부 "" 라 docs_dir 로 채운다 — 결손이 남으면 SystemExit(리뷰 MAJOR:
      D2=∅ → B_docfull==B_d1 로 조용히 무효가 되는 경로를 fail-closed 로 봉쇄·[[55]]).
    """

    def __init__(self, tags, docs_dir):
        self.docs, self.sb, self.origin = {}, None, None
        try:
            self.sb = IVA.Sandbox()
            import t2_search as TS
            cor = TS.corpus_from_env(self.sb.env) or {}
            for k, v in cor.items():
                self.docs[str(k)] = {"title": "", "content": str(v or "")}
            if self.docs:
                self.origin = "sandbox"
        except (Exception, SystemExit) as e:          # x448.Sandbox 는 지형 불일치에 SystemExit 을 낸다
            print("[코퍼스] 샌드박스 없음(%s) — DOCDIR/로컬 폴백" % (repr(e)[:90],))
        if docs_dir and os.path.isdir(docs_dir):
            n = 0
            for f in os.listdir(docs_dir):
                if not f.endswith(".json"):
                    continue
                try:
                    d = json.load(io.open(os.path.join(docs_dir, f), encoding="utf-8"))
                except Exception:
                    continue
                did = str(d.get("id") or f[:-5])
                rec = self.docs.setdefault(did, {"title": "", "content": ""})
                rec["title"] = rec["title"] or str(d.get("title") or "")
                rec["content"] = rec["content"] or str(d.get("content") or "")
                n += 1
            if n and self.origin is None:
                self.origin = "docdir"
        if not self.docs:
            for tag in tags:
                try:
                    ss = F.sims(tag, ".results.json.gz")
                except Exception:
                    continue
                for s in ss:
                    for m in (s.get("messages") or []):
                        if str(m.get("role") or "") != "tool":
                            continue
                        c = str(m.get("content") or "")
                        if "ID: doc_" not in c:
                            continue
                        ents = list(_KB_ENTRY.finditer(c))
                        for i, mt in enumerate(ents):
                            end = ents[i + 1].start() if i + 1 < len(ents) else len(c)
                            body = c[mt.end():end]
                            body = body.split(NLC + "[Timing:")[0].rstrip()
                            rec = self.docs.setdefault(mt.group(2), {"title": "", "content": ""})
                            rec["title"] = rec["title"] or mt.group(1).strip()
                            if len(body) > len(rec["content"]):
                                rec["content"] = body
            if self.docs:
                self.origin = "trajectory"
        if not self.docs:
            raise SystemExit("코퍼스 0편(샌드박스·DOCDIR·로컬 폴백 전부 실패) — 경로부터 본다([[55]])")
        if self.origin == "sandbox":
            untitled = sum(1 for v in self.docs.values() if not v["title"])
            if untitled:
                raise SystemExit(
                    "샌드박스 코퍼스 %d편 중 %d편 제목 없음 — docs_dir(%s) 어긋남. D2 도출이 조용히 ∅ 가"
                    " 되므로 fail-closed(리뷰 MAJOR·[[55]]) — 경로부터 맞춘다"
                    % (len(self.docs), untitled, docs_dir))

    def content_map(self):
        return {k: v["content"] for k, v in self.docs.items()}

    def block(self, did):
        """문서 한 편의 배달 블록 — `### id — title` 헤더 한 줄(C585 `_docs_delivery` 규약) + 본문."""
        txt, src = None, "corpus"
        if self.sb is not None:
            try:
                got, miss = self.sb.cat([did])
                if not miss and got.strip():
                    txt, src = got, "shell"
            except Exception:
                txt = None
        if txt is None:
            rec = self.docs.get(did)
            if not rec or not rec["content"]:
                return None, "missing"
            txt = rec["content"]
        title = (self.docs.get(did) or {}).get("title") or ""
        return "### %s — %s%s%s" % (did, title, NLC, txt), src


# ── 선언 읽기 ───────────────────────────────────────────────────────────────────
def load_registry():
    p = os.path.join(HERE, "a2", "env_surface.json")
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)["banking_knowledge"]["tools"]


def gate_satisfiers(a3):
    out = set()
    for _t, spec in ((a3.get("relations") or {}).get("by_tool") or {}).items():
        out |= set(spec.get("gate_satisfiers") or ())
    return out


# ── 궤적 복원·결정점 ───────────────────────────────────────────────────────────
def load_sim(tag, task_suffix, trial):
    sims = [s for s in F.sims(tag, ".results.json.gz")
            if str(s.get("task_id")).split("_")[-1] == str(task_suffix)]
    sims.sort(key=lambda s: s.get("trial") or 0)
    if trial is not None:
        sims = [s for s in sims if s.get("trial") == trial]
    if not sims:
        raise SystemExit("task %s trial %s 의 sim 이 없다 (%s)" % (task_suffix, trial, tag))
    return sims[0]


def eff_tool(tc):
    """호출 → (래퍼명, 실효 도구명). 래퍼면 안쪽 대상(정본 `inner_name`)."""
    nm = F.nameof(tc)
    ag = F.argsof(tc)
    return nm, (F.inner_name(ag) if nm in F.WRAPPERS else nm)


def dp_write(msgs, registry, satisfiers):
    """첫 **도메인 write** 호출 — 닫힌 술어(역할·래퍼 제외·레지스트리 mutates·gate satisfier 제외)."""
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm, eff = eff_tool(tc)
            if nm in (F.UNLOCK, F.GIVE, F.CALLU):
                continue
            spec = registry.get(eff)
            if not spec or not spec.get("mutates") or eff in satisfiers:
                continue
            return i, eff, tc
    raise SystemExit("도메인 write 호출이 궤적에 없다 — 궤적이 예상과 다르다")


def _usage_pt(m):
    """메시지의 `usage.prompt_tokens`(궤적 축자) — 정보-맞춤 검산의 라이브 기준(리뷰 BLOCK①②)."""
    u = m.get("usage") if isinstance(m, dict) else getattr(m, "usage", None)
    if isinstance(u, dict):
        return u.get("prompt_tokens")
    return getattr(u, "prompt_tokens", None)


def ctx_text(ctx, roles=("user", "tool")):
    return NLC.join(str(m.get("content") or "") for m in ctx if str(m.get("role") or "") in roles)


def schema_text(ctx, tool, registry):
    """도구 서명 전문 — 문맥의 unlock 출력 축자(모델이 본 그것) → 없으면 레지스트리 desc."""
    key = "Tool unlocked: %s" % tool
    for m in ctx:
        if str(m.get("role") or "") == "tool" and key in str(m.get("content") or ""):
            return str(m.get("content") or ""), "unlock-output"
    return str((registry.get(tool) or {}).get("desc") or ""), "registry-desc"


def schema_enums(text):
    """서명의 `Must be one of:` 선언 → {arg: [값]} · `(required)` 인자 목록. 형식 파싱(뜻 해석 0)."""
    enums, required = {}, []
    for line in str(text or "").splitlines():
        mt = _SIG_LINE.match(line)
        if not mt:
            continue
        arg, _typ, req, desc = mt.groups()
        if req == "required":
            required.append(arg)
        if "Must be one of:" in desc:
            vals = _QUOTED.findall(desc.split("Must be one of:", 1)[1])
            if vals:
                enums[arg] = vals
    return enums, required


# ── 문서 도출 ───────────────────────────────────────────────────────────────────
def title_core(t):
    """제목의 핵 — 괄호 구간 제거 · 선두 라벨(≤2 단어 + ':') 제거 · 형태만 정규화. 뜻 해석 0.

    _note_ 출처(리뷰 MINOR·[[23]] 동형 규율): 코퍼스 제목 **형식 실측**이다(2026-08-22 로컬 폴백
    전수 282편 실측: 선두 라벨 88편·핵<3토큰 5편 — 정확 수치는 매 실행 `--wiring-only` 의
    `[도출 provenance]` 인쇄가 정본). 상품명 라벨("Silver Account:" 류)도 같은 형식이라 함께 벗겨지고,
    핵<3토큰 문서는 D2 참조가 영원히 불가하다. 도메인 뜻 해석이 아니라 제목 표기 형식 규칙이다.
    """
    t = _PAREN.sub(" ", str(t or ""))
    if ":" in t:
        head, tail = t.split(":", 1)
        if len(head.split()) <= 2:
            t = tail
    return form_norm(t)


def docs_named_in(texts, corpus, min_tokens=3):
    """제목 핵이 주어진 텍스트(서명·D1 본문) 안에 축자로 등장하는 문서 id — 제목 참조 1홉."""
    flat = [form_norm(x) for x in texts if x]
    out = set()
    for did, rec in corpus.docs.items():
        core = title_core(rec.get("title") or "")
        if len(core.split()) < min_tokens:
            continue
        if any(core in f for f in flat):
            out.add(did)
    return out


def title_provenance(corpus):
    """title_core 전수 통계(라벨 제거 편수·핵<3토큰 제외 편수) — `--wiring-only` provenance(리뷰 MINOR)."""
    lab = exc = 0
    for v in corpus.docs.values():
        t = _PAREN.sub(" ", str(v.get("title") or ""))
        if ":" in t and len(t.split(":", 1)[0].split()) <= 2:
            lab += 1
        if len(title_core(v.get("title") or "").split()) < 3:
            exc += 1
    return lab, exc


def derive_write_docs(tool, a3, corpus, sig_text, docs_dir):
    if not any(v.get("title") for v in corpus.docs.values()):
        raise SystemExit("D2 도출 불가: 코퍼스 제목 0편 — docs_dir/DOCDIR 어긋남. 조용한 ∅ 금지"
                         "(리뷰 MAJOR·[[55]]) — 경로부터 본다")
    d1 = set(x for x in (GP._docs_naming(tool, docs_dir, corpus=corpus.content_map()) or ()) if x)
    for e in ((a3.get("policy_ontology") or {}).get("action_index") or []):
        if tool in (e.get("tools") or ()) and e.get("doc"):
            d1.add(str(e["doc"]))
    d1_texts = [(corpus.docs.get(d) or {}).get("content") or "" for d in sorted(d1)]
    d2 = docs_named_in([sig_text] + d1_texts, corpus) - d1
    return sorted(d1), sorted(d2)


def deliver(ids, corpus, maxchars):
    blocks, src, missing = [], {}, []
    for did in ids:
        b, s = corpus.block(did)
        if b is None:
            missing.append(did)
            continue
        blocks.append(b)
        src[did] = s
    blob = (NLC + NLC).join(blocks)
    cut = len(blob) > maxchars
    return blob[:maxchars], cut, src, missing


# ── 라이브 생성 뷰 재구성 (리뷰 BLOCK②) ────────────────────────────────────────
def _view_objs(ctx):
    """dict 궤적 → 뷰 계산용 객체. tau2 있으면 **정본 변환**(`X465.to_objs` = replay 와 같은 변환),
    없으면 속성 shim(로컬 `--wiring-only`/`--selftest` 전용 — LLM 팔은 tau2 경로를 강제한다)."""
    try:
        objs, dropped = X465.to_objs(ctx)
        return objs, dropped, "tau2"
    except Exception as e:
        class _S(object):
            def __init__(self, d):
                self._d = d
                self.role = str(d.get("role") or "")
                self.content = d.get("content")
                self.error = bool(d.get("error"))
                self.id = d.get("id")
        return [_S(m) for m in ctx], 0, "shim(%s)" % type(e).__name__


def _obj_dict(o):
    if isinstance(o, dict):
        return dict(o)
    if hasattr(o, "_d"):
        d = dict(o._d)
        d["content"] = o.content
        return d
    return o.model_dump()


def _system_msg(policy):
    """라이브 system = tau2 `LLMAgent` 조립식(`SYSTEM_PROMPT.format`)으로 `sim['policy']` 에서 생성 —
    텍스트 사본 0(리뷰 BLOCK②). tau2 없으면 None(로컬 wiring 전용·LLM 팔은 거부)."""
    try:
        from tau2.agent.llm_agent import SYSTEM_PROMPT, AGENT_INSTRUCTION
    except Exception:
        return None
    return {"role": "system",
            "content": SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION)}


def compact_env(objs):
    """`GP._compact_view` 를 **라이브 call-site 와 같은 env·기본값**으로 건다 — 인자가 사는 단 한 자리.

    t7336 은 go_stack.sh 를 source 하므로 T2_VIEW_COMPACT=1·KEEP 6·MINLEN 800·MINTOTAL 60000·
    MSG_CAP 8000 이 라이브 값이다(런 스크립트 도출·docstring §팔). 다른 프로브(x470)가 이 함수를
    **import** 한다 — 인자 사본이 두 곳에서 조용히 갈라지지 않게([[67]] 사본 금지).
    반환 = (뷰 객체 리스트, 다이제스트된 메시지 id 집합).
    """
    return GP._compact_view(
        objs,
        keep_recent=int(os.environ.get("T2_VIEW_COMPACT_KEEP", "6")),
        min_len=int(os.environ.get("T2_VIEW_COMPACT_MINLEN", "800")),
        min_total=int(os.environ.get("T2_VIEW_COMPACT_MINTOTAL", "60000")),
        msg_cap=int(os.environ.get("T2_VIEW_MSG_CAP", "8000")))


def compact_view_dicts(ctx):
    """궤적 dict 리스트 → **라이브 생성 뷰** dict 리스트. (변환 → `compact_env` → dict 환원)

    tau2 가 있으면 정본 변환(`X465.to_objs`), 없으면 속성 shim(로컬 `--wiring-only` 전용).
    반환 = (뷰 dict 리스트, 변환 누락 수, 다이제스트 수, 변환 경로). x470 이 이 함수를 쓴다([[67]]).
    """
    objs, dropped, conv = _view_objs(ctx)
    view, digested = compact_env(objs)
    return [_obj_dict(o) for o in view], dropped, len(digested), conv


def assemble_views(ctx, policy):
    """라이브 **생성 뷰** 재구성 — 원시 커밋 히스토리는 라이브 뷰가 아니다(리뷰 BLOCK②).

    순서 = 객체 변환(정본 `X465.to_objs`) → `compact_env`(라이브 call-site 와 같은 env·기본값 —
    t7336 은 go_stack.sh 로 T2_VIEW_COMPACT=1·MINTOTAL 60000·MSG_CAP 8000) → system 선두 삽입.
    inject(배달)는 이 뷰 위에 한다.
    """
    objs, dropped, conv = _view_objs(ctx)
    if dropped:
        raise SystemExit("메시지 객체 변환 누락 %d — A 문맥이 라이브와 다르다([[55]]·리뷰 MINOR)" % dropped)
    view, digested = compact_env(objs)
    vd = [_obj_dict(o) for o in view]
    sysm = _system_msg(policy)
    base = ([sysm] if sysm else []) + vd
    return {"base": base, "objs_raw": objs, "digested": len(digested), "conv": conv,
            "has_system": sysm is not None, "sys_chars": len((sysm or {}).get("content") or ""),
            "raw_chars": sum(len(str(m.get("content") or "")) for m in ctx),
            "view_chars": sum(len(str(m.get("content") or "")) for m in vd)}


# ── 채점 ────────────────────────────────────────────────────────────────────────
def classify(calls, dp_tool, registry, kb_names=()):
    if not calls:
        return "no_tool", None, None
    nm, ag = calls[0]
    tc = {"name": nm, "arguments": ag}
    wrap, eff = eff_tool(tc)
    spec = registry.get(eff) or {}
    if wrap == F.UNLOCK:
        if eff == dp_tool:
            return "unlock_same", eff, None
        if spec and not spec.get("mutates"):
            return "read:" + eff, eff, None
        return "other_tool", eff, None
    if eff in kb_names:
        # 레지스트리 밖 env 도구(shell·KB_search_*) — '선행 read'와 뷰 다이제스트 탈출구(리뷰 MAJOR).
        return "kb_read:" + eff, eff, None
    if eff == dp_tool and dp_tool:
        return "write_same", eff, F.flat_args(ag)
    if spec.get("mutates"):
        return "write_other", eff, F.flat_args(ag)
    if spec:
        return "read:" + eff, eff, None
    return "other_tool", eff, None


def placeholder(v):
    s = str(v)
    return bool(_ANGLE.search(s)) or s in GP.DEFAULT_PLACEHOLDERS


def score_args(args, enums, required, ground_texts):
    """write 인자의 닫힌 검사 — enum 멤버십·필수 실림·자리표시자·실재. 값의 옳고 그름은 판정하지 않는다.

    grounded 는 **팔-상대 술어**다(B 팔에서만 배달 blob 이 근거에 들어간다) — summarize 헤더에 명기.
    bool_vals 는 bool 원형 + 문자열 'true'/'false' 를 함께 잡는다(원형은 repr 집계로 보존·리뷰 MINOR).
    """
    args = args or {}
    bad_enum = [k for k, vals in enums.items() if k in args and str(args[k]) not in vals]
    missing = [k for k in required if k not in args]
    ph = [k for k, v in args.items() if isinstance(v, str) and placeholder(v)]
    ungrounded = []
    for k, v in args.items():
        if k in enums or not isinstance(v, str) or len(v.strip()) < 3:
            continue
        if not SC.val_grounded(v, ground_texts):
            ungrounded.append(k)
    bools = {k: v for k, v in args.items()
             if isinstance(v, bool) or (isinstance(v, str) and v.strip().lower() in ("true", "false"))}
    return {"enum_ok": not bad_enum, "bad_enum": bad_enum, "required_ok": not missing,
            "missing_required": missing, "placeholder0": not ph, "placeholders": ph,
            "grounded": not ungrounded, "ungrounded": ungrounded, "bool_vals": bools}


# ── 케이스 조립 ────────────────────────────────────────────────────────────────
def build_case(spec, a3, registry, corpus, docs_dir, maxchars, kb_names):
    tag, task, trial, kind = spec.split(":")
    if kind != "write":
        raise SystemExit(
            "kind=%r 는 이 프로브 밖이다(리뷰 BLOCK①): diagnose 형(016류) 결정점은 T2_ACTION_SUB **서브"
            " 격리** 발화(라이브 usage.prompt_tokens 908 ↔ 메인 턴 10k+)라 메인-문맥 재생으로 정보-맞춤이"
            " 성립하지 않는다 — x464(SEARCH_AGENT 재무장) 계열로 보낸다" % kind)
    sim = load_sim(tag, task, int(trial))
    msgs = sim.get("messages") or []
    case = {"spec": spec, "task": sim.get("task_id"), "trial": sim.get("trial"), "kind": kind,
            "n_msgs": len(msgs)}
    print("=" * 100)
    i_dp, tool, live_tc = dp_write(msgs, registry, gate_satisfiers(a3))
    ctx = msgs[:i_dp]
    live_pt = _usage_pt(msgs[i_dp])
    sig, sig_src = schema_text(ctx, tool, registry)
    enums, required = schema_enums(sig)
    d1, d2 = derive_write_docs(tool, a3, corpus, sig, docs_dir)
    ids = sorted(set(d1) | set(d2))
    case.update({"i_dp": i_dp, "tool": tool, "sig_src": sig_src, "enums": enums,
                 "required": required, "d1": d1, "d2": d2, "ids": ids, "live_pt": live_pt,
                 "live_call": [F.nameof(live_tc), F.flat_args(F.argsof(live_tc))]})
    case["live_tc"] = live_tc
    print("x467 · %s t%s · %s · msgs=%d · 결정점=[%d] write=%s (서명 출처 %s)"
          % (case["task"], case["trial"], kind, len(msgs), i_dp, tool, sig_src))
    print("  결정점 라이브 usage.prompt_tokens = %s" % (live_pt,))          # 리뷰 BLOCK① 지시(LLM 0)
    print("  서명 enum 인자 %d종 %s · required %d종" % (len(enums), sorted(enums), len(required)))
    print("  D1(도구명을 대는 문서·_docs_naming+action_index) %d편: %s" % (len(d1), ", ".join(d1)))
    print("  D2(제목 참조 1홉: 서명/D1 본문이 제목을 댄 문서) %d편: %s" % (len(d2), ", ".join(d2)))
    i_tool = X465.last_tool_idx(ctx)
    if i_tool is None:
        raise SystemExit("배달 자리(마지막 도구 출력)가 없다 — 이 결정점에는 이 프로브를 못 쓴다")
    blob, cut, src, missing = deliver(ids, corpus, maxchars)
    d1_blob, d1_cut, _s1, _m1 = deliver(d1, corpus, maxchars)
    case.update({"i_tool": i_tool, "chars": len(blob), "truncated": cut, "missing": missing,
                 "fetch_src": src, "d1_chars": len(d1_blob)})
    print("  배달 자리=[%d] · 문서 %d편 · 전문 %d자%s · 집기=%s%s"
          % (i_tool, len(ids), len(blob), " ⚠절단" if cut else "",
             "/".join(sorted(set(src.values()))) or "-",
             ("  ⚠집기 실패 %r" % (missing,)) if missing else ""))
    for did in ids:
        print("       %s  [%s] %s" % (did, src.get(did, "missing"), (corpus.docs.get(did) or {}).get("title", "")[:70]))
    # ── 라이브 생성 뷰 재구성 + 팔 조립 (리뷰 BLOCK② — inject 는 압축 뷰 위) ──
    v = assemble_views(ctx, str(sim.get("policy") or ""))
    hd = HEAD % tool
    base_view = v["base"]
    case["arm_ctx"] = {"A_asis": base_view,
                       "B_docfull": X465.inject(base_view, hd + NLC + NLC + blob),
                       "B_d1": X465.inject(base_view, hd + NLC + NLC + d1_blob),
                       "N_neg": X465.inject(base_view, NUDGE)}
    case["ctx"] = ctx
    case["blob"], case["d1_blob"], case["sig"] = blob, d1_blob, sig
    case.update({"raw_chars": v["raw_chars"], "view_chars": v["view_chars"],
                 "view_digested": v["digested"], "conv": v["conv"],
                 "has_system": v["has_system"], "sys_chars": v["sys_chars"]})
    print("  생성 뷰 재구성: 원시 %d자 → 압축 뷰 %d자(다이제스트 %d건·변환=%s) + system %d자%s"
          % (v["raw_chars"], v["view_chars"], v["digested"], v["conv"], v["sys_chars"],
             "" if v["has_system"] else " ⚠system 조립 불가(tau2 없음 — 로컬 wiring 전용·LLM 팔은 거부)"))
    # ── 라이브 가드 판정([[70]] 병기·리뷰 MAJOR — 별건 수리 대상·여기서 구현 0) ──
    fits, hist = GP._ctx_fits(v["objs_raw"], hd + NLC + NLC + blob)
    case["live_guard_pass"], case["hist_chars"] = bool(fits), int(hist)
    print("  라이브 가드 GP._ctx_fits(커밋 히스토리 %d자 + 배달 %d자) = %s → 본런 T2_REQUIRE_DOC_DELIVER 는 %s"
          % (int(hist), len(hd) + 2 + len(blob), "PASS" if fits else "FAIL",
             "발화 가능" if fits else "skipped(0회 발화) — 전이엔 `_ctx_fits` 압축-뷰-자수 수정이 별건으로 필요"))
    # ── 라이브 재현 채점: 결정점의 실제 호출을 같은 채점기로 — A 가 재현해야 할 실패의 서명 ──
    live_calls = [(F.nameof(live_tc), F.argsof(live_tc))]
    cat, eff, args = classify(live_calls, tool, registry, kb_names)
    sc = score_args(args, enums, required, [ctx_text(ctx), sig])
    case["live_score"] = dict(sc, cat=cat)
    print("  [라이브 채점] cat=%s enum_ok=%s(%s) required_ok=%s(missing %d) placeholder0=%s grounded=%s(%s) bool=%s"
          % (cat, sc["enum_ok"], ",".join(sc["bad_enum"]) or "-", sc["required_ok"],
             len(sc["missing_required"]), sc["placeholder0"], sc["grounded"],
             ",".join(sc["ungrounded"]) or "-", sc["bool_vals"]))
    print("  헤더(x465 축자·델타 0): %s" % hd[:160])
    return case


def _replay4(cx, tools, model, base, t):
    """`X465.replay` 호출 — 반환은 정본 `ReplayResult`(이름 있는 필드). 여기서는 이 파일의 옛 4-튜플
    호출부에 맞춰 그대로 넘긴다(namedtuple 이라 호환·언팩 개수는 더 이상 여기서 정하지 않는다)."""
    r = X465.replay(cx, tools, model, base, t)
    return (r.calls, r.text, r.dropped, r.prompt_tokens)


def run_case(case, arms, tools, registry, kb_names, a, base):
    rows = []
    a_chars = sum(len(str(m.get("content") or "")) for m in case["arm_ctx"]["A_asis"])
    for arm in arms:
        cx = case["arm_ctx"][arm]
        delta = sum(len(str(m.get("content") or "")) for m in cx) - a_chars
        print(NLC + "── %s t%s · %s (배달 델타 +%d자) ────────────────────────────" % (
            case["task"], case["trial"], arm, delta))
        ground = [ctx_text(case["ctx"]), case.get("sig") or ""]
        if arm.startswith("B_"):
            ground.append(case["blob"] if arm == "B_docfull" else case["d1_blob"])
        for k, t in enumerate([0.0] + [a.temperature] * a.n):
            t0 = time.time()
            try:
                calls, text, dropped, pt = _replay4(cx, tools, a.model, base, t)
            except Exception as e:
                print("  #%d t=%.1f EXC %r" % (k, t, e))
                rows.append({"case": case["spec"], "arm": arm, "k": k, "temp": t, "cat": "EXC",
                             "err": repr(e)[:200], "secs": round(time.time() - t0, 2)})
                continue
            secs = round(time.time() - t0, 2)
            if dropped:
                # 구성 실패는 조용히 버리지 않는다(리뷰 MINOR): 그 shot 은 EXC, A det 면 격리 무효.
                if arm == "A_asis" and k == 0:
                    raise SystemExit("A_asis det 재생에서 메시지 복원 누락 %d — 격리 무효([[55]])" % dropped)
                print("  #%d t=%.1f EXC(dropped_msgs=%d)" % (k, t, dropped))
                rows.append({"case": case["spec"], "arm": arm, "k": k, "temp": t, "cat": "EXC",
                             "dropped_msgs": dropped, "secs": secs})
                continue
            if arm == "A_asis" and k == 0:
                live_pt = case.get("live_pt")
                if pt is None:
                    raise SystemExit("재생이 usage.prompt_tokens 를 안 돌려준다 — x465.replay 정본 갱신"
                                     " 필요(리뷰 BLOCK②·[[67]] 사본 금지)")
                if not live_pt:
                    raise SystemExit("라이브 usage.prompt_tokens 없음 — 정보-맞춤 검산 불가([[55]])")
                if abs(pt - live_pt) > 0.10 * live_pt:
                    raise SystemExit("A_asis det prompt_tokens %s ↔ 라이브 %s — ±10%% 밖. 뷰 재구성이"
                                     " 라이브와 다르다([[55]] — 결과 사용 금지)" % (pt, live_pt))
            row = {"case": case["spec"], "task": case["task"], "arm": arm, "k": k, "temp": t,
                   "delta_chars": delta, "dropped_msgs": dropped, "secs": secs,
                   "prompt_tokens": pt, "live_pt": case.get("live_pt"),
                   "calls": [[nm, json.dumps(ag, ensure_ascii=False, default=str)[:300]] for nm, ag in calls],
                   "text": " ".join(text.split())[:400]}
            cat, eff, args = classify(calls, case["tool"], registry, kb_names)
            row["cat"] = cat
            if cat in ("write_same", "write_other"):
                row.update(score_args(args, case["enums"], case["required"], ground))
            print("  #%d t=%.1f  %-28s pt=%s/%s %.1fs  %s" % (
                k, t, cat, case.get("live_pt"), pt, secs,
                ("enum_ok=%s req_ok=%s ph0=%s grounded=%s bool=%s" % (
                    row["enum_ok"], row["required_ok"], row["placeholder0"], row["grounded"],
                    row["bool_vals"])) if "enum_ok" in row else
                ("calls=%s" % (",".join(nm for nm, _x in calls) or "-"))))
            rows.append(row)
    return rows


def summarize(cases, rows, arms):
    print(NLC + "=" * 100)
    for case in cases:
        rs_all = [r for r in rows if r["case"] == case["spec"]]
        if not rs_all:
            continue
        print("[%s t%s · write · +%d자 · live_guard=%s]"
              % (case["task"], case["trial"], case["chars"],
                 "PASS" if case.get("live_guard_pass") else "FAIL(본런 레버 skipped·[[70]])"))
        cats = sorted({r["cat"] for r in rs_all})
        print("  %-10s %s  | enum_ok req_ok ph0 grounded(팔-상대: B는 배달물 포함) (write 만)"
              " | bool 분포(진단·repr=원형 보존)" % ("팔", " ".join("%-26s" % c for c in cats)))
        for arm in arms:
            rs = [r for r in rs_all if r["arm"] == arm]
            if not rs:
                continue
            w = [r for r in rs if "enum_ok" in r]
            bools = {}
            for r in w:
                for k2, v in (r.get("bool_vals") or {}).items():
                    bools.setdefault(k2, {}).setdefault(repr(v), 0)
                    bools[k2][repr(v)] += 1
            print("  %-10s %s  | %d/%d %d/%d %d/%d %d/%d | %s" % (
                arm, " ".join("%-26s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == c), len(rs)))
                              for c in cats),
                sum(1 for r in w if r["enum_ok"]), len(w), sum(1 for r in w if r["required_ok"]), len(w),
                sum(1 for r in w if r["placeholder0"]), len(w), sum(1 for r in w if r["grounded"]), len(w),
                json.dumps(bools, ensure_ascii=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--cases", default=D_CASES, help="태그:태스크:trial:write 쉼표 열거(write 만 —"
                                                     " diagnose 는 리뷰 BLOCK① 로 이 프로브 밖)")
    ap.add_argument("--n", type=int, default=6, help="temp 표본 수 — det 1발이 늘 앞선다(합 n+1≥7)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--maxchars", type=int, default=90000, help="배달 상한(x465 동일·절단은 표시)")
    ap.add_argument("--arms", default="A_asis,B_docfull,N_neg", help="+B_d1 로 1홉 집합 팔을 켠다")
    ap.add_argument("--docs-dir", dest="docs_dir", default=None,
                    help="코퍼스 json 디렉터리(기본 T2_KB_DOCS_DIR → x430.DOCDIR)")
    ap.add_argument("--corpus-tags", dest="corpus_tags",
                    default="bank_t7336_halfA_20260821b,bank_t7336_halfB_20260821b,bank_t7336_smoke_20260821b",
                    help="로컬 폴백 코퍼스를 복원할 결과 태그(배선 검증 전용)")
    ap.add_argument("--allow-traj-corpus", action="store_true",
                    help="로컬 폴백 코퍼스로 LLM 팔을 돌리는 것을 허용(기본 거부 — 라이브 재료원이 아니다)")
    ap.add_argument("--wiring-only", action="store_true", help="배선 검증만(LLM 0·GPU 불요·[[55]])")
    ap.add_argument("--selftest", action="store_true",
                    help="재생을 stub(라이브 결정점 호출/발화 그대로)으로 바꿔 채점·집계·JSON 경로만 검증(LLM 0)")
    ap.add_argument("--out", default="x467_policy_boolean_doc_iso.json")
    a = ap.parse_args()

    # ── ① 선언·레지스트리·코퍼스 (읽기 전용) ─────────────────────────────────────
    a3 = X465.load_a3()
    registry = load_registry()
    docs_dir = a.docs_dir or os.environ.get("T2_KB_DOCS_DIR") or __import__("x430_account_facts").DOCDIR
    specs = [s.strip() for s in a.cases.split(",") if s.strip()]
    corpus = Corpus(sorted({s.split(":")[0] for s in specs}
                           | {t.strip() for t in a.corpus_tags.split(",") if t.strip()}), docs_dir)
    print("[코퍼스] origin=%s · %d편 · 제목 있음 %d편 · docs_dir=%s(%s)"
          % (corpus.origin, len(corpus.docs), sum(1 for v in corpus.docs.values() if v["title"]),
             docs_dir, "있음" if os.path.isdir(docs_dir or "") else "없음"))
    lab, exc = title_provenance(corpus)
    print("[도출 provenance] title_core(D2 홉): 전수 %d편 · 선두 라벨 제거 %d편 · 핵<3토큰(D2 참조 불가) %d편"
          % (len(corpus.docs), lab, exc))
    if not (a3.get("require_doc_before") or {}).get("tools"):
        raise SystemExit("A3 require_doc_before.tools 선언이 비어 있다 — 층 정합부터([[24]])")
    print("[선언] require_doc_before.tools(현행·이관만) = %s" % ", ".join(a3["require_doc_before"]["tools"]))
    # kb_read 범주 = env 도구 − 레지스트리 차집합(리터럴 0·리뷰 MAJOR). 로컬(샌드박스 없음)은 빈 집합.
    kb_names = set()
    if corpus.sb is not None:
        kb_names = {t.name for t in (corpus.sb.env.get_tools() or [])} - set(registry)
    print("[채점] kb_read 범주(env 도구 − 레지스트리) = %s"
          % (", ".join(sorted(kb_names)) or "(샌드박스 없음 — 리모트에서 채워진다)"))

    # ── ② 케이스 조립 — 궤적 복원·결정점·도출·뷰 재구성·배달·라이브 채점 (LLM 0) ──
    cases = [build_case(s, a3, registry, corpus, docs_dir, a.maxchars, kb_names) for s in specs]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    print(NLC + "[배선] 케이스 %d · 팔 %s · 결정점 %s" % (
        len(cases), ",".join(arms), ", ".join("%s t%s=[%d]" % (c["task"], c["trial"], c["i_dp"]) for c in cases)))
    for c in cases:
        print("       %s t%s: 배달 %d편 %d자 (B_d1 %d자) · 라이브 cat=%s · live_pt=%s · 뷰 %d자"
              " · live_guard=%s" % (
                  c["task"], c["trial"], len(c["ids"]), c["chars"], c["d1_chars"],
                  c["live_score"].get("cat"), c["live_pt"], c["view_chars"] + c["sys_chars"],
                  "PASS" if c["live_guard_pass"] else "FAIL"))
    if a.wiring_only:
        print("[배선] wiring-only 종료 — LLM 0·GPU 0")
        return 0
    if a.selftest:
        # 채점 경로 검증(LLM 0): 모든 팔이 라이브 결정점의 실제 호출을 그대로 "생성"한다고 치고 돈다.
        # 기대 = 모든 팔이 [라이브 채점] 과 같은 서명(A 재현 검정의 계기 검증이지 결과가 아니다).
        def _stub(msgs, tools, model, base, temperature):
            c = next(cc for cc in cases if any(msgs is v for v in cc["arm_ctx"].values()))
            tc = c["live_tc"]
            return [(F.nameof(tc), F.argsof(tc))], "", 0, c.get("live_pt")
        X465.replay = _stub
        tools, base = [], "stub"
        print("  [selftest] 재생 stub — 라이브 결정점 호출/발화를 그대로 돌려준다(LLM 0)")
        rows = []
        for c in cases:
            rows += run_case(c, arms, tools, registry, kb_names, a, base)
        summarize(cases, rows, arms)
        print("[selftest] 끝 — 모든 팔이 [라이브 채점] 서명과 같아야 한다(채점 경로 검증·결과 아님·파일 저장 0)")
        return 0
    if corpus.origin == "trajectory" and not a.allow_traj_corpus:
        raise SystemExit("LLM 팔은 로컬 폴백 코퍼스로 돌리지 않는다(라이브 재료원 아님) — 샌드박스/DOCDIR 을 주거나 --allow-traj-corpus")

    # ── ③ 재생 — 실물 도구 + la.generate · det 1발 + temp n발 / 팔 ───────────────
    if corpus.sb is None:
        raise SystemExit("샌드박스(실물 도구 스키마)가 없어 재생을 못 한다 — PYTHONPATH=src 로 tau2 를 준다")
    bad = ["%s(conv=%s,sys=%s)" % (c["task"], c["conv"], c["has_system"])
           for c in cases if c["conv"] != "tau2" or not c["has_system"]]
    if bad:
        raise SystemExit("생성 뷰 조립이 tau2 정본 경로가 아니다: %s — x465.to_objs·tau2 system 조립이"
                         " 있어야 LLM 팔이 유효하다(리뷰 BLOCK②)" % ", ".join(bad))
    tools = list(corpus.sb.env.get_tools() or [])
    base = "http://localhost:%d/v1" % a.port
    print("  재생 인터페이스: 실물 도구 스키마 %d종 + 실제 메시지 객체 + la.generate (C584)" % len(tools))
    rows = []
    for c in cases:
        rows += run_case(c, arms, tools, registry, kb_names, a, base)

    # ── ④ 집계 — 케이스 × 팔 × 닫힌 분류 + [[70]] 부호표 ─────────────────────────
    summarize(cases, rows, arms)
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"cases": [{k: v for k, v in c.items()
                              if k not in ("arm_ctx", "ctx", "blob", "d1_blob", "sig", "live_tc")}
                             for c in cases], "rows": rows}, f, ensure_ascii=False, indent=1)
    print(NLC + "판정: 각 케이스에서 A(생성 뷰 재구성)가 라이브 채점 서명 + prompt_tokens ±10% 를 재현해야"
          " 격리가 산다(아니면 [[55]] — 결과 사용 금지).")
    print("      B 에서 kb_read:*/read:*(선행 read) 또는 enum_ok/bool 분포가 바뀌고 N 이 A 와 같으면"
          " **원인은 미전달**. B 도 A 와 같으면 경계([[62]]②).")
    print("      B_d1 ≈ B_docfull 이면 `_docs_naming` 변경 불요 — 갈리면 D2 홉을 정본에 이식([[67]]).")
    print("[[70]] 병기: 태스크별 부호표(위 표·live_guard 열 포함)·B 가 판 것 = 문맥 +%s자 + 재생 지연"
          "(rows.secs 실측)·성적 확정은 본런 reward 재실행에서만([[69]])."
          % "/".join(str(c["chars"]) for c in cases))
    for c in cases:
        if not c.get("live_guard_pass"):
            print("      ⚠%s t%s: 라이브 가드 `_ctx_fits` 초과(hist %d자) — 격리 B 가 살아도 본런 레버는"
                  " skipped·0회 발화." % (c["task"], c["trial"], c["hist_chars"]))
    print("      배선 제안 = `require_doc_before.tools` 선언 확장 + `_ctx_fits` 가 압축 뷰 자수를 쓰도록"
          " 수정(별건 수리·여기서 구현 0).")
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
