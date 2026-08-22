# -*- coding: utf-8 -*-
r"""x466 — **ID-해결 read 생략 · 원장-종류 대조 지목** 격리 A/B/S/N (2026-08-22·[[62]] 정보-맞춘·리뷰 반영판)

## 왜 (정본 `T7336_TASK_079.md`·`T7336_TASK_074.md`·`T7335_NT1_FORENSIC_085_2026_08_21.md`·
##     `T7336_FORENSIC_HALFB_2026_08_22.md` §4.5/§4.6/§6 ②)
t7336 079 t0 / 085 t0 / 074 t0: 모델이 계좌 목록 read(`arg_source_reads.account_id` 선언 원천)를
한 번도 부르지 않은 채 **user_id 나 상품 클래스명을 account_id 자리에** 넣는다 — 079 t0 [25]
`account_id="cr89a2b3c4"`(×5 반복·env *"Account … not found"*), 085 t0 [31] `account_id="f7d3a82c91"`,
074 t0 [4] `account_id="Purple Account"`. 기존 기구는 이 순간을 못 본다: `requires_reads`/READ-FIRST 는
**선언된 write/comparator 의 시도**에서만 서고(079 는 대상 도구에 선언 0), PIN_READ·DEMANDED_STEP 은
요건 큐 머리에서만 선다. *"ID 자리에 다른 종류의 원장 값이 들어간 순간"* 자체를 보는 술어가 없다.

## 가설 (무엇을 가르나 — [[62]] 결손을 격리로 잰 뒤에만 레버)
오투입 호출이 나가고 env 가 거부한 **그 자리**(도구-결과 채널·[[64]] deny 자리)에서 엔진이
**닫힌 술어**로 잡는다 — 인자값이 (ⓐ) 모델 자신이 앞서 **선언된 다른 종류**(원천 read 가 다른)의
인자명으로 보낸 값이거나 (ⓑ) A3 `doc_index` 슬러그의 기계 전개(공식 상품명)와 일치하고, 그러면서
**표적 인자의 선언 원천 read 출력 레코드의 해당 id 필드에는 실재하지 않는다**(정본 파서
`t2_scaffold_get._parse_record_dump` — env 고정 포맷 'Record ID:' 전사·[[59]] 허용 범위) — 그리고
해소-read 를 **이름으로 지목**하는 피드백([[64]]: 무엇이 틀렸나 + 무엇을 하면 풀리나)을 실으면 모델이
그 read 를 부르는가.
    "지목하면 부른다"  → B 에서 pointed_read ⇒ 레버는 **전달(이름 지목)뿐**으로 산다
    "지목해도 같은 오투입" → B 도 same_misinput ⇒ 경계 — 전달로 안 닫히고 처방은 다른 층
N_neg 이 A 와 같아야 "결정점에 뭐라도 찔러서"가 아니고([[57]]), S_sham 이 A 와 같고 B 만 이겨야
"아무 도구 이름이라도 대면 되는 것"이 아니라 **지목된 이름의 내용**이 원인이라 말할 수 있다
([[57]] 인자-변화 통제·x448 sham 관용구).

## 팔 (한 변수만 다르다 · 문맥은 t7336 궤적 축자 복원)
    A_asis      현행 그대로 — 오투입 호출 + env 거부까지 축자(--cut after·기본).
                **A 가 라이브의 다음 행동을 재현해야 측정 유효** — det 1발의 분류가 궤적의 실제
                다음 호출 분류와 같은지 코드로 검사해 JSON `valid` 에 박는다(산문 아님).
    B_pointed   같은 문맥 + 끝 **도구-결과 메시지 끝에 지목 피드백**(--channel tool·기본 —
                라이브 deny 채널·[[64]]·x465.inject 관용구 재사용). 재료 전부 선언 도출:
                표적 인자→원천 read = A3 `arg_source_reads`(**마지막 원소=원천**·앞 원소=선행 read —
                리뷰 확정: 선언의 "목록 순서=해소 순서" 의미를 코드가 이렇게 소비한다고 명시) ·
                호출형 = env 레지스트리(접미사·discoverable 여부) · 클래스명 집합 = A3
                `policy_ontology.doc_index` 슬러그 → `t2_gate_patch._slug_disp`.
                값은 한 개도 열거하지 않는다. **B 구성 전 닫힌 자기검사**: 문면이 과거-사실로
                지칭하는 도구명·값·인자명 전부가 문맥의 tool_calls 구조에 실재해야 하며 아니면
                SystemExit(존재할 수 없는 술어=미래 지식 누설·[[25]]·[[62]] §1.4 — 리뷰 BLOCK①).
    S_sham      B 와 같은 문면에서 **지목 read 이름만** 규칙 치환(비-원천 read·같은 노출 계급·이름순
                첫 항목·x448 `sham_ids` 관용구) — S≈A ∧ B>A 여야 '이름 내용'이 원인([[57]]).
    N_neg       같은 채널·같은 자리에 **무내용 재촉 한 줄**(x465.NUDGE import·[[57]] 부정통제)
  --cut before + --channel user  = 보조 팔(선제 힌트): 오투입 **직전**에서 자르고 user 메시지로
                싣는다. 이 컷에서 결정점 호출은 문맥에 없으므로 문면도 **별도**(ⓐ/ⓑ 증거만·
                'you passed as {표적}' 0 — 리뷰 BLOCK① 분리 지시). ⚠값 선택 자체가 결정점 지식이라
                정보-맞춤이 약한 보조 팔임을 출력에 인쇄한다. 끝이 user 인 sim 은 --merge-user 강제.
  --cut before + --channel tool 은 거부한다(도구-결과 채널은 결정점 뒤에만 존재).
⚠재생 인터페이스 = **라이브 system(LLMAgent 빌더) + 실물 도구 스키마(env + A3 scaffold_get_tools
  · `t2_scaffold_get._augment_byref_params` 로 BYREF 안내까지 라이브 동일) + 실제 메시지 객체 +
  `la.generate`**(x459/x465 관용구·C584: 렌더-텍스트는 약한 인터페이스라 라이브 재현 불가).
  replay·inject·NUDGE 는 x465 에서 **import**(사본 금지·[[67]] — 정본 t2_replay 승격은 후속:
  이 단계는 x466 단독 커밋 제약이라 x465 재사용이 최소 비사본 경로다).
  T2_SG_BYREF·T2_A2_VARIANT 는 미지정 시 **go_stack.sh 의 export 값**으로 맞춘다(라이브 기본 도출).

## 채점 (닫힌 술어만·[[59]]ⓐ — 다음 tool-call 의 구조 사실. gold·reward_info 0·[[23]][[69]])
    pointed_read          unlock/call 대상(또는 직접 호출한 노출 도구)이 지목된 read 집합에 든다
    pointed_read_direct   지목 read 가 discoverable 인데 unlock 없이 직접 호출(라이브선 실패하는 형태)
    sham_read             S 팔의 치환 이름을 부른다(닫힌 통제 칸)
    same_misinput         같은 표적 도구·같은 인자에 다시 종류-불일치 값(같은 술어로 재판정)
    same_tool_unknown_value  같은 표적 도구·같은 인자에 **어느 원장에도 없는 새 값**(날조 도피 칸)
    other_misinput        다른 도구/인자에 종류-불일치 값
    other_tool / no_tool / EXC
  n = det 1발 + temp 6발 = **7/팔 (≥7)** · 결정점 = **census 발화 sim 전부**(기본 — 대상 선정도
  술어가 찾는다·`--cases` 는 덮어쓰기 전용). `--census` 는 결과 파일 전 sim 에 같은 술어를 걸어
  발화 위치(정밀도)와 '원천-미실행 공급' 수(재현율 미지 영역)를 LLM 0 으로 찍는다([[70]] Δspurious).

## [[71]] 격리 서브에이전트 계약 — 4문 답
  1) 기능 하나인가 — 하나다. 각 팔의 서브(=결정점 재생성)는 **다음 발화/호출 생성** 하나만 한다.
     술어 판정·채점은 엔진이 닫힌 동등·멤버십 대조로 바깥에서 한다.
  2) 재료가 A2/A3 선언에서 읽혀 나왔나 — 그렇다. 원천 read 이름 = `arg_source_reads`,
     클래스명 = `doc_index` 슬러그 전개, 호출형 = env 레지스트리, 스키마 = A3 `scaffold_get_tools`
     +정본 빌더. 이 파일에 문서 id·도구명·필드명·태스크별 떠먹이기 리터럴 0([[63]]). gold 0([[23]]).
  3) 전달이 선언된 id → 정확 집기인가 — 그렇다. 지목은 선언된 **이름**을 그대로 싣는다. bm25·embedding
     0·검색 0. 값(어느 account_id 가 맞나)은 싣지 않는다 — 읽는 것은 모델이다.
  4) 엔진이 해석·선택·순위를 하지 않는가 — 안 한다. 원천 나열=선언 순서 그대로, 어느 계좌가
     맞는지·무엇을 부를지는 끝까지 모델 몫(argmax·정답 문장 0·[[62]] "지목은 read 이름까지").

## [[62]] 4문 답
  ① 결손을 격리로 쟀나 — 정본 포렌식이 궤적 전수로 쟀다([S]: 079 t0 ×5·085 t0 ×5·074 t0·t7335 동형
     반복). 이 프로브는 그 결손의 **경계**(지목만으로 닫히는가)를 가른다.
  ② 격리에서 되면 레버는 전달뿐인가 — 그렇다. B 가 이기면 처방은 닫힌 술어 + 이름 지목 문면 하나
     (기존 READ-FIRST 문면 계보)이고 결정론기는 값을 고르지 않는다.
  ③ 사라지는 모델 판단 0 — 어느 read 를 부를지·출력에서 어느 계좌를 집을지는 모델이 한다.
  ④ 엔진 출력에 argmax·최댓값·"정답은 X" 0 — 피드백은 원장-사실 진술 + 선언된 read 이름 나열뿐
     ('is not a valid …' 같은 **판정 문구 0** — 증거가 말하는 데까지만·리뷰 MINOR①).
  ⚠[[59]] 준수: 이 파일에 정규식 0·도구 출력의 자유 텍스트 해석 0. 쓰는 것은 (i) 모델 자신의
     tool_calls JSON 구조 순회 (ii) env 고정 레코드-덤프의 **정본 파서**(`_parse_record_dump`) 전사
     (iii) 선언 집합 멤버십 (iv) 하네스 관례 접두 `Error` 확인(tau2 environment.py 관례)뿐이다.

## [[70]] 병기 — 무엇을 파나
  B 가 판 것 = ⑴ 문맥 +N자(피드백 한 덩이) ⑵ 호출마다 닫힌 술어 1회(정본 파서 레코드 대조·지연 ~0)
  ⑶ 과발화 위험: 손님이 준 정당한 값이 앞선 다른 인자값과 우연히 같으면 오지목 — `--census` 가 전 sim
  발화 위치를 찍어 그 위험을 계수한다. ⑷ 재현율 공백: `arg_source_reads` 를 "마지막 원소=원천" 으로
  소비하므로 대안-나열 키(user_id·transaction_id 류)와 email 계열의 by_email 선언 공백에서는 미발화
  가능 — census 가 '원천-미실행 공급' 수로 그 영역 크기를 함께 인쇄한다(제거 아니라 공개·[[70]]).
  부호표는 결정점 ×1 행씩(태스크별 부호 공개). 격리 결과는 경계 판정이지 승격 근거가 아니다 —
  라이브 효과는 본런 reward A/B 가 판정([[69]]).

## ★계기 수리 (2026-08-22 · 도구-스키마 가드 **과엄격** 으로 fail-closed 중단한 건)
리모트 1차 실행이 *"문맥이 호출한 도구가 재생 스키마에 없다"* 로 SystemExit 했다
(예: `…transfer_1114`·`…dispute_6281`). 그 이름들은 **발견형(discoverable)** 이라 초기
`env.get_tools()` 에 없는 것이 **라이브 정상**이고, 가드가 과엄격이었다.
⛔1차 수리는 *"부여(unlock)를 거치면 스키마에 뜬다"* 고 보고 궤적의 부여를 샌드박스에 재생했는데
**그 전제가 틀렸다** — 2차 리모트 실측: 부여 **6/6 성공인데 `env.get_tools()` 는 17 → 17**.
정본 축자가 그렇게 말한다: *"이 env 의 발견형 도구는 도구 목록에 서지 않고 **디스패처로만** 불린다"*
(`t2_gate_patch.py:2554` · x255 §호출 타입 T2 · C418 격리 인과 "실패는 전부 접미사 이름을 직접
호출하려는 시도"). ⇒ 라이브 도구 지형 = `env.get_tools()` + 우리 scaffold 이고 **부여와 무관하게
결정점마다 같다**. 그래서 가드를 끄지 않고 **판정 축을 둘로** 갈랐다:
    hard        = 래퍼 ∪ 문맥이 **직접**(래퍼 없이) 호출한 도구  → 호출 가능 스키마에 없으면 중단
    dispatcher  = 그 밖의 참조 이름(발견형·지목 read·sham·래퍼 안쪽) → **존재만** 확인하고 인쇄
존재는 툴킷 레지스트리(`env.tools.tools` ∪ `get_discoverable_tools()` — 프레임워크 API·도메인
리터럴 0)에 묻고, **레지스트리에도 없으면 그때가 진짜 낯선 도구라 중단**이다.
★로컬에서 리모트 실패를 잡는 닫힌 불변식(신설): **`hard ∩ 발견형 = ∅`**. 발견형이 hard 에 들어가면
그 자체가 배선 결함이라 로컬 `--wiring-only` 가 SystemExit 한다 — 옛 hard 정의로 계산하면 실제로
6종(`…6281`·`…3892`·`…9173`·`…7823`·`…7483`·`…7291`)이 걸린다(이번에 로컬이 놓친 자리).
`unlocked_before_cut` 은 이제 **보고용**으로만 남는다(그 sim 이 컷 전 서명을 본 발견형이 무엇인지).

## ⛔재생 충실도 판정 (2026-08-22 · 완주 224 표본·예외 0 인데 **8/8 라이브 재현 실패**)
리모트 완주 결과가 `A0 ≠ live` 로 8/8 무효였다. 계기 죽음이 아니라 **재생 충실도** 문제이고,
[[55]] 순서대로 우리 배관부터 봐서 원인을 축자 근거로 확정했다:
  ⑴ **라이브 프롬프트 ≠ 커밋 히스토리.** 라이브는 `system + 압축 커밋 히스토리 + 비커밋 레버 문면`
     을 준다(`t2_gate_patch.py:6588` 압축 → `work = work + [UserMessage(...)]`). t7336 로그 실측:
     sim 하나당 `[T2_LEVER]` **16~30회**. 우리 A_asis 는 그 문면을 **0건** 싣는다.
     복원 시도도 못 한다 — 사이드카 `turn` 이 기록 지점마다 커밋 수(`state.messages`)이거나 버퍼
     길이(`work`)라 생성 경계 조인이 **866/1112(78%)**, 역할 보존이 **933/1112(84%)** 다(t7328
     사이드카 로컬 대조). 정확 복원이 안 되므로 이 파일은 **복원하지 않고 계량**만 한다.
  ⑵ **A_asis 는 무지목 통제가 아니다(3/8).** 컷 꼬리에 이미 다른 read 이름이 나와 있다 — 074 는
     **우리 층 자신의 문면** 축자: *"Error: [READ-FIRST] … the required transaction read is missing:
     get_bank_account_transactions. … unlock it, and call it with the checking account's id …"*
     (궤적 [5]~[8]). 라이브의 다음 행동 [9] 는 `unlock_discoverable_agent_tool(
     get_bank_account_transactions_9173)` = **그 지목을 따른 것**이다. 그래서 분류가 `other_tool`.
     ⇒ 그 자리에서 B 는 *지목 vs 무지목* 이 아니라 **경쟁 지목**(어느 이름을 따르나)을 잰다.
     `B_pointed 47/56` 을 "지목이 먹힌다" 로 읽으면 오독이다([[56]]).
★판정: **이 설계로 라이브 결정점을 재현할 수 없다.** 고칠 수 있는 배관 결함이 아니라 영속 산출물의
  정보 부족이다. 그래서 프로브를 죽이는 대신 **주장 범위를 좁혔다** — A/B/S/N 은 *재구성된 문맥
  안에서의* 격리 비교로만 보고하고([[62]] 경계 판정), 라이브 주장·승격에는 쓰지 않는다([[69]]).
  두 축은 이제 결정점마다 인쇄되고 JSON `fidelity` 에 박힌다(경쟁 지목 목록·비커밋 조인 수).
  ⚠다음에 이 축을 제대로 재려면 프로브가 아니라 **라이브 A/B**(같은 스택에서 지목만 갈아 끼운
  본런)여야 한다 — 결정점 재생으로는 안 된다.

## 실행 (리모트·8141·GPU 유휴 시에만 — 지금은 작성 + 무료 배선 검증까지)
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    python scripts/distill/tau2/x466_id_resolution_iso.py --port 8141
    # 배선만(LLM 0·GPU 불요): ... x466_id_resolution_iso.py --wiring-only [--census]
    # 로컬(윈도우·tau2 없음): py -3 x466_id_resolution_iso.py --wiring-only --census
    # ⚠선행 조건: t2_scaffold_get._augment_byref_params (2026-08-22 추출) 이 없으면 즉시 중단한다.
"""
import argparse
import copy
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

import t2_forensic as F                 # noqa: E402  정본 로더·ToolCall 흡수(사본 금지·[[67]])
import t2_gate_patch as GP              # noqa: E402  `_slug_disp` 정본(공식명 기계 전개)
import t2_callable_hint as CH           # noqa: E402  `_fam` 정본(접미사 제거 = 하네스 명명 관행)
import t2_scaffold_get as SG            # noqa: E402  `_build_tool`·`_variant`·`_augment_byref_params`·
#                                                     `_parse_record_dump`(env 고정 포맷 정본 파서)
import x465_transfer_doc_iso as X465    # noqa: E402  replay·inject·NUDGE 관용구(사본 금지·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)
DOMAIN = "banking_knowledge"

# t7336 실물 경로 — 리모트 라이브 results.json 우선, 없으면 로컬 영속 gz(포렌식 정본과 동일·읽기 전용)
HALVES = ("halfA", "halfB")
D_RESULTS_T = ("/home/woori/scratch/tau2-bench/data/simulations/"
               "bank_t7336_%s_20260821b/results.json")
L_RESULTS_T = os.path.join(REP, "sim_results", "bank_t7336_%s_20260821b.results.json.gz")

# discoverable 사슬의 호출 껍데기 — tau2 하네스 프로토콜(도메인 어휘 아님·t2_forensic 상수 재사용)
DISC_TOOLS = (F.UNLOCK, F.CALLA)

# 하네스 관례: env 는 실패를 플래그 없이 content 접두 "Error" 로만 표시한다(tau2 environment.py·
# HALFB U1/F8 수리 동형). 실패한 read 는 "값을 낸 read" 로 세지 않는다.
ERR_PREFIX = "Error"

# 무내용 재촉([[57]]) — 같은 채널·같은 자리·정보 0 (x465 정본 상수 import·사본 금지).
NUDGE = X465.NUDGE

GO_STACK = os.path.join(HERE, "go_stack.sh")

# 같은 예외가 이만큼 **연속**이면 계기 결함으로 보고 재생을 중단한다([[55]]·GPU 낭비 금지).
EXC_STREAK = 3


def go_stack_default(var):
    """정본 런처 `go_stack.sh` 의 `export VAR=val` 값(마지막 것·없으면 None) — 라이브 기본값을
    **파일에서 도출**한다(추측 0·t7336 은 go_stack 을 source 하므로 meta 'on' 목록 밖 변수의 권위)."""
    got = None
    try:
        with io.open(GO_STACK, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s.startswith("export "):
                    continue
                for tok in s[len("export "):].split("#", 1)[0].split():
                    k, eq, val = tok.partition("=")
                    if eq and k == var:
                        got = val
    except Exception:
        return None
    return got


# ── ① 선언 읽기 (읽을 뿐 고르지 않는다) ─────────────────────────────────────────
def load_a3():
    """A3 정본 층(x465 관용구)."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def declared_kinds(a3):
    """인자명 → 원천 read 목록 (A3 `arg_source_reads`·`_` 로 시작하는 note 키 제외).

    ★소비 의미(리뷰 확정·코드화): 선언의 "목록 순서 = 해소 순서" 를 이 프로브는
    **마지막 원소 = 그 종류의 원천(producer), 앞 원소 = 선행 read** 로 소비한다.
    card_id:[3847, 7823] 에서 원천은 7823 — 그래야 account_id(원천 3847)와 card_id 가
    서로 다른 종류로 갈라져 카드 id↔계좌 id 혼입이 보인다(구판 집합-서로소 해석은 이것을 영영
    못 봤다 — 리뷰 MAJOR⑤). 대안-나열 키(user_id 류)는 이 해석에서 재현율이 준다 —
    선언 분리(`produces`/`prerequisite`)가 정답이고 [[72]] 후속으로 남긴다(census 가 공백을 계수).
    """
    amap = (a3 or {}).get("arg_source_reads") or {}
    return {str(k): [str(x) for x in v] for k, v in amap.items()
            if not str(k).startswith("_") and isinstance(v, list) and v}


def producer_fam(kinds, key):
    """key 의 원천 read 계열(접미사 제거) = 선언 목록의 **마지막 원소**."""
    return CH._fam(kinds[key][-1])


def kind_fields(kinds, key):
    """같은 원천을 공유하는 선언 인자명들 — 레코드 필드 대조 집합(선언 도출·해석 0)."""
    pf = producer_fam(kinds, key)
    return sorted(k for k in kinds if producer_fam(kinds, k) == pf)


def official_names(a3):
    """A3 `policy_ontology.doc_index` 의 주어 슬러그 → 공식명 기계 전개(`_note_official_names` 축자:
    공식명 = 슬러그 전개 **축자**). 슬러그 자체와 전개형 둘 다 닫힌 집합에 넣는다(085 t1 `blue_account`).
    반환: casefold(이름) → (군, 슬러그). 해석 0·출처 = env 파일명(doc_index)뿐."""
    out = {}
    di = ((a3 or {}).get("policy_ontology") or {}).get("doc_index") or {}
    for group, subj in di.items():
        for slug in (subj or {}):
            if str(slug).startswith("_"):
                continue
            for nm in (str(slug), GP._slug_disp(slug)):
                out[" ".join(nm.split()).casefold()] = (str(group), str(slug))
    return out


def surface_registry():
    """env 레지스트리(`a2/env_surface.json` — env 에서 기계 도출된 사본): (agent 도구 전부, 노출, discoverable)."""
    p = os.path.join(HERE, "a2", "env_surface.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)[DOMAIN]
    tools = d.get("tools") or {}
    exposed = {str(x) for x in (d.get("exposed") or [])}
    agent = {str(n) for n, s in tools.items() if (s or {}).get("side") == "tools"}
    return agent, exposed, (agent - exposed)


# ── ② 원장 읽기 — 구조 순회 + 정본 레코드-파서 전사만 (이 파일에 정규식 0) ─────────
def norm_v(v):
    return " ".join(str(v).split())


def leaf_kv(args):
    """구조화된 인자 JSON 의 leaf (key, value) 열거 — 중첩 JSON 문자열은 `F.norm_args` 가 푼다.
    텍스트에서 값을 뽑지 않는다: 순회 대상은 **모델 자신이 낸 tool_calls 구조**뿐이다."""
    out = []

    def walk(o, key):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, str(k))
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x, key)
        elif o is not None and key is not None:
            s = norm_v(o)
            if s:
                out.append((key, s))

    walk(F.norm_args(args), None)
    return out


def eff_name(tc):
    """실행 대상 이름(정확·접미사 포함): `call_` 래퍼는 안쪽, unlock 은 겉(실행이 아니므로)."""
    nm = F.nameof(tc)
    inner = F.inner_name(F.argsof(tc))
    return inner if (nm == F.CALLA and inner) else nm


def executed(ctx):
    """[(msg_idx, 실행명, 겉이름, 출력 content, ok)] — 호출과 같은 id 의 tool 메시지를 짝짓는다.
    ok = 출력이 있고 error 플래그 없음·content 가 하네스 `Error` 접두가 아님."""
    outs = {}
    for i, m in enumerate(ctx):
        if str(m.get("role") or "") == "tool" and m.get("id") is not None:
            outs[m.get("id")] = m
    rows = []
    for i, m in enumerate(ctx):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            o = outs.get(F._as_dict(tc).get("id"))
            c = str((o or {}).get("content") or "")
            ok = bool(o) and not bool((o or {}).get("error")) and not c.lstrip().startswith(ERR_PREFIX)
            rows.append((i, eff_name(tc), F.nameof(tc), c, ok))
    return rows


def own_arg_index(ctx):
    """값 → [(인자명, 실행명, msg_idx)] — 모델 자신이 앞서 어떤 **이름**으로 그 값을 보냈나(구조만)."""
    idx = {}
    for i, m in enumerate(ctx):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            e = eff_name(tc)
            for k, v in leaf_kv(F.argsof(tc)):
                idx.setdefault(v, []).append((k, e, i))
    return idx


def rows_of(ctx, fam):
    """fam 계열 read 의 ok 출력들을 **정본 파서**(`SG._parse_record_dump` — env 고정 'Record ID:'
    포맷 전사·[[59]] 허용 범위)로 레코드 dict 목록으로. 파싱 불가 출력(레코드 덤프 아님)은 0 레코드."""
    rows = []
    for (_i, e, _n, out, ok) in executed(ctx):
        if ok and CH._fam(e) == fam:
            try:
                rows += SG._parse_record_dump(out)
            except Exception:
                pass
    return rows


def field_present(rows, fields, v):
    """레코드들의 지정 **id 필드** 값과의 닫힌 동등(공백 정규화·casefold) — 임의 substring 아님
    (리뷰 MAJOR⑤: 계좌 레코드가 user_id 필드를 동봉해도 user_id-as-account_id 를 계속 본다)."""
    vf = norm_v(v).casefold()
    return any(norm_v(r[f]).casefold() == vf for r in rows for f in fields if f in r)


def kind_mismatch(ctx, key, value, kinds, names):
    """★닫힌 술어 — 값 `value` 가 인자 `key` 의 종류가 아니라는 **원장 사실**이 있는가.

    None  = 판정 없음(선언 밖 인자·빈 값·**표적 원천 read 의 레코드에서 해당 종류 필드로 실재** =
            정당할 수 있음). dict = 증거: as_other_key(모델이 다른-원천 인자명으로 보낸 값) /
            as_field_of(다른 종류 원천 레코드의 그 종류 id 필드에 실재) / class_name(A3 공식 상품명).
    트리거는 ⓐ as_other_key ⓑ class_name 뿐 — as_field_of 는 보조 문면(1차 census 과발화 교정 유지).
    판단 0: 어느 값이 맞는지는 말하지 않는다. 손님이 새로 준 값(어느 증거에도 없음)은 잡지 않는다.
    """
    v = norm_v(value)
    if not v or key not in kinds:
        return None
    pf = producer_fam(kinds, key)
    if field_present(rows_of(ctx, pf), kind_fields(kinds, key), v):
        return None                          # 표적 종류의 원천 레코드에 그 종류 필드로 실재 → 오투입 아님
    ran_ok = sorted({e for (_i, e, _n, _o, ok) in executed(ctx) if ok and CH._fam(e) == pf})
    ev = {"key": key, "value": v, "producers": list(kinds[key]), "producer": kinds[key][-1],
          "producer_ran": ran_ok}
    # 트리거 ⓐ — 모델 자신이 **원천이 다른** 선언 인자명으로 보낸 값. 선언 밖 키(query 등)·동일-원천
    #   키(phone↔phone_number)는 종류 증거가 아니다 — 1차 census 에서 그 둘을 안 가려 22/40 sim
    #   과발화(신원확인 전화·이메일 전부 오지목)가 났다([[70]] Δspurious 계측).
    others = [(k, e, i) for (k, e, i) in own_arg_index(ctx).get(v, [])
              if k != key and k in kinds and producer_fam(kinds, k) != pf]
    if others:
        ev["as_other_key"] = others[:6]
    # 트리거 ⓑ — A3 공식 상품명(doc_index 슬러그 전개)과 동등.
    cn = names.get(v.casefold())
    if cn:
        ev["class_name"] = list(cn)
    if not (others or cn):
        return None
    # 보조(트리거 아님) — 값이 **다른 종류** 원천 read 의 레코드에서 그 종류의 id 필드로 실재한다는
    #   사실(정본 파서 필드-동등·임의 substring 아님). 문면에서 출처만 말한다.
    seen_fams, appears = set(), []
    for k2 in sorted(kinds):
        f2 = producer_fam(kinds, k2)
        if f2 == pf or f2 in seen_fams:
            continue
        seen_fams.add(f2)
        r2 = rows_of(ctx, f2)
        for k3 in kind_fields(kinds, k2):
            if field_present(r2, [k3], v):
                appears.append((k3, kinds[k2][-1]))
    if appears:
        ev["as_field_of"] = appears[:6]
    return ev


# ── ③ 결정점 — 첫 종류-불일치 호출 (태스크별 인덱스 리터럴 0) ────────────────────
def find_dp(msgs, kinds, names):
    """첫 오투입 = 선언된 종류 인자에 `kind_mismatch` 가 서는 **첫** assistant 호출. 없으면 None."""
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") != "assistant":
            continue
        ctx = msgs[:i]
        for tc in (m.get("tool_calls") or []):
            for k, v in leaf_kv(F.argsof(tc)):
                ev = kind_mismatch(ctx, k, v, kinds, names)
                if ev:
                    return {"i_dp": i, "tool": eff_name(tc), "outer": F.nameof(tc), "ev": ev,
                            "call": [F.nameof(tc), F.argsof(tc)]}
    return None


def unexplained_supplies(msgs, kinds, names):
    """census 재현율-공백 계수([[70]] 공개): 선언 종류 인자에 값이 실렸는데 ⑴ 그 원천 read 가 ok 로
    돈 적 없고 ⑵ 값이 원천 레코드에 실재하지 않고 ⑶ 트리거 증거도 0 인 호출 수 — 술어가 원리상
    못 보는(선언 공백/증거 공백) 영역의 크기다. 판정이 아니라 **미지 영역 계수**다."""
    n = 0
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") != "assistant":
            continue
        ctx = msgs[:i]
        for tc in (m.get("tool_calls") or []):
            for k, v in leaf_kv(F.argsof(tc)):
                if k not in kinds or not norm_v(v):
                    continue
                if kind_mismatch(ctx, k, v, kinds, names) is not None:
                    continue
                pf = producer_fam(kinds, k)
                if any(ok and CH._fam(e) == pf for (_i, e, _n, _o, ok) in executed(ctx)) \
                        and field_present(rows_of(ctx, pf), kind_fields(kinds, k), v):
                    continue
                n += 1
    return n


def cut_context(msgs, i_dp, cut):
    """before = 오투입 직전까지 · after = 오투입 호출 + 그 도구 출력(들)까지(라이브 deny 자리·기본)."""
    if cut == "before":
        return msgs[:i_dp]
    j = i_dp + 1
    while j < len(msgs) and str(msgs[j].get("role") or "") == "tool":
        j += 1
    return msgs[:j]


def live_next_calls(msgs, ctx_len):
    """문맥 바로 다음의 **라이브** assistant 행동(tool_calls·없으면 빈 목록) — A 재현성 판정 기준."""
    for m in msgs[ctx_len:]:
        if str(m.get("role") or "") == "assistant":
            return [(F.nameof(tc), F.argsof(tc)) for tc in (m.get("tool_calls") or [])]
    return None


# ── ④ 피드백 — [[64]] 두 칸(무엇이 틀렸나 + 무엇을 하면 풀리나) · 값 열거 0 ────────
def callable_forms(prods, agent, exposed, disc):
    """선언 원천 read 의 **부를 수 있는 형태**(t2_callable_hint 문면 계보·레지스트리 실재 검증)."""
    forms, unresolved = [], []
    for p in prods:
        if p in disc:
            forms.append('%s(agent_tool_name="%s") then %s with that name' % (F.UNLOCK, p, F.CALLA))
        elif p in exposed or p in agent:
            forms.append("call %s directly" % p)
        else:
            unresolved.append(p)
    return forms, unresolved


def _evidence_parts(ev, k):
    """ⓐ/ⓑ/보조 증거의 **증거-진술** 문장들(판정 문구 0·리뷰 MINOR①)."""
    parts = []
    if ev.get("as_other_key"):
        ks = sorted({x[0] for x in ev["as_other_key"]})
        es = sorted({x[1] for x in ev["as_other_key"]})
        parts.append("In this conversation you yourself sent that exact value as %s (in %s)."
                     % (", ".join(ks), ", ".join(es)))
    if ev.get("as_field_of"):
        fs = sorted({x[0] for x in ev["as_field_of"]})
        rs = sorted({x[1] for x in ev["as_field_of"]})
        parts.append("In the records read so far it appears only as a %s field (output of %s), "
                     "never as a %s field." % (", ".join(fs), ", ".join(rs), k))
    if ev.get("class_name"):
        parts.append("'%s' is a product class name on file (%s), not a %s value."
                     % (ev["value"], ev["class_name"][0], k))
    return parts


def _resolution_parts(k, ev, kinds, agent, exposed, disc, tail):
    """무엇을 하면 풀리나 — 선언 원천 read 이름 나열 + 호출형(선택 0·값 0)."""
    parts = []
    prods = kinds[k]
    if ev.get("producer_ran"):
        parts.append("The %s values read in this conversation are those in the output of %s - "
                     "copy one from there exactly as written." % (k, ", ".join(ev["producer_ran"])))
        return parts
    parts.append("No %s-producing read has run in this conversation yet. %s values come from the "
                 "output of: %s." % (k, k, ", then ".join(prods)))
    forms, unresolved = callable_forms(prods, agent, exposed, disc)
    if forms:
        parts.append("Their exact callable forms are: %s." % "; ".join(forms))
    if unresolved:
        parts.append("The remaining names (%s) must still be looked up in the knowledge base."
                     % ", ".join(unresolved))
    parts.append(tail % k)
    return parts


def feedback(ev, tool, kinds, agent, exposed, disc, cut):
    """cut=after: 방금 나간(문맥에 실재하는) 호출을 지칭하는 deny 문면.
    cut=before: **선제 힌트** — 결정점 호출이 문맥에 없으므로 그 호출을 지칭하는 절 0
    ('you passed … to <표적>' 0·ⓐ/ⓑ 증거만·리뷰 BLOCK① 분리)."""
    k, v = ev["key"], ev["value"]
    if cut == "after":
        parts = ["Error: [ID-KIND] '%s' was supplied as %s to %s, but no record read in this "
                 "conversation shows it as a %s." % (v, k, tool, k)]
        parts += _evidence_parts(ev, k)
        parts += _resolution_parts(k, ev, kinds, agent, exposed, disc,
                                   "Do that read first, copy the %s from its output, then re-issue "
                                   "your call.")
        return " ".join(parts)
    parts = ["[ID-KIND] No record read in this conversation shows '%s' as a %s." % (v, k)]
    parts += _evidence_parts(ev, k)
    parts += _resolution_parts(k, ev, kinds, agent, exposed, disc,
                               "Before passing any %s, do that read and copy the value from its "
                               "output.")
    return " ".join(parts)


def assert_fb_grounded(ctx, dp, cut):
    """★닫힌 자기검사(리뷰 BLOCK①): 문면이 **과거-사실**로 지칭하는 도구명·값·인자명이 전부
    문맥의 tool_calls 구조(leaf_kv/own_arg_index)에 실재해야 한다. 아니면 SystemExit —
    존재할 수 없는 술어(미래 지식)를 실은 팔은 재기 전에 무효다([[25]]·[[62]] §1.4)."""
    ev, probs = dp["ev"], []
    idx = own_arg_index(ctx)
    if cut == "after":
        pairs, tools_in = set(), set()
        for m in ctx:
            if str(m.get("role") or "") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                tools_in.add(eff_name(tc))
                for k, v in leaf_kv(F.argsof(tc)):
                    pairs.add((k, v))
        if (ev["key"], ev["value"]) not in pairs:
            probs.append("(%s, '%s') 호출이 문맥에 없다" % (ev["key"], ev["value"]))
        if dp["tool"] not in tools_in:
            probs.append("표적 도구 %s 호출이 문맥에 없다" % dp["tool"])
    for (k, e, _i) in (ev.get("as_other_key") or []):
        if not any(k == k2 and e == e2 for (k2, e2, _j) in idx.get(ev["value"], [])):
            probs.append("as_other_key %s@%s 가 문맥 구조에 없다" % (k, e))
    for (f, rd) in (ev.get("as_field_of") or []):
        if not field_present(rows_of(ctx, CH._fam(rd)), [f], ev["value"]):
            probs.append("as_field_of %s@%s 가 문맥 레코드에 없다" % (f, rd))
    if probs:
        raise SystemExit("B 문면 자기검사 실패(미래 지식/허위 지칭·[[25]]): %s" % "; ".join(probs))


def place_user(ctx, text, merge_user=False):
    """문맥 사본 끝에 user 채널 메시지로 싣는다(x459⒝ 관용구). 새 메시지는 **content·role·turn_idx 만**
    갖는다 — 과거 timestamp/usage 등 메타 복제 금지(리뷰 MINOR⑤). merge_user=True 면 끝이 user 일 때
    그 본문에 덧붙인다."""
    out = copy.deepcopy(ctx)
    if merge_user and out and str(out[-1].get("role") or "") == "user":
        out[-1]["content"] = str(out[-1].get("content") or "") + NLC + NLC + text
        return out
    m = {"role": "user", "content": text}
    if out and isinstance(out[-1].get("turn_idx"), int):
        m["turn_idx"] = int(out[-1]["turn_idx"]) + 1
    out.append(m)
    return out


def place(ctx, text, channel, merge_user):
    """채널 배치: tool = 문맥 마지막 도구-출력 끝에 덧붙임(x465.inject 재사용 — 라이브 deny 채널·
    [[64]]·리뷰 MAJOR②) · user = 문맥 끝 user 메시지(보조 팔)."""
    if channel == "tool":
        return X465.inject(ctx, text)
    return place_user(ctx, text, merge_user)


def sham_names(kinds, key, agent, exposed, disc, muts, n):
    """S_sham 치환 이름(리뷰 MINOR③·[[57]] 인자-변화 통제·x448 `sham_ids` 관용구): **비-원천**
    read(어느 선언 종류의 원천도 아님·비변이)를 지목 read 와 같은 노출 계급에서 **이름순**으로 n개.
    규칙 하나뿐 — 엔진이 내용으로 고르지 않는다."""
    banned = {CH._fam(p) for ps in kinds.values() for p in ps}
    pool = disc if kinds[key][-1] in disc else (exposed & agent)
    out = []
    for nm in sorted(pool):
        if CH._fam(nm) in banned or nm in muts or nm in DISC_TOOLS:
            continue
        out.append(nm)
        if len(out) >= n:
            break
    return out


# ── ⑤ 재생 — 실물 도구 스키마 + 라이브 system + 실제 메시지 객체 (x465.replay 재사용) ─
def unlocked_before_cut(ctx):
    """★궤적에서 **성공한 부여**(unlock/give)로 그 시점 노출돼 있던 discoverable 도구 — 기계 도출.

    라이브에서 discoverable 도구는 부여가 성공해야 **에이전트 스키마에 뜬다**(그 전에는 래퍼
    `call_discoverable_*` 로만 닿는다). 그래서 초기 `env.get_tools()` 에 없는 것이 정상이고, 재생
    스키마는 **그 sim 이 그 시점까지 실제로 연 것**만 담아야 라이브와 같은 도구 지형이 된다.
    성공 판정 = 그 호출 id 의 도구-결과가 `error` 플래그도 아니고 하네스 관례 접두 `Error` 도 아님
    (tau2 environment.py 관례·이 파일의 기존 술어 재사용). 이름 리터럴 0 — 전부 궤적 축자.
    """
    want, ok = {}, set()
    for m in ctx:
        r = str(m.get("role") or "")
        if r == "assistant":
            for tc in (m.get("tool_calls") or []):
                if F.nameof(tc) not in F.GRANTS:
                    continue
                inner = F.inner_name(F.argsof(tc))
                tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if inner:
                    want[tid] = inner
        elif r == "tool":
            rid = m.get("id") or m.get("tool_call_id")
            if rid in want and not m.get("error") \
                    and not str(m.get("content") or "").startswith(ERR_PREFIX):
                ok.add(want[rid])
    return ok


def called_directly(ctx):
    """문맥에서 **래퍼 없이 직접** 호출된 도구 이름 — 그 시점 스키마에 반드시 있던 것들이다."""
    out = set()
    for m in ctx:
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm = F.nameof(tc)
            if nm and nm not in F.WRAPPERS:
                out.add(nm)
    return out


def env_registry_names(sb):
    """env 의 **전체 에이전트 도구 이름**(발견형 포함) — 존재 여부 판정의 권위원.

    이 env 에서 발견형 도구는 **도구 목록에 서지 않고 디스패처로만** 불린다(정본 축자:
    `t2_gate_patch.py:2554`·x255 §호출 타입 T2). 그러니 `get_tools()` 에 없는 것은 정상이고,
    "존재하는가" 는 툴킷 레지스트리(`env.tools.tools` ∪ `get_discoverable_tools()`)에 물어야 한다.
    프레임워크 API 만 쓴다 — 도메인 리터럴 0. 못 물으면 빈 집합(호출부가 폴백을 인쇄한다).
    """
    try:
        tk = getattr(sb.env, "tools", None)
        if tk is None:
            return set()
        names = {str(n) for n in (getattr(tk, "tools", {}) or {})}
        g = getattr(tk, "get_discoverable_tools", None)
        if callable(g):
            names |= {str(n) for n in (g() or {})}
        return names
    except Exception:
        return set()


# ★재생 충실도 상수 (2026-08-22 실측 — t7328 halfB 사이드카 1,112행 로컬 대조)
#   비커밋 사이드카(`fb_<tag>.jsonl`)의 `turn` 은 기록 지점마다 **커밋 수**(`state.messages`)이거나
#   **버퍼 길이**(`work`)라 생성 경계와 정확히 맞지 않는다: 866/1112 = **78%** 만 assistant 경계에
#   떨어지고 207 은 role=tool, 34 는 role=user, 5 는 범위 밖이다. 역할은 `record_many` 경유
#   933/1112 = **84%** 만 남는다(단건 `record()` 는 역할 미기록).
#   ⇒ 사이드카로 라이브 프롬프트를 **정확히** 복원할 수 없다. 그래서 이 파일은 복원을 시도하지 않고
#   **간극을 계량**해 인쇄한다([[55]] 침묵 금지·수를 지어내지 않는다).
FB_TURN_JOIN = "866/1112(78%)"
FB_ROLE_KNOWN = "933/1112(84%)"


def declared_read_universe(kinds):
    """A3 `arg_source_reads` 가 선언한 **모든** read 이름(선언 출처·리터럴 0)."""
    out = set()
    for v in kinds.values():
        out |= {str(x) for x in (v or ())}
    return out


def named_reads_in_tail(ctx, i_dp, universe, pointed):
    """★컷 꼬리(결정점 호출의 도구 출력들)에 **이미 이름이 나와 있는** 선언 read 들 — 닫힌 멤버십.

    엔진은 도메인 텍스트를 해석하지 않는다([[59]]): **선언된 식별자**가 축자로 있는지만 본다
    (이 파일의 `assert_fb_grounded` 와 같은 술어 모양). 접미사 해제는 정본 `CH._fam`.
    왜 필요한가 — 실측: 결정점 8개 중 4개(074×2·094×2)의 컷 꼬리에 **우리 층 자신의 지목 문면**이
    이미 들어 있다(축자: *"Error: [READ-FIRST] … the required transaction read is missing:
    get_bank_account_transactions. … unlock it, and call it …"*). 라이브의 다음 행동은 그 지목을
    **따른 것**(`unlock_discoverable_agent_tool(get_bank_account_transactions_9173)`)이다.
    ⇒ 그 결정점에서 `A_asis` 는 **무지목 통제가 아니고**, B 는 *지목 vs 무지목* 이 아니라
    **다른 이름을 가리키는 경쟁 지목**이다. 이 사실 없이 B 의 부호를 읽으면 오독한다([[56]]).
    반환 = {"pointed_in_tail": bool, "competing": [이름…]}
    """
    tail = " ".join(str(m.get("content") or "") for m in ctx[i_dp:]
                    if str(m.get("role") or "") == "tool")
    fams = {CH._fam(n) for n in pointed}
    hit = {n for n in universe if n in tail or CH._fam(n) in tail}
    return {"pointed_in_tail": bool({n for n in hit if CH._fam(n) in fams}),
            "competing": sorted(n for n in hit if CH._fam(n) not in fams)}


def noncommitted_census(rows_by_sim, simtag, turn):
    """★라이브가 그 생성에 본 **비커밋 문면**의 크기 — 우리 재생에는 없는 부분을 계량한다.

    재료 = 우리 런의 사이드카(정본 `F.sidecar_rows`). 정확 복원은 위 상수대로 불가능하므로
    **복원하지 않고 센다**: 그 턴에 조인되는 행 수·문자 수·채널. 사이드카가 없으면(로컬) 0 을
    돌려주고 호출부가 '측정 불가'로 인쇄한다.
    """
    rows = [r for r in rows_by_sim.get(simtag, ()) if r.get("turn") == turn]
    return {"n": len(rows), "chars": sum(int(r.get("len") or 0) for r in rows),
            "kinds": sorted({str(r.get("kind")) for r in rows})}


AXIS_EXPOSED = "exposed"        # 도구 목록에 직접 서 있다 — 그대로 호출
AXIS_DISC = "discoverable"      # 레지스트리에만 있다 — 디스패처(`call_discoverable_*`) 경유


def name_axis(name, callable_names, reg_names):
    """★도구 이름이 **어느 축**에 있나 — 이 술어가 정본 한 자리다(x470 도 이걸 import 한다·[[67]]).

    이 env 에서 발견형 도구는 도구 목록에 서지 않고 **디스패처로만** 불린다
    (`t2_gate_patch.py:2554` 축자 · x255 §호출 타입 T2 · 리모트 실측: 부여 6/6 성공인데
    `env.get_tools()` 는 17 → 17). 그래서 *"이 이름이 존재하는가"* 는 **두 축을 함께** 물어야 하고,
    `get_tools()` 한 축으로만 물으면 gold 변이 도구조차 '없는 이름'이 된다(x470 1차 FAIL 의 원인).
    접미사 해제는 정본 `t2_callable_hint._fam`(`_NNNN` = env 명명 관행) 폴백이다.
    반환: AXIS_EXPOSED / AXIS_DISC / None(어느 축에도 없음 = 진짜 낯선 이름 → 호출부가 중단한다).
    """
    n = str(name or "")
    if n in callable_names:
        return AXIS_EXPOSED
    if n in reg_names:
        return AXIS_DISC
    fam = CH._fam(n)
    if any(CH._fam(x) == fam for x in callable_names):
        return AXIS_EXPOSED
    if any(CH._fam(x) == fam for x in reg_names):
        return AXIS_DISC
    return None


def scaffold_tools(a3):
    """라이브가 주입하는 A3 `scaffold_get_tools` 를 **같은 빌더 사슬**(`_variant` →
    `_augment_byref_params` → `_build_tool`)로 만든다. 074 결정점 호출이 scaffold 도구 + `@last:`
    참조라 BYREF 안내까지 라이브와 같아야 A 가 재현된다(리뷰 MAJOR④·[[67]] 사본 금지)."""
    if not hasattr(SG, "_augment_byref_params"):
        raise SystemExit("t2_scaffold_get._augment_byref_params 가 없다 — 2026-08-22 추출 커밋을 먼저 "
                         "반영하라(사본 금지·[[67]]·리뷰 MAJOR④)")
    try:
        from tau2.environment.tool import Tool
    except Exception as e:
        print("  ⚠scaffold 도구 빌드 불가(%r) — env 도구만으로 재생(074 재현성 저하)" % (e,))
        return []
    out = []
    for d in (a3.get("scaffold_get_tools") or []):
        try:
            out.append(SG._build_tool(Tool, SG._augment_byref_params(SG._variant(d))))
        except Exception as e:
            print("  ⚠scaffold 도구 %s 빌드 실패: %r" % (d.get("name"), e))
    return out


def system_dicts(policy, tools, model):
    """라이브 빌더로 system 을 만든다(리뷰 MAJOR③): tau2 `LLMAgent(tools, domain_policy).system_prompt`
    (AGENT_INSTRUCTION + policy). T2_RULES_PROMPT 가 서 있으면 라이브 러너(t2_run_gated)와 같이
    `t2_agent_rules_patch.apply()` 를 먼저 적용한다(go_stack 은 --rules_prompt 를 안 넘긴다 = 기본 무).
    x470 `system_messages` 관용구 동형 재작성(그 파일은 같은 배치의 미커밋 프로브라 import 불가 —
    정본 승격은 후속). 반환: ([{"role":"system",...}], 출처표기)."""
    if os.environ.get("T2_RULES_PROMPT"):
        import t2_agent_rules_patch
        t2_agent_rules_patch.apply()
    import tau2.agent.llm_agent as la
    ag = la.LLMAgent(tools=tools, domain_policy=policy, llm="openai/%s" % model, llm_args={})
    try:
        st = ag.get_init_state()
        sm = list(getattr(st, "system_messages", None) or [])
        if sm:
            return ([{"role": "system", "content": str(getattr(x, "content", "") or "")} for x in sm],
                    "state.system_messages")
    except Exception:
        pass
    return [{"role": "system", "content": str(ag.system_prompt)}], "agent.system_prompt"


def replay(sys_msgs, msgs, tools, model, base, temperature, gen=None):
    """x465.replay **재사용**(사본 금지·[[67]]·리뷰 MAJOR⑥) — system 은 dict 로 앞에 끼워 넣으면
    x465 의 CLS 매핑(role=system→SystemMessage)이 같은 경로로 복원한다. call_name 은 x465_replay 로
    찍힌다(공유 함수의 표기·별도 의미 없음).

    반환 = 정본 `X465.ReplayResult`(calls·text·dropped·prompt_tokens) — **이름으로 읽는다**.
    ⛔2026-08-22 사고: 여기 호출부가 3-언팩으로 남아 있어 리모트 재생이 **전 표본**
    `ValueError: too many values to unpack (expected 3)` 로 죽었다(배선 검산은 재생 경로를 안 타서
    못 잡았다). 그래서 ⑴ 계약을 이름 있는 필드로 고정하고 ⑵ `--wiring-only` 가 `gen` 주입으로
    이 경로를 합성 표본 하나로 태운다.
    """
    return X465.replay(list(sys_msgs) + list(msgs), tools, model, base, temperature, gen=gen)


def shot(c, arm, cx, k, t, sys_msgs, tools, model, base, kinds, names, disc, gen=None):
    """★한 표본의 **전 경로** — 재생 → 언팩 → 분류 → 행 조립 → 인쇄줄. 실제 루프와 배선 검산이
    **같은 이 함수**를 부른다(그래야 "배선 PASS 인데 재생에서 죽음" 이 구조적으로 불가능해진다).
    """
    r = replay(sys_msgs, cx, tools, model, base, t, gen=gen)
    cat, cats = classify(r.calls, c["ctx"], c["dp"], kinds, names, disc, c["shams"])
    row = {"sim": F.sim_key(c["sim"]), "arm": arm, "k": k, "temp": t, "cat": cat, "cats": cats,
           "calls": [[nm, json.dumps(ag, ensure_ascii=False, default=str)[:200]]
                     for nm, ag in r.calls],
           "dropped_msgs": r.dropped, "delta_chars": c["delta"][arm],
           "prompt_tokens": r.prompt_tokens,
           "text": " ".join(r.text.split())[:300]}
    line = ("  #%d t=%.1f  %-15s calls=%s%s"
            % (k, t, SHORT.get(cat, cat),
               ",".join(F.label(nm, ag) for nm, ag in r.calls) or "-",
               ("  ⚠복원 누락 %d" % r.dropped) if r.dropped else ""))
    return row, line


class _FakeCall(object):
    """합성 tool_call — 라이브 ToolCall 의 속성 모양(`F.nameof`/`F.argsof` 가 흡수하는 경로)."""

    def __init__(self, name, arguments):
        self.name, self.arguments, self.id = name, arguments, "wiring"


class _FakeResp(object):
    def __init__(self, calls):
        self.tool_calls = list(calls)
        self.content = "" if calls else "(synthetic text-only)"
        self.usage = {"prompt_tokens": 0}


def selftest_shot(c, sys_msgs, tools, model, base, kinds, names, disc):
    """★배선 검산이 **재생 경로를 태운다**(LLM 0·GPU 0·2026-08-22 신설).

    `gen` 주입으로 `la.generate` 를 합성 응답으로 갈아 끼우고 `shot()` 을 그대로 통과시킨다 —
    언팩·분류·행 조립·인쇄줄까지 실제 루프와 **같은 코드**다. 기대 분류까지 대조하므로 "안 죽었다"
    가 아니라 "제대로 돈다" 를 본다. 표본은 이 결정점의 선언에서 나온다(리터럴 0):
      ⑴ 지목 read 를 unlock → `pointed_read`   ⑵ 호출 0 → `no_tool`
    """
    first = kinds[c["dp"]["ev"]["key"]][-1]
    cases = [("unlock(원천)", [_FakeCall(F.UNLOCK, {"agent_tool_name": first})], "pointed_read"),
             ("무호출", [], "no_tool")]
    saved = None
    try:
        import tau2  # noqa: F401
    except Exception:
        # 메시지 객체 변환만 대역으로 통과시킨다 — 나머지(언팩·분류·행 조립)는 실물 코드다.
        saved, X465.to_objs = X465.to_objs, (lambda ms: (list(ms), 0))
        print("   ⚠tau2 없음 — 합성 재생의 **메시지 객체 변환만** 대역(리모트는 실물·[[55]] 침묵 금지)")
    ok = True
    for label, calls, want in cases:
        try:
            row, line = shot(c, "A_asis", c["arm_ctx"]["A_asis"], 0, 0.0, sys_msgs, tools,
                             model, base, kinds, names, disc,
                             gen=lambda **kw: _FakeResp(calls))
        except Exception as e:
            print("   FAIL 합성 재생(%s) — %s %r" % (label, type(e).__name__, e))
            ok = False
            break
        good = row["cat"] == want
        ok &= good
        print("   %-4s 합성 재생·언팩·분류·행조립(%s) → %s (기대 %s)%s"
              % ("ok" if good else "FAIL", label, row["cat"], want, line))
    if saved is not None:
        X465.to_objs = saved
    return bool(ok)


# ── ⑥ 채점 — 닫힌 분류 ─────────────────────────────────────────────────────────
ORDER = ("pointed_read", "pointed_read_direct", "sham_read", "same_misinput",
         "same_tool_unknown_value", "other_misinput", "other_tool")
SHORT = {"pointed_read": "pointed", "pointed_read_direct": "pointed_direct", "sham_read": "sham",
         "same_misinput": "same_mis", "same_tool_unknown_value": "same_unknown",
         "other_misinput": "other_mis", "other_tool": "other", "no_tool": "no_tool", "EXC": "EXC"}


def classify(calls, ctx, dp, kinds, names, disc, shams=()):
    key = dp["ev"]["key"]
    pointed = {CH._fam(p) for p in kinds[key]}
    pf = producer_fam(kinds, key)
    sham_fams = {CH._fam(s) for s in shams}
    cats = []
    for nm, ag in calls:
        inner = F.inner_name(ag)
        tgt = inner if (nm in DISC_TOOLS and inner) else nm
        tf = CH._fam(tgt)
        if tf in pointed and (nm in DISC_TOOLS or nm == tgt):
            # discoverable 을 unlock 없이 직접 — 라이브에선 그 호출이 실패한다(리뷰 MINOR②)
            cats.append("pointed_read_direct" if (nm == tgt and tgt in disc) else "pointed_read")
            continue
        if sham_fams and tf in sham_fams and (nm in DISC_TOOLS or nm == tgt):
            cats.append("sham_read")
            continue
        mis = [(k, v) for k, v in leaf_kv(ag) if kind_mismatch(ctx, k, v, kinds, names)]
        if mis:
            same = (tf == CH._fam(dp["tool"]) and any(k == key for k, _v in mis))
            cats.append("same_misinput" if same else "other_misinput")
            continue
        if tf == CH._fam(dp["tool"]):
            vals = [v for k, v in leaf_kv(ag) if k == key]
            if vals and any(not field_present(rows_of(ctx, pf), kind_fields(kinds, key), v)
                            for v in vals):
                cats.append("same_tool_unknown_value")   # 증거 0 의 새 값(날조 도피 칸·리뷰 MINOR②)
                continue
        if nm:
            cats.append("other_tool")
    return next((c for c in ORDER if c in cats), "no_tool"), cats


# ── ⑦ 로딩 ───────────────────────────────────────────────────────────────────────
def default_results():
    """기본 = halfA+halfB **둘 다**(리뷰 MINOR④ — census 8/40 이 재현되는 분모를 기본값으로)."""
    outs = []
    for h in HALVES:
        d = D_RESULTS_T % h
        outs.append(d if os.path.exists(d) else L_RESULTS_T % h)
    return ",".join(outs)


def load_sims(path):
    sims = F.sims(path)
    if not sims:
        raise SystemExit("sim 0 건: %s" % path)
    return sims


def pick(sims, suffix, trial):
    got = [s for s in sims if str(F.task_id(s)).split("_")[-1] == str(suffix)
           and (trial is None or s.get("trial") == trial)]
    got.sort(key=lambda s: s.get("trial") or 0)
    if not got:
        raise SystemExit("task %s trial %s 의 sim 이 없다" % (suffix, trial))
    return got[0]


def fmt_ev(ev):
    bits = []
    if ev.get("as_other_key"):
        bits.append("as_other_key=" + ",".join("%s@%s[%d]" % x for x in ev["as_other_key"][:3]))
    if ev.get("as_field_of"):
        bits.append("as_field_of=" + ",".join("%s@%s" % x for x in ev["as_field_of"][:3]))
    if ev.get("class_name"):
        bits.append("class_name=%s/%s" % tuple(ev["class_name"]))
    bits.append("producer_ran=%s" % (ev.get("producer_ran") or "-"))
    return " · ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--results", default=default_results(), help="결과 파일(콤마로 여럿·기본 halfA+halfB)")
    ap.add_argument("--cases", default=None,
                    help="task:trial 목록 — **덮어쓰기 전용**. 기본 = census 발화 sim 전부(대상 선정도 술어)")
    ap.add_argument("--cut", choices=("before", "after"), default="after",
                    help="after(기본)=오투입+env 거부까지(술어가 설 수 있는 유일 자리·리뷰 BLOCK①) · "
                         "before=선제 힌트 보조 팔(--channel user 전용)")
    ap.add_argument("--channel", choices=("tool", "user"), default="tool",
                    help="tool(기본)=마지막 도구-출력 끝에 덧붙임(라이브 deny 채널·x465.inject) · user=보조")
    ap.add_argument("--n", type=int, default=6, help="temp 표본 수 — det 1발이 늘 앞선다(합 n+1≥7)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--arms", default="A_asis,B_pointed,S_sham,N_neg")
    ap.add_argument("--merge-user", action="store_true", help="끝이 user 면 새 메시지 대신 본문에 덧붙임")
    ap.add_argument("--no-system", action="store_true",
                    help="라이브 system(LLMAgent 빌더·기본 ON) 생략 — 진단용. 기본은 라이브와 같은 system")
    ap.add_argument("--census", action="store_true", help="결과 파일 전 sim 의 첫 발화 위치 찍기(LLM 0)")
    ap.add_argument("--wiring-only", action="store_true", help="배선 검증만(LLM 0·GPU 불요·[[55]])")
    ap.add_argument("--out", default="x466_id_resolution_iso.json")
    a = ap.parse_args()

    if a.cut == "before" and a.channel != "user":
        raise SystemExit("--cut before 는 선제-힌트 보조 팔이라 --channel user 전용이다 "
                         "(도구-결과 deny 채널은 결정점 **뒤**에만 존재·리뷰 BLOCK①/MAJOR②)")

    # ── 라이브 기본값 정합(리뷰 MAJOR④): BYREF·A2 변이는 go_stack 의 export 값으로 ────
    for var in ("T2_SG_BYREF", "T2_A2_VARIANT"):
        gs = go_stack_default(var)
        if var not in os.environ and gs is not None:
            os.environ[var] = gs
        print("[배선] %s=%s (go_stack 기본=%s·환경 우선)" % (var, os.environ.get(var), gs))

    # ── 선언 ────────────────────────────────────────────────────────────────────
    a3 = load_a3()
    kinds = declared_kinds(a3)
    names = official_names(a3)
    if not kinds:
        raise SystemExit("A3 arg_source_reads 선언이 비어 있다 — 층 정합부터([[24]])")
    agent, exposed, disc = surface_registry()
    muts = F.mutating_tools(DOMAIN)
    print("=" * 100)
    print("x466 · 선언: 종류 인자 %d개(%s) · 공식명 %d건 · env 레지스트리 agent %d(노출 %d·discoverable %d)"
          % (len(kinds), ", ".join(sorted(kinds)), len(names), len(agent), len(exposed), len(disc)))
    bad = sorted({p for ps in kinds.values() for p in ps if p not in agent})
    if bad:
        print("  ⚠선언 원천 read 가 레지스트리에 없다: %s — 선언 정합부터([[24]])" % bad)

    paths = [p.strip() for p in a.results.split(",") if p.strip()]
    sims_by = {p: load_sims(p) for p in paths}

    # ── census (과발화 정밀도 + 재현율-공백 계수·LLM 0) ─────────────────────────
    fired = []                                    # [(path, sim)] — 기본 cases 도출에도 쓴다
    for p, sims in sims_by.items():
        for s in sims:
            dp = find_dp(s.get("messages") or [], kinds, names)
            if dp is not None:
                fired.append((p, s, dp))
    if a.census:
        print(NLC + "── census: 결과 파일 전 sim 에 같은 술어 — 첫 발화 위치 · 미지 영역 계수 ──")
        n_all = n_unex_sims = 0
        firemap = {(id(s)): dp for (_p, s, dp) in fired}
        for p, sims in sims_by.items():
            print("  [%s]" % os.path.basename(p))
            for s in sims:
                n_all += 1
                dp = firemap.get(id(s))
                unex = unexplained_supplies(s.get("messages") or [], kinds, names)
                n_unex_sims += 1 if unex else 0
                tail = ("  원천-미실행 공급 %d건" % unex) if unex else ""
                if dp is None:
                    print("    %-14s -%s" % (F.sim_key(s), tail))
                    continue
                print("    %-14s [%d] %s %s='%s'  %s%s"
                      % (F.sim_key(s), dp["i_dp"], dp["tool"], dp["ev"]["key"], dp["ev"]["value"][:28],
                         fmt_ev(dp["ev"]), tail))
        print("  발화 %d / %d sim · 원천-미실행 공급 보유 sim %d" % (len(fired), n_all, n_unex_sims))
        print("  ⚠재현율 단서([[70]]·리뷰 MAJOR⑤): 미발화 sim 중 일부는 **선언 공백** 탓일 수 있다 — "
              "대안-나열 키를 '마지막 원소=원천' 으로 소비하고, email 계열 원천에 by_email 선언이 "
              "없다(관찰·[[24]] 후속). census 8/40 은 정밀도 주장이지 재현율 주장이 아니다.")

    # ── 결정점 복원 — 기본: census 발화 sim 전부(대상 선정도 술어·리뷰 MINOR④) ────
    picked = []
    if a.cases:
        for spec in [x.strip() for x in a.cases.split(",") if x.strip()]:
            suf, _, tr = spec.partition(":")
            trial = int(tr) if tr else None
            sim = None
            for p, sims in sims_by.items():
                try:
                    sim = pick(sims, suf, trial)
                    break
                except SystemExit:
                    continue
            if sim is None:
                raise SystemExit("case %s 를 어느 결과 파일에서도 못 찾았다" % spec)
            dp = find_dp(sim.get("messages") or [], kinds, names)
            if dp is None:
                print(NLC + "⚠ %s: 술어가 서는 호출이 없다 — 이 sim 은 이 프로브의 대상이 아니다"
                      % F.sim_key(sim))
                continue
            picked.append((sim, dp))
    else:
        picked = [(s, dp) for (_p, s, dp) in fired]
        print(NLC + "[대상] census 발화 sim 전부 = %d건 (%s) — --cases 로만 덮어쓸 수 있다"
              % (len(picked), ", ".join(F.sim_key(s) for s, _ in picked)))

    universe = declared_read_universe(kinds)
    fb_rows = {}
    for _p in sims_by:
        try:
            for r in F.sidecar_rows(F.tag_of_file(_p)):
                fb_rows.setdefault(str(r.get("simtag") or "?"), []).append(r)
        except Exception:
            pass
    print(NLC + "[충실도] 비커밋 사이드카 행 %d (simtag %d) · turn-조인 신뢰도 %s · 역할 보존 %s"
          % (sum(len(v) for v in fb_rows.values()), len(fb_rows), FB_TURN_JOIN, FB_ROLE_KNOWN)
          + ("" if fb_rows else "  ⚠로컬엔 없다 — 리모트(`/home/woori/scratch/logs`)에서만 계량된다"))

    cases = []
    for sim, dp in picked:
        msgs = sim.get("messages") or []
        ctx = cut_context(msgs, dp["i_dp"], a.cut)
        if a.cut == "before" and a.channel == "user" and ctx \
                and str(ctx[-1].get("role") or "") == "user" and not a.merge_user:
            raise SystemExit("%s: --cut before 의 문맥 끝이 user 다 — user-user 연속은 라이브 교대 "
                             "구조에 없다. --merge-user 를 켜라(리뷰 MAJOR②)" % F.sim_key(sim))
        fb = feedback(dp["ev"], dp["tool"], kinds, agent, exposed, disc, a.cut)
        assert_fb_grounded(ctx, dp, a.cut)          # ★닫힌 자기검사(리뷰 BLOCK①) — 실패 시 중단
        chain = kinds[dp["ev"]["key"]]
        shams = sham_names(kinds, dp["ev"]["key"], agent, exposed, disc, muts, len(chain))
        fb_sham = None
        if len(shams) == len(chain):
            fb_sham = feedback(dp["ev"], dp["tool"], dict(kinds, **{dp["ev"]["key"]: shams}),
                               agent, exposed, disc, a.cut)
        lv = live_next_calls(msgs, len(ctx))
        lv_cat = classify(lv or [], ctx, dp, kinds, names, disc)[0] if lv is not None else None
        # ★재생 충실도 두 축(2026-08-22) — 둘 다 **계량만** 한다. 복원하지 않는다.
        tailnames = named_reads_in_tail(ctx, dp["i_dp"], universe, chain)
        nc = noncommitted_census(fb_rows, F.simtag(sim), len(ctx))
        cases.append({"sim": sim, "dp": dp, "ctx": ctx, "fb": fb, "shams": shams,
                      "fb_sham": fb_sham, "live_cat": lv_cat,
                      "tailnames": tailnames, "noncommitted": nc})
        print(NLC + "── %s · msgs=%d · 결정점=[%d] %s%s · cut=%s/%s → 문맥 %d msg (끝 role=%s) · "
              "라이브 다음 행동=%s"
              % (F.sim_key(sim), len(msgs), dp["i_dp"], dp["tool"],
                 "" if dp["outer"] == dp["tool"] else " (via %s)" % dp["outer"], a.cut, a.channel,
                 len(ctx), ctx[-1].get("role") if ctx else "-", lv_cat))
        print("   오투입: %s='%s'  증거: %s" % (dp["ev"]["key"], dp["ev"]["value"][:40], fmt_ev(dp["ev"])))
        print("   지목 read(선언 순서·마지막=원천): %s · sham=%s" % (", ".join(chain), shams or "-"))
        print("   충실도: 컷 꼬리에 이미 나온 선언 read = 지목자신 %s · **경쟁 지목** %s │ "
              "라이브가 더 본 비커밋 문면 %s"
              % (tailnames["pointed_in_tail"], tailnames["competing"] or "없음",
                 ("%d건 %d자 %s" % (nc["n"], nc["chars"], nc["kinds"])) if nc["n"]
                 else ("측정 불가(사이드카 없음·리모트 전용)" if not fb_rows else "0건")))
        print("   B 텍스트(%d자): %s" % (len(fb), fb))
        if a.cut == "before":
            print("   ⚠선제-힌트 보조 팔: 값 선택 자체가 결정점 지식이라 정보-맞춤이 약하다(문면은 "
                  "ⓐ/ⓑ 증거만·'you passed as %s' 0)" % dp["ev"]["key"])

    if not cases:
        raise SystemExit("결정점 0 — 중단")

    # ── 배선 선검증: 샌드박스 + 문맥-도구 실재(리뷰 MAJOR④·2026-08-22 2차 수리) ──────
    # ★수리 이력: 옛 가드는 `need`(문맥 호출 ∪ 지목 read ∪ sham) **전부**가 `env.get_tools()` 에
    #   있기를 요구했다. 1차 수리는 "부여(unlock)를 거치면 스키마에 뜬다" 고 보고 궤적의 부여를
    #   샌드박스에 재생했는데, 그 전제가 **틀렸다** — 리모트 실측: 부여 6/6 성공인데 `get_tools()`
    #   는 17 → 17 로 그대로였다. 정본 축자가 그렇게 말한다: *"이 env 의 발견형 도구는 도구 목록에
    #   서지 않고 **디스패처로만** 불린다"*(`t2_gate_patch.py:2554` · x255 §호출 타입 T2).
    #   ⇒ 라이브 도구 지형 = `env.get_tools()` + 우리 scaffold, **부여와 무관**하게 결정점마다 같다.
    #   그래서 판정을 두 축으로 나눈다:
    #     hard        = 래퍼 ∪ 문맥이 **직접**(래퍼 없이) 호출한 도구  → 호출 가능 스키마에 없으면 중단
    #     dispatcher  = 그 밖의 참조 이름(발견형·지목 read·sham·래퍼 안쪽) → **존재만** 확인하고 인쇄
    #   존재는 툴킷 레지스트리에 묻는다(`env_registry_names`). 레지스트리에도 없으면 그때가 진짜
    #   낯선 도구라 **중단**이다(가드는 살아 있다).
    tools, sb = [], None
    for c in cases:
        c["unlocked"] = unlocked_before_cut(c["ctx"])
        c["direct"] = called_directly(c["ctx"])
        c["wrapped"] = set()
        for m in c["ctx"]:
            if str(m.get("role") or "") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if F.nameof(tc) in F.WRAPPERS and F.inner_name(F.argsof(tc)):
                    c["wrapped"].add(F.inner_name(F.argsof(tc)))
    hard = set(DISC_TOOLS) | {n for c in cases for n in c["direct"]}
    dispatcher = ({c["dp"]["tool"] for c in cases}
                  | {p for c in cases for p in kinds[c["dp"]["ev"]["key"]]}
                  | {s for c in cases for s in c["shams"]}
                  | {n for c in cases for n in c["wrapped"]}
                  | {n for c in cases for n in c["unlocked"]}) - hard
    # ★로컬에서도 도는 닫힌 불변식(2026-08-22 신설·리모트 실패를 여기서 잡는다):
    #   발견형은 디스패처로만 불리므로 **hard 에 발견형이 들어가면 그 자체가 배선 결함**이다.
    hard_disc = sorted(hard & disc)
    if hard_disc:
        raise SystemExit("hard 가드에 **발견형** 도구가 들어갔다 — 이 env 의 발견형은 도구 목록에 "
                         "서지 않고 디스패처로만 불린다(`t2_gate_patch.py:2554`·x255 T2). 배선 결함: %s"
                         % hard_disc)
    print(NLC + "[배선] 도구 판정 축: hard %d(래퍼+직접호출) · dispatcher-only %d(발견형·지목·sham) · "
          "hard ∩ 발견형 = 0 ✓" % (len(hard), len(dispatcher)))
    try:
        import x448_index_vs_all_iso as IVA
        sb = IVA.Sandbox()
        env_tools, sg_tools = list(sb.env.get_tools() or []), scaffold_tools(a3)
        tools = env_tools + sg_tools
        have = {getattr(t, "name", None) for t in tools}
        reg = env_registry_names(sb)
        print("[배선] 샌드박스 호출 가능 %d종(env %d + scaffold %d) · 툴킷 레지스트리 %d종"
              % (len(tools), len(env_tools), len(sg_tools), len(reg)))
        miss = sorted(hard - have)
        if miss:
            fam = {n: sorted(h for h in have if CH._fam(h) == CH._fam(n)) for n in miss}
            raise SystemExit("문맥이 **직접 호출한** 도구가 재생 스키마에 없다(모델에게 낯선 문맥·"
                             "리뷰 MAJOR④): %s · 접미사 계열 이웃 %s" % (miss, fam))
        if not reg:
            print("  ⚠툴킷 레지스트리를 못 읽었다 — dispatcher-only 존재 확인 생략([[55]] 침묵 금지)")
        else:
            ghost = sorted(n for n in dispatcher if name_axis(n, have, reg) is None)
            if ghost:
                raise SystemExit("레지스트리에도 없는 이름을 문맥/문면이 참조한다(진짜 낯선 도구·"
                                 "가드 유지): %s" % ghost)
        print("  dispatcher-only(존재 ✓·호출은 %s 경유 — 라이브에서도 스키마 밖이 정상): %s"
              % (F.CALLA, sorted(dispatcher)))
        for c in cases:
            c["tools"] = tools          # 라이브와 같다 — 결정점마다 같은 지형(부여와 무관)
            print("  %-14s 재생 스키마 %d종 · 이 sim 이 컷 전 연 발견형 %d종(서명만 본 상태): %s"
                  % (F.sim_key(c["sim"]), len(tools), len(c["unlocked"]),
                     ", ".join(sorted(c["unlocked"])) or "-"))
    except SystemExit:
        raise
    except Exception as e:
        print("[배선] 샌드박스 없음(%r) — 레지스트리(env_surface)로만 검사(로컬 wiring 모드·"
              "리모트 샌드박스가 강제 검사를 다시 한다)" % (e,))
        sg = {d.get("name") for d in (a3.get("scaffold_get_tools") or [])}
        # 로컬 모사: **호출 가능** = 노출 ∪ scaffold ∪ 래퍼 (발견형은 여기 없다 = 리모트와 같은 모양)
        callable_local = exposed | sg | set(F.WRAPPERS)
        unseen = sorted(n for n in hard if n not in callable_local and n not in agent)
        if unseen:
            print("  ⚠env_surface 에 아예 없는 이름(하네스/KB 도구 — 리모트가 실재를 확인한다): %s"
                  % unseen)
        outside = sorted(n for n in hard if n in agent and n not in callable_local)
        if outside:
            raise SystemExit("hard 가 로컬 모사 호출-가능 집합 밖이다(리모트에서 그대로 죽는다): %s"
                             % outside)
        ghost = sorted(n for n in dispatcher if name_axis(n, callable_local, agent | sg) is None)
        if ghost:
            raise SystemExit("레지스트리 선언에도 없는 이름을 참조한다(진짜 낯선 도구): %s" % ghost)
        print("  dispatcher-only(선언 실재 ✓·호출은 %s 경유): %s" % (F.CALLA, sorted(dispatcher)))
        for c in cases:
            c["tools"] = []
            print("  %-14s 이 sim 이 컷 전 연 발견형 %d종: %s"
                  % (F.sim_key(c["sim"]), len(c["unlocked"]),
                     ", ".join(sorted(c["unlocked"])) or "-"))

    # ── 팔 구성 — 한 변수(끝 메시지/끝 도구-출력)만 다르다 ────────────────────────
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for c in cases:
        if c["fb_sham"] is None and "S_sham" in arms:
            print("   ⚠%s: sham 이름 부족 — 이 결정점의 S_sham 팔 생략" % F.sim_key(c["sim"]))
        built = {"A_asis": c["ctx"],
                 "B_pointed": place(c["ctx"], c["fb"], a.channel, a.merge_user),
                 "S_sham": (place(c["ctx"], c["fb_sham"], a.channel, a.merge_user)
                            if c["fb_sham"] else None),
                 "N_neg": place(c["ctx"], NUDGE, a.channel, a.merge_user)}
        c["arm_ctx"] = {arm: built[arm] for arm in arms if built.get(arm) is not None}
        base_n = sum(len(str(m.get("content") or "")) for m in c["ctx"])
        c["delta"] = {arm: sum(len(str(m.get("content") or "")) for m in cx) - base_n
                      for arm, cx in c["arm_ctx"].items()}
        # 팔 문맥의 닫힌 자기검사: A 와의 차이가 정확히 한 자리(채널이 선언한 자리)뿐인가
        for arm, cx in c["arm_ctx"].items():
            if arm == "A_asis":
                continue
            if a.channel == "tool":
                diffs = [i for i in range(len(c["ctx"]))
                         if len(cx) == len(c["ctx"]) and cx[i] != c["ctx"][i]]
                ok = (len(cx) == len(c["ctx"]) and len(diffs) == 1
                      and str(cx[diffs[0]].get("role") or "") == "tool"
                      and str(cx[diffs[0]].get("content") or "")
                      .startswith(str(c["ctx"][diffs[0]].get("content") or "")))
            else:
                merged = a.merge_user and c["ctx"] and str(c["ctx"][-1].get("role") or "") == "user"
                ok = (len(cx) == len(c["ctx"]) and cx[:-1] == c["ctx"][:-1]) if merged \
                    else (len(cx) == len(c["ctx"]) + 1 and cx[:-1] == c["ctx"])
            if not ok:
                raise SystemExit("팔 %s 의 문맥이 A 와 선언된 한 자리 밖에서 다르다 — 배선 결함" % arm)
        print("   [%s] 팔 델타: %s" % (F.sim_key(c["sim"]),
                                       ", ".join("%s=+%d" % kv for kv in sorted(c["delta"].items()))))
        # 채점기 자기검사(LLM 0·[[55]] 계기): 4칸 전부 — 실패는 경고가 아니라 중단이다
        pre = c["sim"]["messages"][:c["dp"]["i_dp"]]
        first = kinds[c["dp"]["ev"]["key"]][-1]
        checks = [("라이브 호출", classify([tuple(c["dp"]["call"])], pre, c["dp"], kinds, names,
                                          disc)[0], "same_misinput"),
                  ("unlock(원천)", classify([(F.UNLOCK, {"agent_tool_name": first})], c["ctx"],
                                           c["dp"], kinds, names, disc)[0], "pointed_read"),
                  ("직접(원천)", classify([(first, {})], c["ctx"], c["dp"], kinds, names, disc)[0],
                   "pointed_read_direct" if first in disc else "pointed_read"),
                  ("새 값", classify([(c["dp"]["tool"],
                                       {c["dp"]["ev"]["key"]: "zz-novel-x466-check"})], pre,
                                     c["dp"], kinds, names, disc)[0], "same_tool_unknown_value")]
        if c["shams"]:
            checks.append(("sham unlock", classify([(F.UNLOCK, {"agent_tool_name": c["shams"][-1]})],
                                                   c["ctx"], c["dp"], kinds, names, disc,
                                                   c["shams"])[0], "sham_read"))
        bad = [(nm, got, want) for nm, got, want in checks if got != want]
        print("   채점기 자기검사: " + " · ".join("%s→%s" % (nm, got) for nm, got, _w in checks))
        if bad:
            raise SystemExit("채점기 자기검사 실패([[55]] 계기 — 결과 사용 금지): %s"
                             % "; ".join("%s=%s(기대 %s)" % b for b in bad))

    # ── 재생 경로 자기검정 (LLM 0·GPU 0 — 실제 루프와 **같은 `shot()`** 을 태운다) ───────
    print(NLC + "[배선] 재생 경로 합성 표본 (gen 주입 → replay·언팩·classify·행 조립·인쇄줄)")
    shot_ok = selftest_shot(cases[0], [], list(cases[0].get("tools") or []), a.model,
                            "http://localhost:%d/v1" % a.port, kinds, names, disc)

    if a.wiring_only:
        print("[배선] wiring-only %s — LLM 0·GPU 0 (재생 경로 포함)"
              % ("PASS" if shot_ok else "FAIL"))
        return 0 if shot_ok else 1
    if not shot_ok:
        raise SystemExit("재생 경로 자기검정 실패 — GPU 를 쓰지 않는다([[55]])")
    if not tools or not all(c.get("tools") for c in cases):
        raise SystemExit("실물 도구 스키마 없이 재생할 수 없다(샌드박스 필요·C584)")

    # ── system (라이브 빌더·기본 ON — 리뷰 MAJOR③) ──────────────────────────────
    # ★system 도 **결정점별** 도구 목록으로 짓는다 — tau2 system 은 도구 지형을 담으므로 공용 목록을
    #   쓰면 아직 안 연 discoverable 이 새어 들어간다(수리 2026-08-22).
    sys_src = "없음(--no-system)"
    for c in cases:
        c["sys_msgs"] = []
    if not a.no_system:
        for c in cases:
            pol = str(c["sim"].get("policy") or "")
            if not pol:
                raise SystemExit("sim['policy'] 가 비어 있어 라이브 system 을 만들 수 없다 — "
                                 "--no-system 은 진단용일 뿐 기본 유효 조건이 아니다")
            c["sys_msgs"], sys_src = system_dicts(pol, c["tools"], a.model)
    print(NLC + "[배선] system=%s · %d메시지 · %d바이트(첫 결정점) · T2_RULES_PROMPT=%s"
          % (sys_src, len(cases[0]["sys_msgs"]),
             sum(len(m["content"].encode("utf-8")) for m in cases[0]["sys_msgs"]),
             os.environ.get("T2_RULES_PROMPT") or "-"))

    # ── 재생 — det 1발 + temp n발 / 팔 / 결정점 ───────────────────────────────────
    base = "http://localhost:%d/v1" % a.port
    total = sum(len(c["arm_ctx"]) for c in cases) * (1 + a.n)
    print("[재생] 결정점 %d × 팔 ≤%d × (1+%d) = 호출 %d건" % (len(cases), len(arms), a.n, total))
    rows = []
    run_exc = (None, 0)                      # (예외 타입, 연속 횟수) — 연속 EXC 가드
    for c in cases:
        sk = F.sim_key(c["sim"])
        for arm in arms:
            if arm not in c["arm_ctx"]:
                continue
            cx = c["arm_ctx"][arm]
            print(NLC + "── %s · %s (+%d자) ────────────────────────────" % (sk, arm, c["delta"][arm]))
            for k, t in enumerate([0.0] + [a.temperature] * a.n):
                try:
                    row, line = shot(c, arm, cx, k, t, c["sys_msgs"], c["tools"], a.model, base,
                                     kinds, names, disc)
                except Exception as e:
                    # [[55]] 예외를 삼키지 않는다 — 표본마다 인쇄하되, **같은 예외가 연속 %d 번**이면
                    # 즉시 중단한다(2026-08-22: 전 표본 EXC 로 GPU 를 태운 사고).
                    print("  #%d t=%.1f EXC %s %r" % (k, t, type(e).__name__, e))
                    rows.append({"sim": sk, "arm": arm, "k": k, "temp": t, "cat": "EXC",
                                 "exc": type(e).__name__, "err": repr(e)[:200]})
                    run_exc = (type(e).__name__, (run_exc[1] + 1) if run_exc[0] == type(e).__name__
                               else 1)
                    if run_exc[1] >= EXC_STREAK:
                        raise SystemExit("같은 예외 %s 가 연속 %d 표본 — 계기 결함이다. 재생을 "
                                         "중단한다([[55]]·GPU 낭비 금지)" % run_exc)
                    continue
                run_exc = (None, 0)
                rows.append(row)
                print(line)

    # ── 집계 — 결정점 × 팔 × 닫힌 분류 + 유효성(리뷰 MAJOR③) + [[70]] 병기 ────────
    cats_all = list(ORDER) + ["no_tool", "EXC"]
    print(NLC + "=" * 100)
    print("분류 축약: " + " · ".join("%s=%s" % (SHORT[c], c) for c in cats_all))
    print("%-16s %-10s %s  (n)" % ("결정점", "팔", " ".join("%-13s" % SHORT[c] for c in cats_all)))
    validity = {}
    for c in cases:
        sk = F.sim_key(c["sim"])
        a0 = next((r["cat"] for r in rows if r["sim"] == sk and r["arm"] == "A_asis" and r["k"] == 0),
                  None)
        valid = (a0 is not None and a0 == c["live_cat"] and c["live_cat"] != "pointed_read")
        validity[sk] = {"A0": a0, "live": c["live_cat"], "valid": valid}
        mark = "" if valid else ("  ⚠라이브 재현 실패(A0=%s ≠ live=%s) — 아래 §충실도가 이유다"
                                 % (a0, c["live_cat"]))
        print("%s%s" % (sk, mark))
        for arm in arms:
            rs = [r for r in rows if r["sim"] == sk and r["arm"] == arm]
            if not rs:
                continue
            print("%-16s %-10s %s  %d" % ("", arm, " ".join(
                "%-13s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == cc), len(rs))) for cc in cats_all),
                len(rs)))
    # ── §충실도 — 라이브 재현이 왜 실패하는가(계량·2026-08-22) ────────────────────
    comp = [F.sim_key(c["sim"]) for c in cases if c["tailnames"]["competing"]]
    ncsum = sum(c["noncommitted"]["n"] for c in cases)
    print(NLC + "§충실도 (라이브 프롬프트 = system + 커밋 히스토리 + **비커밋 레버 문면**)")
    print("  ⑴ 경쟁 지목 보유 %d/%d 결정점 %s — 그 자리에서 A_asis 는 **무지목 통제가 아니다**. "
          "컷 꼬리에 이미 우리 층/환경이 다른 read 이름을 대고 있고, 라이브의 다음 행동은 그것을 "
          "따른 것이다 ⇒ B 는 *지목 vs 무지목* 이 아니라 **경쟁 지목**을 잰다([[56]])."
          % (len(comp), len(cases), comp or "-"))
    print("  ⑵ 비커밋 문면: 그 턴에 조인된 사이드카 %d건%s. 정확 복원은 불가하다 — `turn` 이 기록 "
          "지점마다 커밋 수/버퍼 길이라 생성 경계 조인이 %s, 역할 보존이 %s(실측)."
          % (ncsum, "" if ncsum else "(로컬엔 사이드카가 없다)", FB_TURN_JOIN, FB_ROLE_KNOWN))
    print("  ⇒ **판정: 이 프로브로 라이브 결정점을 재현할 수 없다.** A/B/S/N 비교는 *재구성된 문맥 "
          "안에서만* 유효한 격리 결과이고([[62]] 경계 판정), 라이브에서 그렇게 됐을 것이라는 주장에는 "
          "쓸 수 없다. 승격은 본런 reward A/B 만이 정한다([[69]]).")

    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"cut": a.cut, "channel": a.channel, "system": sys_src,
                   "fidelity": {"live_replay": "불가", "competing_pointer_cases": comp,
                                "noncommitted_rows_joined": ncsum,
                                "fb_turn_join": FB_TURN_JOIN, "fb_role_known": FB_ROLE_KNOWN},
                   "validity": validity,
                   "cases": [{"sim": F.sim_key(c["sim"]), "i_dp": c["dp"]["i_dp"],
                              "tool": c["dp"]["tool"], "ev": c["dp"]["ev"],
                              "live_cat": c["live_cat"],
                              "valid": validity[F.sim_key(c["sim"])]["valid"],
                              "feedback": c["fb"], "shams": c["shams"],
                              "tailnames": c["tailnames"], "noncommitted": c["noncommitted"],
                              "delta": c["delta"]} for c in cases],
                   "rows": rows}, f, ensure_ascii=False, indent=1, default=str)
    print(NLC + "판정: A det 가 라이브 다음-행동을 재현하지 못하면 그 결정점의 **라이브 주장**은 죽는다.")
    print("      2026-08-22 실측으로 그 원인은 배관이 아니라 **재현 불가**로 확정됐다(위 §충실도) —")
    print("      복원할 수 없는 것을 복원한 척하지 않는다([[55]]). 남는 것은 격리-내부 비교뿐이다.")
    print("      B 에서 pointed_read ∧ N≈A ∧ S≈A 면 **원인은 무지목**(레버=이름 지목만 산다·[[57]]).")
    print("      B 도 same_misinput 이면 **지목해도 안 부른다** — 전달로 안 닫히는 경계다([[62]]②).")
    print("[[70]] 병기: 결정점 × 팔 부호표는 위 표 그대로(태스크별 부호 공개). B 가 판 것 = 문맥 "
          "+N자(위 델타)·호출마다 닫힌 술어 1회·과발화/재현율-공백 위험(--census 로 계수). "
          "성적 확정은 본런 reward 재실행에서만([[69]]).")
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
