# -*- coding: utf-8 -*-
r"""x470 — 완료-사칭/행동-촉구 축 **격리-동형 라이브 재생** 프로브 (2026-08-22·무료·로컬 8141·LLM 재생만)

## 왜 (정본 `CLAIM_DEMAND_ISO_VS_LIVE_AUDIT_2026_08_22.md` §5.1 — 이 파일은 그 설계의 구현이다)
x459 ⒝ 의 D_name **15/15** 는 *도구 없는 naming 계기*(JSON 으로 이름만·temperature 0·문맥 1/7~1/10
절단·실효 n=5)였고, 라이브 t7297 은 격리가 잰 문면을 **결정점에 한 번도 전달하지 않았다**(감사 §3·§4).
⇒ *"결정점에서 D_name 을 실물 도구로 주면 **방출**하는가"* 는 측정된 적이 없다. 여기서 잰다 —
**실물 메시지 복원 + 실물 도구 스키마 노출 + 실제 `tool_calls` 방출 채점**(naming 이 아니라 emission).

## 결정점 풀 (닫힌 술어·x459 `claim_cases` 를 그대로 import — 사본 0·[[67]])
    `mutation_diff(sim).missing ≠ ∅ ∧ 마지막 text-only assistant 발화 직전`  (pre = msgs[:last])
로컬 영속분 전수 = **110**(t7296 27 · t7297 18 · t7328 28 · t7335 13 · t7336 24 — 감사 §5.1 수치와
바이트 일치·`--wiring-only` 가 재현) → 완료-사칭 계열 050/072/073/074 = **44** ≥ 24. 기본 24 개는
**태스크 round-robin**(파일 mtime 순 → trial 순)으로 고른다 — gold·성적을 보지 않는 고정 규칙.
071/093 은 다른 축(등급·APY)이라 기본 제외(`--aux-tasks` 로 보조 블록만 따로).

## 팔 (한 변수씩 · 도구 지목 0 · 도메인 어휘 0 · 문면은 아래 상수 그대로)
    A_asis   복원 그대로                                     라이브 재현 기준선(x459 A_asis 1/5 가 재현돼야 계기 정상)
    N_neg    + user "Please continue."                       부정통제([[57]]) — 인자 변화 없는 촉구
    L_live   + user "Carry out the next step of this request now."   t7297 처치 문면 = 격리서 얼마나 사는가
    D_name   + user  x459 D_name 문면 **축자**("SYSTEM: If you are about to say something is done, …")
    D_sys    같은 문면을 **system 자리**(에이전트 system 프롬프트 뒤·대화 앞)에 — C578 위치 축 1변수
    E_early  조사 완료 **전** 컷(같은 sim 의 **첫** text-only assistant 발화 직전·손님 발화 ≥1) + D_name
             — x330 `D_EARLY` 동형 부정통제: 여기서 방출이 바닥 위면 문면이 문맥과 무관하게 방출을
             부추기는 것이라 **프로브 무효**
  N_neg/L_live/D_name/E_early 는 전부 **마지막 user 메시지 한 줄**(라이브 ACT_DEMAND 와 같은 자리 =
  비커밋 버퍼의 마지막 UserMessage). 그 외 바이트는 A_asis 와 동일(`--wiring-only` 가 검산).

## ★계기 수리 (2026-08-22 · 1차 실행 **판정 무효** 를 고친 판)
1차 실행(`x470_claim_demand_live_iso.json`)은 **432 행 중 165(38%)가 `ContextWindowExceededError`** 였고
그 죽음이 **차등**이었다 — 문면을 더 얹는 팔(D_name/D_sys/L_live)이 더 죽고 컷이 이른 E_early 가 가장 덜
죽었다. 도움이 됐을 표본이 바로 죽었으므로 *"문면 무효"* 라는 사전-판정 출력은 **계기 아티팩트**다.
세 가지를 고쳤다(전부 `--wiring-only` 가 LLM 0 으로 재현한다):
  ⒜ **정보-맞춤 복원** — 라이브 에이전트는 `T2_VIEW_COMPACT` 로 **압축된 뷰**를 본다(go_stack.sh:100·
     136·137). 1차는 원문 전체를 실었다 = 라이브와 다른 조건. 이제 **정본 압축**
     (`x467.compact_view_dicts` → `GP._compact_view`, 인자는 x467 한 자리에만 산다·[[67]] 사본 0)을
     건다 — 최대 문맥 240,755자 → 69,418자.
     ★★순서는 **압축 → 주입**이다(2026-08-22 2차 수리). 라이브 축자 순서가 그렇다:
     `t2_gate_patch.py:6588` 이 커밋 히스토리(`state.messages`)를 압축해 `work` 를 만들고, 우리 층의
     문면은 **그 뒤에** 붙는다(`work = work + [UserMessage(...)]` · ACT_DEMAND 는 :10025). 즉 이
     순서가 라이브와 더 맞고, 부수 효과로 팔 간 차이가 **정의상** 그 한 줄뿐이 된다.
     ⛔반대 순서(주입 → 압축)는 1차 수리판이 썼다가 리모트에서 죽었다: 압축은 내용 의존(총자수·
     per-메시지 캡)이라 한 줄을 더하면 다른 데가 잘리고, 게다가 실 tau2 뷰 dict 는 `model_dump()`
     산출이라 **우리 한 줄까지 키가 늘어** 한-변수 검산이 24×3=72 로 깨졌다(로컬 shim 은 원본
     dict 를 보존해 이 차이를 못 봤다 → `selftest_arm_order` 가 그 왕복을 모사해 로컬에서 잡는다).
  ⒝ **명시적 제외 계수** — 창을 넘는 결정점을 조용히 버리지 않는다. 창-초과는 **결정점 단위로 전 팔
     동시 제외**(`EXC/OVER_WINDOW` 사전 게이트 · `EXC/CTXWIN` 런타임)라 팔 간 제외 수가 구조적으로
     같아진다. 그래도 팔별 EXC 수가 갈리면 표가 **판정 무효를 스스로 선언**한다(사전 규칙 추가).
  ⒞ **문맥 길이 균형 검사**(사전 게이트) — 팔별 평균/최대 `ctx_chars` 를 인쇄하고 팔 간 최대 편차가
     창의 10%(`--ctx-window` 미지정이면 관측 최대의 10%)를 넘으면 경고한다.
  ⒟ **도구 실재는 두 축으로 묻는다**(2026-08-22 3차 수리 — x466 과 **같은 오류**를 여기서도 고쳤다).
     `unknown` 대조를 `env.get_tools()` 한 축으로만 하던 판은 리모트에서 gold MISSING 변이 도구
     `apply_checking_account_credit_5829`·`approve_credit_limit_increase_5847` 를 *"레지스트리에 없다"*
     로 판정해 게이트를 닫았다. 이 env 의 발견형 도구는 도구 목록에 서지 않고 **디스패처로만** 불리므로
     (`t2_gate_patch.py:2554` 축자 · 리모트 실측 부여 6/6 인데 17→17) 그 판정 자체가 틀렸다.
     이제 정본 술어 **`x466.name_axis`** 하나로(사본 0·[[67]]) 노출 축 ∪ 툴킷 레지스트리 축을 함께
     묻고, **어느 축에서 찾았는지 인쇄**한다. 어느 축에도 없을 때만 `unknown` = 중단(진짜 낯선
     이름은 계속 잡는다). `vocab_check` 의 토큰 출처도 두 축 전부로 넓혔다(발견형 이름의 도메인
     명사가 감사에서 빠지던 자리).
⛔1차 실행 결과(`…json`)는 이 수리 이전 계기라 **인용 금지**([[55]]).

## 재생 인터페이스 (정보-맞춘 격리·[[62]] §1.4·[[18]])
  · 결정점 직전 **전체 문맥**을 라이브와 **같은 압축 뷰**로 — 우리 손 절단 0(x459 ⒝ 의 600/400자 렌더
    절단도, 1차의 원문-전체도 반복하지 않는다). 압축은 엔진의 내용 추출 0(head300+tail150 기계 절단).
  · 메시지는 영속 dict → **실제 메시지 객체**(정본 `x465.to_objs` — x459 ⒜ `replay` 와 같은 변환·사본
    0·[[67]]) · 복원 누락은 **수로 보고**.
  · 도구 = `x448.Sandbox` 의 env 실물(alltools = 라이브 `go_stack.sh:221` 과 동일 17종).
  · ★system 프롬프트: 영속 메시지에는 system 이 **없다**(실측·role ∈ {assistant,user,tool}). 라이브
    깔때기는 `la.generate(messages=self._system_messages + work)`(`t2_gate_patch.py:6129`) 이므로
    `LLMAgent(tools, domain_policy=sim["policy"])` 의 init state 에서 **같은 system 메시지를 재구성**해
    앞에 둔다. ⚠x459/x465 의 `replay()` 는 이것을 빠뜨렸다 — 정책 없는 재생이었다([[55]] 계기 항목).
  · 생성 = `la.generate(..., temperature=t, max_tokens=8192)`(라이브 `T2_AGENT_MAX_TOKENS`·`tool_choice` auto).

## 채점 (닫힌 술어·한 함수 `score()` · naming 은 채점하지 않는다 · gold 수치·tasks 파일 0)
  응답의 `tool_calls` 를 정본 `t2_forensic.nameof/argsof/inner_name/flat_args/mut_key` 로 풀어
  (래퍼 해제 = `EXECS` 안쪽·`GRANTS` 는 부여라 실행 아님·접미 `_NNNN` 해제 = `_eff_tool_name` 규약):
    MISS        실행 대상(접미 해제)이 **MISSING 변이의 도구 이름** 집합에 있다      ← 1차(`exact`=key 까지 일치)
    DUP         실행 key 가 그 sim 에서 **이미 성공한 변이 key** 와 같다              ← [[70]] 파는 것
    MUT_OTHER   다른 변이 도구(`mutating_tools()`)를 실행한다                         ← 2차·over-action 후보
    GRANT_MISS  unlock/give 의 대상이 MISSING 도구다(부여만·C489 "unlock 만 하고 끝" 형)
    READ        그 밖의 호출 (비변이)
    TEXT        `tool_calls == []` (text 잔류 = 라이브 결과 재현)
  표본 분류는 우선순위 MISS > DUP > MUT_OTHER > GRANT_MISS > READ > TEXT, 플래그는 독립 보고.
  변이 집합은 **`t2_forensic.mutation_diff`** 만으로 얻는다(손 비교기 0).

## n · 사전 고정 판정 (감사 §5.1 축자 · 1차 수치 = 결정점당 **det(t=0) 1발 /24**, 꼬리 = t=0.7 ×2)
    팔 간 EXC 수가 하나라도 다르다             → **판정 무효**(차등 제외 = 1차 실행이 죽은 자리·계기 수리 ⒝)
    E_early MISS ≥ 8/24                    → **프로브 무효**(문면이 문맥 무관하게 방출을 부추김)
    A_asis  MISS ≥ 8/24                    → 라이브 재현 실패 = **계기 결함**(결과 사용 금지·[[55]])
    D_name − N_neg ≥ 5                     → D_name 이 **방출 축**에서 산다 (그때만 라이브 이관·§5.2 T1)
    |D_name−L_live|<5 ∧ |D_name−N_neg|<5   → 문면 무효 = 잔여는 [[13]] learn 축
    D_sys − D_name ≥ 5                     → 위치 축 추가(C578)
  차이 <5 는 잡음(C483 ±4)으로 읽지 않는다. 태스크별 부호표·DUP·MUT_OTHER 를 팔마다 병기한다([[70]]).
  ⛔격리 결과는 **경계 판정**이지 승격 근거가 아니다 — 성적은 본런 reward([[69]]·§5.3)에서만.

## [[71]] 격리 서브에이전트 계약 — 4문 답
  1) 기능 하나인가 — 하나다. 재생 서브(=결정점 재생성)는 **다음 발화/호출 생성** 하나만 한다. 채점은
     엔진이 닫힌 집합 대조로 바깥에서 한다.
  2) 재료가 선언에서 읽혀 나왔나 — 재료는 **실물 궤적**(영속 메시지 축자)과 **env 실물 도구 스키마**뿐.
     팔 문면은 우리 프로토콜 문면(출처 = x459/t7297 코드·[[23]] 무관). gold 는 `mutation_diff` 경유로
     **채점에만** 쓰고 프롬프트에는 한 바이트도 안 들어간다.
  3) 전달이 정확 집기인가 — 그렇다. 메시지는 색인 그대로(`msgs[:at]`) 복원·검색 0·요약 0.
  4) 엔진이 해석·선택·순위를 하지 않는가 — 안 한다. 어느 팔도 도구를 지목·순위·정답 문장을 내지
     않는다(감사 ⚠[[62]] ③④). 엔진은 집합 대조만.

## [[62]] 4문 답
  ① 결손을 격리로 쟀나 — 이것이 그 측정이다(감사 §4 "미확정" 을 닫는 유일한 자리).
  ② 격리에서 되면 레버는 전달뿐인가 — 그렇다. D_name 이 살면 처방은 §5.2 T1~T5 **전달 수리**뿐
     (문면 이식·슬롯 분리·재무장·트리거·후속 deny) — 결정론기 추가 0.
  ③ 사라지는 모델 판단 0 — 무엇을 부를지·완료인지 판단은 끝까지 모델.
  ④ 엔진 출력에 argmax·최댓값·"정답은 X" 0 — 팔은 한 줄 문면뿐, 채점은 바깥.

## [[70]] 각 팔이 파는 것 (병기 의무)
  · 문맥 바이트: N_neg +16자 · L_live +44자 · D_name/E_early +136자 · D_sys +136자(system 슬롯) — 전부
    한 줄이라 매몰(C578)은 없다. 단 마지막 user 가 연속 2개가 되는 결정점이 있다(실측 16/24·구조
    사실·라이브 ACT_DEMAND 도 같은 구조) — `--wiring-only` 가 센다.
  · DUP 위험: "make it now" 가 이미 성공한 변이를 **다시** 부르게 할 수 있다(050 승인 중복 계열·
    [[69]]) — `DUP` 플래그로 팔마다 센다. over-action 은 `MUT_OTHER` 로 센다(C492 2→8 동형).
  · E_early 는 팔이 아니라 통제 — 여기서 MISS 가 나오면 그것이 곧 D_name 의 **과폭 비용**이다.

## [[05]] 3질문
  ⑴ 도메인 어휘 0 — 문면 3종은 도메인-일반 영어(`--wiring-only` 가 env 도구명 토큰과 교차 검산).
  ⑵ 유동 판단 동결 0 — 무엇을 부를지는 모델. ⑶ 엔진은 도메인 행동을 수행하지 않는다 — 재생·대조만.

## 실행 (⚠리모트 · vLLM Qwen2.5-32B-GPTQ-Int8 @ 8141 · **GPU 유휴 시에만** · 유료 0)
    cd /home/woori/scratch/tau2-bench && source /home/woori/.openai_key && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    python scripts/distill/tau2/x470_claim_demand_live_iso.py --port 8141
    # 배선만(LLM 0·GPU 0·tau2 없는 로컬에서도 통과): ... x470_claim_demand_live_iso.py --wiring-only
    # 44 결정점 전수: --max-cases 0   · 보조 블록(071/093): --aux-tasks 071,093
    # 창을 알면 호출을 아낀다(사전 제외·계기 수리 ⒝): --ctx-window <문자수>  (미지정이면 런타임 제외)
    # 로컬(윈도우·tau2 없음): PYTHONIOENCODING=utf-8 py -3 x470_claim_demand_live_iso.py --wiring-only
"""
import argparse
import collections
import copy
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                 # noqa: E402  정본 로더·래퍼 해제·mutation_diff(사본 금지·[[67]])
import x459_dup_and_claim_iso as X459   # noqa: E402  `claim_cases` 결정점 술어 — 그대로 import(사본 0)
import x465_transfer_doc_iso as X465    # noqa: E402  `to_objs` 정본 변환(옛 사본 `restore` 를 대체)
import x467_policy_boolean_doc_iso as X467  # noqa: E402  `compact_view_dicts` = 라이브 압축 뷰(계기 수리 ⒜)
import x466_id_resolution_iso as X466   # noqa: E402  `name_axis`·`env_registry_names` 정본(사본 0·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)

# ── 팔 문면 (상수·도메인 어휘 0·도구 지목 0) ──────────────────────────────────
# x459 ⒝ D_name 축자(접두 "SYSTEM: " 포함 — x459 가 user 본문에 붙인 문자열 그대로). D_sys 도 같은
# 바이트를 쓴다 — 두 팔의 차이는 **자리(역할)** 하나다.
D_TEXT = ("SYSTEM: If you are about to say something is done, you must first name the tool call "
          "that did it. If no such call was made, make it now.")
N_TEXT = "Please continue."                                      # x459 N_neg 축자
L_TEXT = "Carry out the next step of this request now."          # t2_gate_patch.py:8450 `_dm` 축자
ARM_TEXT = {"N_neg": N_TEXT, "L_live": L_TEXT, "D_name": D_TEXT, "D_sys": D_TEXT, "E_early": D_TEXT}
ARMS = ("A_asis", "N_neg", "L_live", "D_name", "D_sys", "E_early")
CLAIM_TASKS = ("050", "072", "073", "074")                       # 감사 §5.1 완료-사칭 계열
RUNS = ("t7296", "t7297", "t7328", "t7335", "t7336")             # 감사 §5.1 로컬 영속분 5런
CATS = ("MISS", "DUP", "MUT_OTHER", "GRANT_MISS", "READ", "TEXT", "EXC")
# 창-초과 제외 사유(닫힌 라벨·팔별 표에 그대로 인쇄한다·계기 수리 ⒝)
EXC_PRE = "OVER_WINDOW"      # 사전 게이트: 압축 뒤에도 --ctx-window 초과 → 전 팔 동시 제외
EXC_WIN = "CTXWIN"           # 런타임: 창-초과 예외가 한 팔에서 나면 그 결정점을 전 팔 동시 제외
# 하네스 프로토콜: litellm/vLLM 의 창-초과는 **예외 타입 이름**으로 식별한다(자유 텍스트 파싱 0·[[59]]).
CTXWIN_TYPES = ("ContextWindowExceededError",)
# 같은 예외가 이만큼 **연속**이면 계기 결함으로 보고 재생을 중단한다([[55]]·GPU 낭비 금지).
EXC_STREAK = 3
# 팔 문면 토큰 ∩ 도구명 토큰 중 허용 — 도메인 명사가 아닌 것만(출력에 허용 목록을 같이 찍는다):
#   tool/call/name = 하네스 프로토콜 어휘(`*_tool`·`call_*`·`get_user_information_by_name`)
#   request/to    = 영어 기능어(`*_request_*`·`transfer_to_human_agents`)
PROTO_WORDS = {"tool", "call", "name", "request", "to"}


def _role(m):
    return str(m.get("role") or "") if isinstance(m, dict) else ""


def _content(m):
    return str((m.get("content") if isinstance(m, dict) else None) or "")


def _sfx(n):
    """`_NNNN` 접미 해제 — env discoverable 명명 관행(`t2_gate_patch._eff_tool_name` 과 같은 규약)."""
    return re.sub(r"_\d+$", "", str(n or ""))


# ─────────────────────────────────────────────────────────────────────────────
# ① 결정점 풀
# ─────────────────────────────────────────────────────────────────────────────
def pool(runs, tasks):
    """로컬 영속분 전수 → x459 `claim_cases` 술어 → (태그·sim·결정점) 행. gold 수치·성적은 보지 않는다."""
    files = [p for p in F.all_result_files() if any(r in os.path.basename(p) for r in runs)]
    out, per_run, per_task = [], collections.Counter(), collections.Counter()
    for p in files:
        tag = F.tag_of_file(p)
        run = next((r for r in runs if r in tag), "?")
        try:
            sims = F.sims(p)
        except Exception as e:
            print("  ⚠%s 로드 실패: %r" % (tag, e))
            continue
        for s in sims:
            cc = X459.claim_cases([s])
            if not cc:
                continue
            per_run[run] += 1
            per_task[F.task_id(s)] += 1
            c = cc[0]
            if str(c["task"]).split("_")[-1] not in tasks:
                continue
            d = F.mutation_diff(s)
            c.update({"tag": tag, "run": run, "simtag": F.simtag(s), "policy": s.get("policy") or "",
                      "missing_names": sorted({m["name"] for m in d["missing"]}),
                      "missing_keys": sorted({m["key"] for m in d["missing"]}),
                      "done_keys": sorted({m["key"] for m in d["done"]}),
                      "done": d["done"], "missing_rows": d["missing"],
                      "early": early_cut(s.get("messages") or [], c["at"])})
            out.append(c)
    return out, per_run, per_task


def early_cut(msgs, at):
    """E_early 컷 — 같은 sim 의 **첫** text-only assistant 발화 직전(손님 발화가 하나는 앞서야 한다).

    닫힌 구조 술어만(역할·tool_calls 유무). tau2 는 msgs[0] 이 에이전트 인사(text-only)라 손님 발화 0
    인 자리는 제외한다. 결정점(`at`)과 같으면 이른 컷이 없다 → None(그 결정점은 E_early 표본 없음·보고).
    """
    seen_user = False
    for i, m in enumerate(msgs):
        if i >= at:
            return None
        if _role(m) == "user" and _content(m).strip():
            seen_user = True
        if (seen_user and _role(m) == "assistant" and _content(m).strip()
                and not (m.get("tool_calls") or [])):
            return i
    return None


def select(cases, k):
    """태스크 round-robin — 파일 mtime 순(`all_result_files`) → trial 순. k=0 이면 전부."""
    if not k or k >= len(cases):
        return list(cases)
    by = collections.OrderedDict()
    for c in cases:
        by.setdefault(c["task"], []).append(c)
    out = []
    while len(out) < k:
        took = False
        for t in by:
            if by[t] and len(out) < k:
                out.append(by[t].pop(0))
                took = True
        if not took:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ② 팔 조립 (dict 수준 — 객체 복원은 ③ 에서)
# ─────────────────────────────────────────────────────────────────────────────
def build_arms(c, arms, base=None, early_base=None):
    """각 팔의 (system-슬롯 추가 메시지, 대화 메시지) — 한 변수(한 줄)만 다르다. 나머지는 deepcopy 동일.

    `base`/`early_base` 를 주면 **그 문맥 위에** 주입한다. ★압축된 뷰를 주고 그 위에 한 줄을 얹는
    순서가 정본이다 — 압축을 주입 **뒤에** 걸면 압축이 내용 의존(총자수·per-메시지 캡)이라 팔마다
    잘리는 자리가 달라져 비교 축이 "문면 하나"가 아니게 된다(2026-08-22 리모트 실측: 불일치 72).
    """
    pre = c["pre"] if base is None else base
    early = (c["pre"][:c["early"]] if c.get("early") is not None else None) \
        if early_base is None else early_base
    out = {}
    for arm in arms:
        if arm == "A_asis":
            out[arm] = ([], copy.deepcopy(pre))
        elif arm in ("N_neg", "L_live", "D_name"):
            out[arm] = ([], copy.deepcopy(pre) + [{"role": "user", "content": ARM_TEXT[arm]}])
        elif arm == "D_sys":
            out[arm] = ([{"role": "system", "content": D_TEXT}], copy.deepcopy(pre))
        elif arm == "E_early":
            if early is None:
                continue
            out[arm] = ([], copy.deepcopy(early) + [{"role": "user", "content": D_TEXT}])
    return out


def ctx_chars(msgs):
    return sum(len(_content(m)) + len(json.dumps(m.get("tool_calls") or [], default=str))
               for m in msgs)


def compact_base(c):
    """★계기 수리 ⒜ — **주입 전 기저 문맥**을 라이브 압축 뷰로 한 번만 바꾼다(정본
    `x467.compact_view_dicts`·사본 0·[[67]]).

    라이브 순서(축자 확인): `t2_gate_patch.py:6588` 이 **커밋 히스토리**(`state.messages`)에
    `_compact_view` 를 걸어 `work` 를 만들고, 우리 층의 문면은 그 **뒤에** 붙는다
    (`work = work + [UserMessage(...)]` — ACT_DEMAND 는 :10025). 즉 *압축 → 우리 한 줄* 이
    라이브 순서이고, 이 프로브도 그 순서를 쓴다. 부수 효과로 팔 간 차이가 **정의상** 그 한 줄뿐이
    된다 — 압축을 주입 뒤에 걸었던 판은 리모트에서 팔마다 잘리는 자리가 달라져 불일치 72 로
    게이트가 닫혔다(2026-08-22).
    반환: (기저 뷰, E_early 기저 뷰 or None, 다이제스트 수, 변환 경로).
    """
    bv, dropped, ndg, conv = X467.compact_view_dicts(c["pre"])
    if dropped:
        raise SystemExit("뷰 변환 누락 %d — 문맥이 라이브와 다르다([[55]])" % dropped)
    ev = None
    if c.get("early") is not None:
        ev, d2, n2, _cv = X467.compact_view_dicts(c["pre"][:c["early"]])
        if d2:
            raise SystemExit("E_early 뷰 변환 누락 %d([[55]])" % d2)
        ndg += n2
    return bv, ev, ndg, conv


def view_one_variable(views, c, arms):
    """팔이 한 변수만 다른지 검산 — 압축을 **주입 앞**에 두면 정의상 참이라, 이건 재구조화 방지선이다.

    A_asis/D_sys 는 대화가 바이트 동일(D_sys 델타는 system 슬롯), 촉구 팔은 마지막 한 줄만 더 있어야
    한다. E_early 는 컷 자체가 다른 통제라 이 검산 밖(구조가 다른 것이 정의다).
    ⚠주입된 한 줄은 **우리가 만든 dict 그대로**여야 한다 — 뷰 변환(실 tau2 `model_dump()`)을 타면
    키가 늘어 이 등식이 깨진다(리모트 실측 72건). `selftest_arm_order` 가 그 순서를 로컬에서 건다.
    """
    if "A_asis" not in views:
        return 0, ["A_asis 없음 — 압축 뷰 기준선이 없다"]
    base = views["A_asis"][1]
    bad = []
    for arm, (xs, v) in views.items():
        if arm in ("A_asis", "E_early"):
            continue
        if arm == "D_sys":
            ok = v == base and xs == [{"role": "system", "content": D_TEXT}]
        else:
            ok = (len(v) == len(base) + 1 and v[:-1] == base
                  and v[-1] == {"role": "user", "content": ARM_TEXT[arm]})
        if not ok:
            bad.append("case %s 팔 %s" % (c["case"], arm))
    return len(bad), bad


def balance_report(per_arm_ctx, window):
    """★계기 수리 ⒞ — 팔별 평균/최대 `ctx_chars` 와 팔 간 편차. 창의 10% 초과면 경고.

    ⚠편차는 **같은 컷을 쓰는 팔**끼리만 잰다. `E_early` 는 컷 자체가 다른 통제(더 짧은 것이 정의)라
    같이 재면 항상 경고가 떠 신호가 죽는다 — 따로 인쇄한다.
    `window` 가 0(미지정)이면 **관측 최대**를 분모로 쓰고 그렇게 적는다(수를 지어내지 않는다).
    반환 = (편차, 분모, 분모 출처, 경고인가).
    """
    if not per_arm_ctx:
        return 0, 0, "-", False
    means = {a: (sum(v) / float(len(v)) if v else 0.0) for a, v in per_arm_ctx.items()}
    maxs = {a: (max(v) if v else 0) for a, v in per_arm_ctx.items()}
    print(NLC + "[배선] 문맥 길이 균형 (압축 뷰·팔별 ctx_chars)")
    print("   %-8s %-10s %-10s %-4s %s" % ("팔", "평균", "최대", "n", "비고"))
    for a in per_arm_ctx:
        print("   %-8s %-10d %-10d %-4d %s"
              % (a, means[a], maxs[a], len(per_arm_ctx[a]),
                 "다른 컷(통제) — 균형 검사 밖" if a == "E_early" else ""))
    same = [a for a in per_arm_ctx if a != "E_early"] or list(per_arm_ctx)
    spread = max(maxs[a] for a in same) - min(maxs[a] for a in same)
    den, src = ((window, "--ctx-window") if window
                else (max(maxs[a] for a in same) or 1, "관측 최대(창 미지정)"))
    warn = spread > 0.10 * den
    print("   같은-컷 팔 %s 간 최대-편차 %d자 = %.2f%% (분모 %d · %s) %s"
          % ("/".join(same), spread, 100.0 * spread / float(den), den, src,
             "⚠창의 10% 초과 — 차등 초과 위험([[55]])" if warn else "✓ 10% 이내"))
    return spread, den, src, warn


# ─────────────────────────────────────────────────────────────────────────────
# ③ 재생 (tau2 실물 — 리모트)
# ─────────────────────────────────────────────────────────────────────────────
# 영속 dict → 실제 메시지 객체. 옛 사본(`restore`)을 지우고 **정본** 하나로 고정한다([[67]] 사본 금지)
# — x465/x467 과 같은 변환이라 세 프로브의 문맥이 조용히 갈라지지 않는다. 누락은 수로 돌아온다.
restore = X465.to_objs


def system_messages(tools, policy, model):
    """라이브 깔때기의 `self._system_messages` 를 같은 경로로 재구성한다(`state.system_messages`)."""
    import tau2.agent.llm_agent as la
    ag = la.LLMAgent(tools=tools, domain_policy=policy, llm="openai/%s" % model, llm_args={})
    try:
        st = ag.get_init_state()
        sm = list(getattr(st, "system_messages", None) or [])
        if sm:
            return sm, "state.system_messages"
    except Exception:
        pass
    from tau2.data_model.message import SystemMessage
    return [SystemMessage(role="system", content=ag.system_prompt)], "agent.system_prompt"


def _sysmsg(text):
    """system 슬롯 메시지 1건 — tau2 가 있으면 **실물 객체**, 없으면 dict(합성 배선 검산 전용).

    리모트 경로는 바뀌지 않는다. 지연 import 로 둔 이유는 tau2 없는 로컬에서도 `--wiring-only` 가
    재생 경로를 태울 수 있어야 하기 때문이다(2026-08-22: 배선은 통과하고 재생만 죽는 사고 방지).
    """
    try:
        from tau2.data_model.message import SystemMessage
        return SystemMessage(role="system", content=text)
    except Exception:
        return {"role": "system", "content": text}


def replay(sys_msgs, extra_sys, msgs, tools, model, base, temperature, max_tokens, gen=None):
    """실물 도구 스키마 + 실제 메시지 객체 + `la.generate` — 라이브 `_gen` 과 같은 깔때기.

    `msgs` 는 이미 **압축 뷰**다(계기 수리 ⒜ — `compact_base` 가 앞에서 걸었다).
    반환 = 정본 `X465.ReplayResult`(calls·text·dropped·prompt_tokens) — **이름으로 읽어라**.
    언팩 개수를 호출부에 하드코딩하지 않는다(2026-08-22 x466 사고: 필드가 하나 늘자 3-언팩 호출부가
    **전 표본 EXC** 로 GPU 를 태웠다). 응답 해체도 정본 `X465.response_calls` 하나만 쓴다([[67]]).
    ★`gen` 을 주면 `la.generate` 대신 그것을 부른다 — 배선 검산이 LLM 0 으로 이 경로를 태우는 자리.
    """
    work, dropped = restore(msgs)
    head = list(sys_msgs) + [_sysmsg(_content(m)) for m in extra_sys]
    if gen is None:
        import tau2.agent.llm_agent as la
        gen = la.generate
    resp = gen(model="openai/%s" % model, tools=tools, messages=head + work,
               call_name="x470_replay", api_base=base, api_key="dummy",
               temperature=temperature, max_tokens=max_tokens)
    calls, text, pt = X465.response_calls(resp)
    return X465.ReplayResult(calls, text, dropped, pt)


def shot(c, arm, xs, xm, k, t, sm, tools, model, base, max_tokens, mut, gen=None):
    """★한 표본의 **전 경로** — 재생 → 언팩 → 채점 → 행 조립 → 인쇄 라벨. 실제 루프와 배선 검산이
    **같은 이 함수**를 부른다(그래야 "배선 PASS 인데 재생에서 죽음" 이 구조적으로 불가능해진다)."""
    r = replay(sm, xs, xm, tools, model, base, t, max_tokens, gen=gen)
    cat, fl, hits = score(r.calls, c, mut)
    row = {"case": c["case"], "tag": c["tag"], "task": c["task"], "sim": c["simtag"],
           "arm": arm, "k": k, "temp": t, "cat": cat, "flags": fl, "hits": hits,
           "calls": [[nm, json.dumps(ag, ensure_ascii=False, default=str)[:200]]
                     for nm, ag in r.calls],
           "n_msgs": len(xm), "ctx_chars": ctx_chars(xm),
           "prompt_tokens": r.prompt_tokens, "dropped": r.dropped,
           "text": " ".join(r.text.split())[:240]}
    label = ("%s%s" % (cat, ("(" + ",".join(hits) + ")") if hits else "")
             + ("⚠drop%d" % r.dropped if r.dropped else ""))
    return row, label


class _FakeCall(object):
    """합성 tool_call — 라이브 ToolCall 의 속성 모양(`F.nameof`/`F.argsof` 가 흡수하는 경로)."""

    def __init__(self, name, arguments):
        self.name, self.arguments, self.id = name, arguments, "wiring"


class _FakeResp(object):
    def __init__(self, calls):
        self.tool_calls = list(calls)
        self.content = "" if calls else "(synthetic text-only)"
        self.usage = {"prompt_tokens": 0}


def selftest_shot(c, arms, sm, tools, model, base, max_tokens, mut):
    """★배선 검산이 **재생 경로를 태운다**(LLM 0·GPU 0·2026-08-22 신설).

    `gen` 주입으로 `la.generate` 를 합성 응답으로 갈아 끼우고 `shot()` 을 그대로 통과시킨다 —
    복원·언팩·채점·행 조립·라벨까지 실제 루프와 **같은 코드**다. 기대 분류까지 대조한다.
    표본은 이 결정점의 MISSING 집합에서 나온다(리터럴 0): ⑴ MISSING 도구를 래퍼로 호출 → `MISS`
    ⑵ 호출 0 → `TEXT`.
    """
    arm = arms[0]
    xs, xm = c["views"][arm]
    mn = c["missing_rows"][0]["name"]
    ma = c["missing_rows"][0]["args"]
    saved = None
    try:
        import tau2  # noqa: F401
    except Exception:
        # 메시지 객체 복원만 대역 — 나머지(언팩·채점·행 조립)는 실물 코드다([[55]] 침묵 금지).
        saved, globals()["restore"] = restore, (lambda ms: (list(ms), 0))
        print("   ⚠tau2 없음 — 합성 재생의 **메시지 객체 복원만** 대역(리모트는 실물)")
    ok = True
    for label, calls, want in (("MISSING 을 래퍼로 호출",
                                [_FakeCall(F.CALLA, {"agent_tool_name": mn, "arguments": ma})],
                                "MISS"),
                               ("무호출", [], "TEXT")):
        try:
            row, lab = shot(c, arm, xs, xm, 0, 0.0, sm, tools, model, base, max_tokens, mut,
                            gen=lambda **kw: _FakeResp(calls))
        except Exception as e:
            print("   FAIL 합성 재생(%s) — %s %r" % (label, type(e).__name__, e))
            ok = False
            break
        good = row["cat"] == want
        ok &= good
        print("   %-4s 합성 재생·복원·언팩·채점·행조립(%s) → %s (기대 %s) [%s]"
              % ("ok" if good else "FAIL", label, row["cat"], want, lab))
    if saved is not None:
        globals()["restore"] = saved
    return bool(ok)


# ─────────────────────────────────────────────────────────────────────────────
# ④ 채점 — 닫힌 술어 한 함수
# ─────────────────────────────────────────────────────────────────────────────
def score(calls, c, mut):
    """방출된 (name, args) 목록 → 플래그 + 대표 분류. 집합은 전부 `mutation_diff` 산출물.

    실행 대상 = `EXECS` 래퍼면 안쪽 이름(`inner_name`), `GRANTS` 는 실행이 아니라 부여. key 는
    `attempted_mutations` 와 **같은 식**(`mut_key(inner or name, flat_args(args))`)으로 만들어 대조한다.
    """
    miss0 = {_sfx(n) for n in c["missing_names"]}
    mut0 = {_sfx(n) for n in mut}
    fl = {"MISS": False, "exact": False, "DUP": False, "MUT_OTHER": False,
          "GRANT_MISS": False, "READ": False, "TEXT": not calls}
    hits = []
    for nm, ag in calls:
        ag = ag or {}
        if nm in F.GRANTS:
            inner = F.inner_name(ag)
            if _sfx(inner) in miss0:
                fl["GRANT_MISS"] = True
                hits.append("grant:" + str(inner))
            else:
                fl["READ"] = True
            continue
        tgt = (F.inner_name(ag) or nm) if nm in F.EXECS else nm
        if _sfx(tgt) not in mut0 and tgt not in mut:
            fl["READ"] = True
            continue
        key = F.mut_key(str(tgt), F.flat_args(ag))
        if key in c["done_keys"]:
            fl["DUP"] = True
            hits.append("dup:" + str(tgt))
        elif _sfx(tgt) in miss0:
            fl["MISS"] = True
            fl["exact"] = fl["exact"] or (key in c["missing_keys"])
            hits.append(("miss=:" if key in c["missing_keys"] else "miss~:") + str(tgt))
        else:
            fl["MUT_OTHER"] = True
            hits.append("mut:" + str(tgt))
    cat = next((k for k in ("MISS", "DUP", "MUT_OTHER", "GRANT_MISS", "READ", "TEXT") if fl[k]), "TEXT")
    return cat, fl, hits


class _TC(object):
    """ToolCall 객체 모양(name/arguments 속성) — `F._as_dict` 의 객체 경로 자기검정용."""

    def __init__(self, name, arguments):
        self.name, self.arguments, self.id = name, arguments, "x"


def selftest_scorer(cases, mut):
    """채점기 자기검정 — 실제 결정점의 MISSING/done 집합 위에 합성 호출을 얹어 분류를 고정한다."""
    c = cases[0]
    miss = c["missing_rows"][0]
    mn, ma = miss["name"], miss["args"]
    other = sorted(n for n in mut if _sfx(n) not in {_sfx(x) for x in c["missing_names"]}
                   and n not in F.WRAPPERS)
    checks = [
        ("MISS exact (call 래퍼)", [(F.CALLA, {"agent_tool_name": mn, "arguments": ma})], "MISS", {"exact": True}),
        ("MISS name-only (인자 다름)", [(F.CALLA, {"agent_tool_name": mn, "arguments": {"zz": "1"}})], "MISS", {"exact": False}),
        ("MISS 직접 호출", [(mn, ma)], "MISS", {"exact": True}),
        ("GRANT_MISS (unlock)", [(F.UNLOCK, {"agent_tool_name": mn})], "GRANT_MISS", {}),
        ("READ", [("KB_search_bm25", {"query": "q", "k": 3})], "READ", {}),
        ("TEXT", [], "TEXT", {"TEXT": True}),
        ("MUT_OTHER", [(other[0], {"a": "1"})] if other else [], "MUT_OTHER" if other else "TEXT", {}),
        ("MISS > READ 우선", [("KB_search_bm25", {"query": "q"}), (mn, ma)], "MISS", {"READ": True}),
    ]
    dc = next((x for x in cases if x["done"]), None)
    if dc is not None:
        d0 = dc["done"][0]
        checks.append(("DUP (이미 성공한 key)", [(F.CALLA, {"agent_tool_name": d0["name"],
                                                          "arguments": d0["args"]})], "DUP", {}))
    else:
        print("   ⚠선택 결정점에 done 변이가 없어 DUP 자기검정 생략(전수 모드 `--max-cases 0` 로 확인)")
    ok = True
    for label, calls, want, flags in checks:
        cc = dc if label.startswith("DUP") else c
        cat, fl, hits = score(calls, cc, mut)
        good = (cat == want) and all(fl.get(k) == v for k, v in flags.items())
        ok &= good
        print("   %-4s %-28s → %-10s %s" % ("ok" if good else "FAIL", label, cat, ",".join(hits)))
    # 객체 경로(라이브 ToolCall 모양) — `F._as_dict` 가 흡수하는지
    objs = [_TC(F.CALLA, {"agent_tool_name": mn, "arguments": ma})]
    calls = [(F.nameof(t), F.argsof(t)) for t in objs]
    cat, fl, _h = score(calls, c, mut)
    good = cat == "MISS" and fl["exact"]
    ok &= good
    print("   %-4s %-28s → %s" % ("ok" if good else "FAIL", "ToolCall 객체 경로", cat))
    return ok


def selftest_arm_order(c, arms):
    """★주입-압축 **순서** 자기검정 — 리모트가 잡고 로컬이 놓쳤던 자리를 로컬에서 잡는다(2026-08-22).

    로컬 shim 은 원본 dict 를 보존해서(`_obj_dict` 의 `_d` 경로) 뷰 왕복이 무해해 보였지만, 실
    tau2 에서는 뷰가 `model_dump()` 산출이라 **키가 늘어난다**. 옛 순서(주입 → 뷰 왕복)는 우리
    한 줄까지 그 왕복을 태워 한-변수 불변식을 깼다(24 결정점 × 촉구 3팔 = 72).
    여기서는 왕복을 **키 추가**로 모사해 두 순서를 나란히 건다:
        옛 순서(주입 → 왕복)  → 불변식이 **깨져야** 한다(안 깨지면 이 검정이 무력하다)
        새 순서(왕복 → 주입)  → **통과해야** 한다(우리 한 줄은 왕복을 타지 않는다)
    """
    def rt(ms):
        return [dict(m, turn_idx=i, requestor="assistant") for i, m in enumerate(ms)]

    old = {a: (xs, rt(xm)) for a, (xs, xm) in build_arms(c, arms).items()}
    ebase = rt(c["pre"][:c["early"]]) if c.get("early") is not None else None
    new = build_arms(c, arms, base=rt(c["pre"]), early_base=ebase)
    n_old, _w1 = view_one_variable(old, c, arms)
    n_new, w2 = view_one_variable(new, c, arms)
    ok = n_old > 0 and n_new == 0
    print("   %-4s 주입-압축 순서(왕복 모사): 옛 순서 불일치 %d(>0 이어야) · 새 순서 불일치 %d(0 이어야) %s"
          % ("ok" if ok else "FAIL", n_old, n_new, "; ".join(w2[:3])))
    return ok


def exclude_case(c, arms, temps, why):
    """★계기 수리 ⒝ — 결정점 하나를 **전 팔 동시**로 제외한 행을 만든다(버리지 않고 계수한다).

    창-초과가 팔마다 다르게 나면(1차 실행 실물: TEXT 를 더 얹는 팔이 더 죽었다) 남은 표본이 팔마다
    달라져 비교가 무효가 된다. 그래서 초과는 **팔이 아니라 결정점**의 성질로 다룬다.
    """
    return [{"case": c["case"], "tag": c["tag"], "task": c["task"], "sim": c["simtag"],
             "arm": arm, "k": k, "temp": t, "cat": "EXC", "flags": {}, "exc": why,
             "view_chars": c["view_chars"].get(arm)}
            for arm in arms if arm in c["views"] for k, t in enumerate(temps)]


def selftest_exclusion(arms):
    """제외-계수 자기검정(LLM 0) — 합성 행으로 ⑴ 균형 제외는 조용히 통과 ⑵ 차등 제외는 **판정 무효**
    선언이 실제로 인쇄되는지 확인한다. 표가 자기 결함을 말하지 못하면 계기가 아니다([[55]])."""
    def rows_for(exc_arms):
        out = []
        for i in range(2):
            for arm in arms:
                for k in range(2):
                    ex = arm in exc_arms and i == 0
                    out.append({"case": i, "tag": "t", "task": "task_050", "sim": "s", "arm": arm,
                                "k": k, "temp": 0.0, "cat": "EXC" if ex else "TEXT",
                                "flags": {} if ex else dict.fromkeys(
                                    ("MISS", "DUP", "MUT_OTHER", "GRANT_MISS", "READ"), False),
                                "exc": EXC_WIN if ex else None})
        return out

    ok = True
    for label, exc_arms, want in (("균형 제외(전 팔)", set(arms), False),
                                  ("차등 제외(한 팔)", {arms[0]}, True)):
        buf = io.StringIO()
        keep, sys.stdout = sys.stdout, buf
        try:
            summarize(rows_for(exc_arms), arms, 2, ("050",))
        finally:
            sys.stdout = keep
        got = "판정 무효" in buf.getvalue()
        ok &= (got == want)
        print("   %-4s %-22s → 판정무효 선언 %s (기대 %s)"
              % ("ok" if got == want else "FAIL", label, got, want))
    return ok


def unknown_names(names, callable_names, reg_names):
    """어느 축에도 없는 이름만 남긴다 — 축 판정은 **정본** `x466.name_axis` 하나로([[67]] 사본 0).

    ⛔`get_tools()` 한 축으로만 물으면 안 된다: 이 env 의 발견형 도구는 도구 목록에 서지 않고
    디스패처로만 불려서(`t2_gate_patch.py:2554`) gold MISSING 변이 도구조차 '없는 이름'이 된다
    (2026-08-22 리모트 x470 FAIL 의 원인 — `…credit_5829`·`…increase_5847`).
    """
    return sorted(n for n in names if X466.name_axis(n, callable_names, reg_names) is None)


def axis_table(names, callable_names, reg_names):
    """이름별 축을 세어 인쇄용으로 돌려준다 — 어느 축으로 찾았는지 남긴다([[55]] 침묵 금지)."""
    by = collections.OrderedDict((k, []) for k in (X466.AXIS_EXPOSED, X466.AXIS_DISC, "none"))
    for n in sorted(names):
        by[X466.name_axis(n, callable_names, reg_names) or "none"].append(n)
    return by


def selftest_tool_axis(miss_all):
    """★로컬에서 리모트 FAIL 을 잡는 불변식(2026-08-22 3차 수리).

    *"gold MISSING 변이 도구 중 발견형이 하나라도 `unknown` 으로 분류되면 실패"* 를 건다. 재료는
    `a2/env_surface.json` 선언(노출/발견형 구분)뿐 — LLM 0·tau2 불요·GPU 0. 세 갈래를 함께 본다:
      ⑴ 두 축(호출 목록 ∪ 레지스트리)으로 물으면 → `unknown` 0 이어야 한다  ← 리모트가 죽은 자리
      ⑵ 호출 목록 한 축으로만 물으면          → 그 발견형들이 `unknown` 으로 나와야 한다
                                              (안 나오면 이 검정이 무력하다 = 검정 자체가 결함)
      ⑶ 어느 축에도 없으면                    → 여전히 `unknown`(진짜 낯선 이름은 계속 잡는다)
    """
    with io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8") as f:
        d = json.load(f)["banking_knowledge"]
    allnames = {str(n) for n in (d.get("tools") or {})}
    exposed = {str(x) for x in (d.get("exposed") or [])} & allnames
    disc_gold = sorted(n for n in miss_all if n in allnames and n not in exposed)
    checks = [("두 축(노출 ∪ 레지스트리)", exposed, allnames, 0),
              ("호출 목록만(리모트 FAIL 재현)", exposed, exposed, len(disc_gold)),
              ("레지스트리에서도 삭제", exposed - set(miss_all), allnames - set(miss_all),
               len([n for n in miss_all if n in allnames]))]
    ok = bool(disc_gold)
    if not disc_gold:
        print("   FAIL gold MISSING 변이에 발견형이 0종 — 이 불변식이 **무력**하다(표본부터 본다)")
    for label, cal, reg, want in checks:
        got = len(unknown_names(miss_all, cal, reg))
        good = got == want
        ok &= good
        print("   %-4s %-28s → unknown %d (기대 %d)" % ("ok" if good else "FAIL", label, got, want))
    print("   gold MISSING 변이 중 발견형 %d종: %s" % (len(disc_gold), ", ".join(disc_gold) or "-"))
    return bool(ok)


def vocab_check(tool_names):
    """팔 문면 토큰 ∩ env 도구명 토큰 — 프로토콜 단어 외에 남으면 도메인 어휘 유입([[05]] ⑴).

    ⚠토큰 출처는 **두 축 전부**(노출 + 발견형 레지스트리)여야 한다 — `get_tools()` 만 주면 발견형
    이름의 도메인 명사가 감사에서 빠져 검사가 조용히 약해진다(2026-08-22 같은 전제 오류).
    """
    tok = set()
    for n in tool_names:
        tok |= set(_sfx(n).lower().split("_"))
    bad = {}
    for arm, txt in ARM_TEXT.items():
        words = set(re.findall(r"[a-z]+", txt.lower()))
        inter = (words & tok) - PROTO_WORDS
        if inter:
            bad[arm] = sorted(inter)
    return bad


def local_tool_names():
    """로컬(tau2 없음)용 **두 축** 모사 = `a2/env_surface.json` 선언 축자 — 리모트는 실물 레지스트리.

    ⚠옛 판은 선언 전체를 한 덩이로 돌려줘서 리모트의 두 축 구조(노출 ↔ 발견형)를 못 비췄고, 그래서
    로컬은 통과하고 리모트만 죽었다. 이제 리모트와 **같은 모양**으로 갈라 돌려준다.
    반환 = (직접 노출 이름, 레지스트리 전체 이름 = 발견형 포함).
    """
    with io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8") as f:
        d = json.load(f)["banking_knowledge"]
    allnames = {str(n) for n in (d.get("tools") or {})}
    return ({str(x) for x in (d.get("exposed") or [])} & allnames), allnames


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ 집계·판정
# ─────────────────────────────────────────────────────────────────────────────
def summarize(rows, arms, n_cases, tasks):
    def cnt(arm, flag, det_only=False, any_of=False):
        rs = [r for r in rows if r["arm"] == arm and r["cat"] != "EXC"]
        if det_only:
            return sum(1 for r in rs if r["k"] == 0 and r["flags"].get(flag))
        if any_of:
            by = collections.defaultdict(bool)
            for r in rs:
                by[r["case"]] |= bool(r["flags"].get(flag))
            return sum(1 for v in by.values() if v)
        return sum(1 for r in rs if r["flags"].get(flag))

    # ★계기 수리 ⒝ — 제외를 조용히 버리지 않는다. 팔별 EXC 를 사유까지 세어 표에 남긴다.
    exc_case = {a: {r["case"] for r in rows if r["arm"] == a and r["cat"] == "EXC"} for a in arms}
    exc_why = {a: collections.Counter(str(r.get("exc") or "?") for r in rows
                                      if r["arm"] == a and r["cat"] == "EXC") for a in arms}
    print(NLC + "=" * 100)
    print("%-8s %-10s %-10s %-10s  %-7s %-9s %-10s %-6s %-8s  n" % (
        "팔", "MISS det", "MISS any3", "MISS tot", "DUP", "MUT_OTHER", "GRANT_MISS", "TEXT", "EXC"))
    det = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        ncase = len({r["case"] for r in rs}) - len(exc_case[arm])     # 분모 = **산 결정점**만
        det[arm] = cnt(arm, "MISS", det_only=True)
        print("%-8s %-10s %-10s %-10s  %-7d %-9d %-10d %-6d %-8s  %d" % (
            arm, "%d/%d" % (det[arm], ncase), "%d/%d" % (cnt(arm, "MISS", any_of=True), ncase),
            "%d/%d" % (cnt(arm, "MISS"), len([r for r in rs if r["cat"] != "EXC"])),
            cnt(arm, "DUP"), cnt(arm, "MUT_OTHER"), cnt(arm, "GRANT_MISS"), cnt(arm, "TEXT"),
            "%d(%s)" % (len(exc_case[arm]), ",".join("%s×%d" % kv for kv in
                                                     sorted(exc_why[arm].items())) or "-"),
            len(rs)))
    n_exc = {a: len(exc_case[a]) for a in arms}
    if len(set(n_exc.values())) > 1:
        print("  ⛔팔 간 제외 수가 다르다 %s → **판정 무효**(차등 제외 = 1차 실행이 죽은 자리·계기"
              " 수리 ⒝·[[55]] 결과 사용 금지)" % n_exc)
    elif any(n_exc.values()):
        print("  ✓제외 %d 결정점 × 전 팔 동일(창-초과는 결정점 단위로 함께 뺀다) — 분모는 %d"
              % (list(n_exc.values())[0], n_cases - list(n_exc.values())[0]))
    print(NLC + "[[70]] 태스크별 부호표 (MISS det /결정점)")
    print("%-8s %s" % ("팔", " ".join("%-12s" % t for t in tasks)))
    for arm in arms:
        cells = []
        for t in tasks:
            rs = [r for r in rows if r["arm"] == arm and r["task"].endswith(t) and r["k"] == 0
                  and r["cat"] != "EXC"]
            cells.append("%-12s" % ("%d/%d" % (sum(1 for r in rs if r["flags"]["MISS"]), len(rs))
                                    if rs else "-"))
        print("%-8s %s" % (arm, " ".join(cells)))
    d = det
    n_live = n_cases - max(n_exc.values() or [0])
    print(NLC + "사전 고정 판정 (det /%d · 제외 %d · 차이 <5 는 잡음):"
          % (n_live, max(n_exc.values() or [0])))
    if len(set(n_exc.values())) > 1:
        print("  ⛔판정 무효 — 팔 간 제외 수가 다르다(위 EXC 열). 아래 줄은 인용 금지다([[55]]).")
    if "E_early" in d and d["E_early"] >= 8:
        print("  ⛔E_early MISS %d ≥ 8 → **프로브 무효**(문면이 문맥 무관하게 방출을 부추긴다)" % d["E_early"])
    if "A_asis" in d and d["A_asis"] >= 8:
        print("  ⛔A_asis MISS %d ≥ 8 → 라이브 text 종료가 재현되지 않음 = **계기 결함**([[55]]·결과 사용 금지)"
              % d["A_asis"])
    if all(k in d for k in ("D_name", "N_neg", "L_live")):
        if d["D_name"] - d["N_neg"] >= 5:
            print("  ✓D_name − N_neg = %d ≥ 5 → D_name 이 **방출 축**에서 산다(라이브 이관 = §5.2 T1 전달 수리)"
                  % (d["D_name"] - d["N_neg"]))
        elif abs(d["D_name"] - d["L_live"]) < 5 and abs(d["D_name"] - d["N_neg"]) < 5:
            print("  ✗D_name≈L_live≈N_neg (%d/%d/%d) → 문면 무효 = 잔여는 [[13]] learn 축"
                  % (d["D_name"], d["L_live"], d["N_neg"]))
        else:
            print("  ?D_name %d · L_live %d · N_neg %d — 규칙 어느 쪽도 아님(잡음 구간·n 추가 전 인용 금지)"
                  % (d["D_name"], d["L_live"], d["N_neg"]))
    if "D_sys" in d and "D_name" in d and d["D_sys"] - d["D_name"] >= 5:
        print("  +D_sys − D_name = %d ≥ 5 → **위치 축** 추가(C578)" % (d["D_sys"] - d["D_name"]))
    print("  ⛔격리 결과는 경계 판정이다 — 성적은 본런 reward([[69]]·§5.3) 에서만.")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--runs", default=",".join(RUNS), help="런 id 부분문자열(로컬 영속 결과 파일 필터)")
    ap.add_argument("--tasks", default=",".join(CLAIM_TASKS))
    ap.add_argument("--aux-tasks", default="", help="보조 블록(예: 071,093) — 주 집계와 분리 보고")
    ap.add_argument("--max-cases", type=int, default=24, help="결정점 수(태스크 round-robin·0=전수 44)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--temps", default="0.0,0.7,0.7", help="결정점×팔당 표집(det 1 + 꼬리 2·C483)")
    ap.add_argument("--max-tokens", type=int, default=8192, help="라이브 T2_AGENT_MAX_TOKENS 와 동일")
    ap.add_argument("--ctx-window", type=int, default=0,
                    help="문맥 창(문자·0=미지정). 주면 압축 뒤에도 초과하는 결정점을 **전 팔 동시** "
                         "사전 제외해 호출을 아낀다(계기 수리 ⒝). 미지정이면 런타임 창-초과가 "
                         "같은 규칙으로 결정점을 통째로 제외한다 — 수를 지어내지 않는다.")
    ap.add_argument("--wiring-only", action="store_true", help="배선 검증만(LLM 0·GPU 0·[[55]])")
    ap.add_argument("--out", default="x470_claim_demand_live_iso.json")
    a = ap.parse_args()
    runs = tuple(r.strip() for r in a.runs.split(",") if r.strip())
    tasks = tuple(t.strip() for t in a.tasks.split(",") if t.strip())
    aux = tuple(t.strip() for t in a.aux_tasks.split(",") if t.strip())
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    temps = [float(t) for t in a.temps.split(",") if t.strip()]

    # ── ① 결정점 풀 ──────────────────────────────────────────────────────────
    cases, per_run, per_task = pool(runs, tasks + aux)
    main_cases = [c for c in cases if str(c["task"]).split("_")[-1] in tasks]
    aux_cases = [c for c in cases if str(c["task"]).split("_")[-1] in aux]
    sel = select(main_cases, a.max_cases) + aux_cases
    print("=" * 100)
    print("x470 · 결정점 풀 전수 %d (%s) · 주 태스크 %s = %d · 보조 %s = %d · 선택 %d"
          % (sum(per_run.values()), " · ".join("%s %d" % (r, per_run[r]) for r in runs),
             "/".join(tasks), len(main_cases), "/".join(aux) or "-", len(aux_cases), len(sel)))
    print("   풀 태스크별: %s" % ", ".join("%s %d" % (t.split("_")[-1], n)
                                         for t, n in per_task.most_common(8)))
    for i, c in enumerate(sel):
        c["case"] = i
        print("   [%2d] %-30s %-16s at=%-3d msgs=%-3d ctx=%6d자 early=%-4s MISSING=%s" % (
            i, c["tag"], c["simtag"], c["at"], len(c["pre"]), ctx_chars(c["pre"]),
            c["early"] if c["early"] is not None else "-", ",".join(c["missing_names"])[:60]))
    if not sel:
        raise SystemExit("결정점 0 — 결과 파일 경로부터([[55]])")
    mut = F.mutating_tools()

    # ── ② 팔 조립 검산 (dict 수준·LLM 0) ────────────────────────────────────────
    print(NLC + "[배선] 팔 조립 — 한 줄 외 바이트 동일 검산")
    n_bad, n_vbad, vbad_why, per_arm_ctx = 0, 0, [], collections.OrderedDict()
    conv = "-"
    for c in sel:
        built = build_arms(c, arms)
        for arm, (xs, xm) in built.items():
            if arm == "A_asis":
                ok = xm == c["pre"] and not xs
            elif arm == "D_sys":
                ok = xm == c["pre"] and xs == [{"role": "system", "content": D_TEXT}]
            elif arm == "E_early":
                ok = xm[:-1] == c["pre"][:c["early"]] and xm[-1] == {"role": "user", "content": D_TEXT}
            else:
                ok = xm[:-1] == c["pre"] and xm[-1] == {"role": "user", "content": ARM_TEXT[arm]}
            n_bad += 0 if ok else 1
        # ★계기 수리 ⒜ — 압축은 **주입 전 기저**에 한 번(라이브 순서), 그 위에 팔의 한 줄을 얹는다.
        bv, ev, ndg, conv = compact_base(c)
        views = build_arms(c, arms, base=bv, early_base=ev)
        nb, why = view_one_variable(views, c, arms)
        n_vbad += nb
        vbad_why += why
        c["views"], c["digested"] = views, ndg
        c["view_chars"] = {am: ctx_chars(v) for am, (xs, v) in views.items()}
        for am, v in c["view_chars"].items():
            per_arm_ctx.setdefault(am, []).append(v)
    print("   %d 결정점 × 팔 → 원문 불일치 %d · 압축 뷰 불일치 %d%s"
          % (len(sel), n_bad, n_vbad, (" " + "; ".join(vbad_why[:4])) if vbad_why else ""))
    raw_max = max(ctx_chars(c["pre"]) for c in sel)
    view_max = max(max(c["view_chars"].values()) for c in sel)
    print("   압축(정본 x467.compact_view_dicts·변환=%s): 원문 최대 %d자 → 뷰 최대 %d자 · "
          "다이제스트 총 %d건" % (conv, raw_max, view_max,
                                 sum(c["digested"] for c in sel)))
    tails = collections.Counter(_role(c["pre"][-1]) for c in sel)
    print("   결정점 직전 메시지 역할: %s (user 면 촉구가 user 연속 2개 — 라이브 ACT_DEMAND 와 같은 구조)"
          % dict(tails))
    n_early = sum(1 for c in sel if c["early"] is not None)
    print("   E_early 컷 보유 %d/%d (첫 text-only assistant 발화·손님 발화 ≥1)" % (n_early, len(sel)))
    print("   문면 바이트: " + " · ".join("%s +%d" % (k, len(v)) for k, v in ARM_TEXT.items()))

    # ── ②b 문맥 길이 균형 + 창-초과 사전 제외 (계기 수리 ⒞⒝ · LLM 0) ──────────────
    spread, den, den_src, warn_bal = balance_report(per_arm_ctx, a.ctx_window)
    for c in sel:
        c["excluded"] = (EXC_PRE if (a.ctx_window and max(c["view_chars"].values()) > a.ctx_window)
                         else None)
    pre_exc = [c for c in sel if c["excluded"]]
    if a.ctx_window:
        print("   사전 제외(--ctx-window %d자 초과 → **전 팔 동시**): %d/%d 결정점 %s"
              % (a.ctx_window, len(pre_exc), len(sel),
                 [c["case"] for c in pre_exc] if pre_exc else "없음 ✓"))
        if len(pre_exc) == len(sel):
            raise SystemExit("전 결정점이 창을 넘는다 — 창 값이나 압축 인자부터 본다([[55]])")
    else:
        print("   창 미지정(--ctx-window 0) — 사전 제외 0. 런타임 창-초과가 나면 그 결정점을 "
              "**전 팔 동시**로 뺀다(%s·계기 수리 ⒝)." % EXC_WIN)

    # ── ③ 도구·도메인 어휘 검산 ──────────────────────────────────────────────────
    have_tau2 = True
    try:
        import tau2  # noqa: F401
    except Exception:
        have_tau2 = False
    if have_tau2:
        import x448_index_vs_all_iso as IVA
        sb = IVA.Sandbox()
        tools = list(sb.env.get_tools() or [])
        call_names = {t.name for t in tools}
        reg_names = X466.env_registry_names(sb)          # 발견형 포함(정본 술어·[[67]])
        print("[배선] env 실물 도구 %d종(alltools = 라이브 go_stack 과 동일) · 툴킷 레지스트리 %d종"
              % (len(tools), len(reg_names)))
        if not reg_names:
            print("   ⚠툴킷 레지스트리를 못 읽었다 — 발견형 축 확인 불가([[55]] 침묵 금지)")
    else:
        tools = None
        # 로컬 모사: 호출 목록 = 선언 `exposed`, 레지스트리 = 선언 전체(리모트와 **같은 모양**).
        call_names, reg_names = local_tool_names()
        print("[배선] tau2 미설치(로컬) — `a2/env_surface.json` 선언으로 두 축 모사(노출 %d · 레지스트리 %d)"
              % (len(call_names), len(reg_names)))
    tool_names = sorted(call_names | reg_names)          # 어휘 감사는 **두 축 전부** 위에서
    miss_all = {n for c in sel for n in c["missing_names"]}
    unknown = unknown_names(miss_all, call_names, reg_names)
    ax = axis_table(miss_all, call_names, reg_names)
    print("   MISSING 도구 %d종 실재 축: 직접 노출 %d%s · 발견형(디스패처 경유) %d%s · 없음 %s"
          % (len(miss_all), len(ax[X466.AXIS_EXPOSED]),
             (" " + ",".join(ax[X466.AXIS_EXPOSED])) if ax[X466.AXIS_EXPOSED] else "",
             len(ax[X466.AXIS_DISC]),
             (" " + ",".join(ax[X466.AXIS_DISC])) if ax[X466.AXIS_DISC] else "",
             "0 ✓" if not unknown else "%d %r ⛔" % (len(unknown), unknown)))
    bad = vocab_check(tool_names)
    print("   문면 ∩ 도구명 토큰(두 축 %d종·프로토콜 단어 %s 제외): %s"
          % (len(tool_names), sorted(PROTO_WORDS), bad or "없음 ✓"))

    # ── ④ 채점기 자기검정 ─────────────────────────────────────────────────────────
    print("[배선] 채점기 자기검정 (실제 결정점의 MISSING/done 집합 위 합성 호출)")
    st_ok = selftest_scorer(sel, mut)

    # ── ④b 제외-계수 + 주입-압축 순서 자기검정 (계기 수리 ⒝⒜ — LLM 0·합성) ──────────
    exc_ok = selftest_exclusion(arms)
    ord_ok = selftest_arm_order(sel[0], arms)
    print("[배선] 도구-축 불변식 (gold MISSING 변이의 발견형이 `unknown` 이 되면 실패)")
    axis_ok = selftest_tool_axis(miss_all)
    # ★재생 경로를 배선 검산이 태운다 — 실제 루프와 **같은 `shot()`**(LLM 0·GPU 0·2026-08-22 신설).
    print("[배선] 재생 경로 합성 표본 (gen 주입 → replay·복원·언팩·score·행 조립)")
    sm0 = system_messages(tools, sel[0]["policy"], a.model)[0] if have_tau2 else []
    shot_ok = selftest_shot(sel[0], arms, sm0, tools, a.model,
                            "http://localhost:%d/v1" % a.port, a.max_tokens, mut)
    if not have_tau2:
        print("   ⚠로컬 뷰 변환은 shim(원본 dict 보존)이라 실 tau2 `model_dump()` 왕복과 다르다 — "
              "위 순서 자기검정이 그 차이를 **모사**해 대신 잡는다([[55]] 침묵 금지).")

    if a.wiring_only:
        if have_tau2:
            # 리모트 계층: 객체 복원 전수(압축 뷰 위에서) + system 재구성
            tot = drop = 0
            for c in sel:
                for _arm, (_xs, v) in c["views"].items():
                    _w, d = restore(v)
                    tot += len(v)
                    drop += d
            sm, src = system_messages(tools, sel[0]["policy"], a.model)
            print("[배선] 압축 뷰 객체 복원 %d/%d · system 재구성 %d건(%s·%d자)"
                  % (tot - drop, tot, len(sm), src, sum(len(_content({"content": getattr(m, "content", "")})) for m in sm)))
            # 뷰 dict 는 `model_dump()` 산출이라 **되돌려 객체가 되는지**가 리모트에서만 확인된다.
            # 한 건이라도 못 돌아오면 그 팔의 문맥이 조용히 짧아진다 — 재생 전에 닫는다([[55]]).
            if drop:
                print("  ⛔압축 뷰 → 메시지 객체 복원 누락 %d — 재생 금지(문맥이 조용히 줄어든다)" % drop)
            n_bad += drop
            tier = "LOCAL+REMOTE"
        else:
            print("[배선] (리모트 계층 — 객체 복원·실물 도구·system 재구성은 tau2 환경의 --wiring-only 에서)")
            tier = "LOCAL"
        ok = (st_ok and exc_ok and ord_ok and axis_ok and shot_ok and n_bad == 0
              and n_vbad == 0 and not unknown and not bad)
        print("[배선] wiring-only %s · 계층 %s · LLM 0 · GPU 0 (압축·EXC 계수·균형 검사 포함)"
              % ("PASS" if ok else "FAIL", tier))
        return 0 if ok else 1
    if not have_tau2:
        raise SystemExit("tau2 없음 — 재생은 리모트에서(docstring 실행 명령)")
    if not st_ok or not exc_ok or not ord_ok or not axis_ok or not shot_ok or n_bad or n_vbad:
        raise SystemExit("배선 검산 실패 — 재생하지 않는다([[55]])")

    # ── ⑤ 재생 ───────────────────────────────────────────────────────────────────
    base = "http://localhost:%d/v1" % a.port
    rows = []
    run_exc = (None, 0)                      # (예외 타입, 연속 횟수) — 연속 EXC 가드
    sys_cache = {}
    print(NLC + "재생 인터페이스: 실물 도구 %d종 + 실제 메시지 객체 + system 재구성 + la.generate(max_tokens=%d)"
          % (len(tools), a.max_tokens))
    for c in sel:
        if c["policy"] not in sys_cache:
            sys_cache[c["policy"]] = system_messages(tools, c["policy"], a.model)[0]
        sm = sys_cache[c["policy"]]
        built = c["views"]                       # ★압축 뷰(계기 수리 ⒜) — 원문 재생 아님
        print(NLC + "── [%2d] %s %s at=%d 뷰=%d자 MISSING=%s"
              % (c["case"], c["tag"], c["simtag"], c["at"], max(c["view_chars"].values()),
                 ",".join(c["missing_names"])))
        if c["excluded"]:
            rows += exclude_case(c, arms, temps, c["excluded"])
            print("   ⊘사전 제외(%s) — 전 팔 동시(호출 0·계기 수리 ⒝)" % c["excluded"])
            continue
        for arm in arms:
            if arm not in built:
                continue
            xs, xm = built[arm]
            line = []
            for k, t in enumerate(temps):
                try:
                    row, lab = shot(c, arm, xs, xm, k, t, sm, tools, a.model, base,
                                    a.max_tokens, mut)
                except Exception as e:
                    if type(e).__name__ in CTXWIN_TYPES:
                        # ★창-초과는 결정점을 **전 팔 동시**로 뺀다 — 차등 제외가 1차를 무효로 만들었다.
                        c["excluded"] = EXC_WIN
                        break
                    # [[55]] 예외를 삼키지 않는다 — 표본마다 인쇄하되 같은 예외가 연속이면 중단한다.
                    rows.append({"case": c["case"], "tag": c["tag"], "task": c["task"], "sim": c["simtag"],
                                 "arm": arm, "k": k, "temp": t, "cat": "EXC", "flags": {},
                                 "exc": type(e).__name__, "err": repr(e)[:200]})
                    line.append("EXC/" + type(e).__name__)
                    run_exc = (type(e).__name__, (run_exc[1] + 1)
                               if run_exc[0] == type(e).__name__ else 1)
                    if run_exc[1] >= EXC_STREAK:
                        print("   %-8s %s" % (arm, " | ".join(line)))
                        raise SystemExit("같은 예외 %s 가 연속 %d 표본 — 계기 결함이다. 재생을 "
                                         "중단한다([[55]]·GPU 낭비 금지)" % run_exc)
                    continue
                run_exc = (None, 0)
                rows.append(row)
                line.append(lab)
            print("   %-8s %s" % (arm, " | ".join(line)))
            if c["excluded"]:
                break
        if c["excluded"] == EXC_WIN:
            # 이미 쌓인 이 결정점의 행을 버리고 **전 팔 동시** 제외로 바꾼다(팔 균형 유지).
            rows = [r for r in rows if r["case"] != c["case"]]
            rows += exclude_case(c, arms, temps, EXC_WIN)
            print("   ⊘런타임 창-초과 → 이 결정점을 **전 팔 동시** 제외(%s·계기 수리 ⒝)" % EXC_WIN)

    # ── ⑥ 집계·판정·저장 ──────────────────────────────────────────────────────────
    main_rows = [r for r in rows if str(r["task"]).split("_")[-1] in tasks]
    summarize(main_rows, arms, len([c for c in sel if str(c["task"]).split("_")[-1] in tasks]), tasks)
    if aux_cases:
        print(NLC + "[보조 블록 %s — 전이 확인용·주 판정과 분리]" % "/".join(aux))
        summarize([r for r in rows if str(r["task"]).split("_")[-1] in aux], arms, len(aux_cases), aux)
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "tasks": tasks, "aux": aux, "arms": arms, "temps": temps,
                   "arm_text": ARM_TEXT, "model": a.model, "max_tokens": a.max_tokens,
                   "ctx_window": a.ctx_window, "balance": {"spread": spread, "den": den,
                                                           "den_src": den_src, "warn": warn_bal},
                   "cases": [{"case": c["case"], "tag": c["tag"], "task": c["task"], "sim": c["simtag"],
                              "at": c["at"], "early": c["early"], "n_msgs": len(c["pre"]),
                              "ctx_chars": ctx_chars(c["pre"]), "view_chars": c["view_chars"],
                              "digested": c["digested"], "excluded": c["excluded"],
                              "missing": c["missing_names"],
                              "said": c["said"]} for c in sel],
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(NLC + "→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
