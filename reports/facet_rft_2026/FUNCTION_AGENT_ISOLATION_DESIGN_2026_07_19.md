# 기능별 서브에이전트 격리 설계 (FUNCTION-AGENT ISOLATION · 2026-07-19)

> 계기: 사용자 지시 — *"결정 부분의 중요 부분만 남기고, 기능별 별도 sub agent 기록은 호출 결과만 남기면 되지 않나?
> 기능별로 sub agent 만들어서 서로 정보를 격리하여 context 줄일 수는 없나?"*
> = `RATE_SUBAGENT_DESIGN §2g`(사용자 통찰·KB가 컨텍스트 중앙값 75%)의 **일반형**.
> 앵커: §2b~2e(T2_SG_ISOLATE·021 라이브 1.0 실증) · [[16]] GENERALIZED_SCAFFOLD(GET/FIND/INFER/ASK) ·
> §2d(반증 2건) · §2n(WEV unified 배선·022 CWE 포렌식) · T2_READ_DEDUP(ed95b48f·중복분 레버).

## 0. 문제 정의 (실측·[S])
- 022 = ContextWindowExceeded 3연속(32K→40960서도 41,038). **루프 아님**(7b 로그: RETRY/PROV/CONS deny 전부 0)·
  200스텝 캡 아님 — **페이로드 구조**가 원인.
- 029 anatomy(74msgs·110.6K chars ≈ 35K tok): KB 덤프 53.5K(48%·그중 20K는 byte-identical 중복) ·
  거래덤프 14K · **출력→args 재전송 13.3K**(msg31의 txn 47개 전부가 msg32 args로 재직렬화).
- 중복분은 `T2_READ_DEDUP`(배포됨·opt-in)이 잡는다. **잔여 = 기저 페이로드**: KB 단일 덤프 20K·거래덤프 14K가
  메인 대화에 그대로 눌러앉는 구조. 이것이 본 설계의 대상.

> **★LOCK 확정 (2026-07-19 저녁·사용자 설계 리뷰 반영)**: 6개 핵심 결정(3층 구조·라우팅 2모드·quote grounding·
> 게이트 증거 계약·§2d 제약 내장·측정 계획) 전부 사용자 리뷰로 승인. **P0 구현 완료**: 엔진 `_sub_wrap`
> (t2_scaffold_get.py)+exec2 배선(`T2_FN_ISOLATE=1`·기본 OFF·폴백 보장·per-orch 캐시)+A2 `policy_qa` 1호기+
> 오프라인 4케이스(정상 wrap/환각 quote 폴백/notfound/getter cap 폴백) PASS. [05] 정직 기록: A2 순증 —
> **측정(§5) 통과 전 라이브 활성화 금지**가 게이트.

## 1. 원칙 (LOCK — 사용자 리뷰 확정)
1. **메인 = 대화·흐름·결정만**: 어떤 기능을 언제 쓸지의 판단(유동성)은 메인 모델에 남긴다. 메인 문맥에는
   기능 **호출 1건 + compact 반환 1건**만 쌓인다.
2. **서브 = 한 기능·자기 슬라이스만**: 입력은 A2 화이트리스트(`row_fields` 선례)로 절단. 서브끼리 공유 문맥 0
   (상호 정보 격리). 무거운 자료(KB 문서·전체 레코드)는 서브 문맥에서 소비되고 버려진다.
3. **엔진 = 범용 서브루프 1개**: `_sub_formalize`의 일반화. 기능의 존재·계약·도구셋·반환 스키마는 전부
   A2 `function_agents[]` 데이터([[05]] 엔진 도메인 리터럴 0).
4. **[[10]] 분담선 유지**: 서브 LLM = 해석/formalize(생성기)만. 라우팅(어느 호출을 서브로 보낼지)·getter 실행·
   반환 검증(grounding) = 결정론.
5. **부수 이득**: 격리 = 부하 절감 = §2m 조항-간섭(cue overload) 절감 — Track A 서사("구조적 완화")와 합류.
   격리 단위 축소가 정확도를 올린 실증 3건(§2b operand·§2e 카드·카테고리) 위에 "기능" 축을 추가.

## 2. 아키텍처 (3층 · 라우팅 2모드)
```
메인 에이전트 (대화·흐름·결정)
   │  호출 1 + compact 반환 1만 메인 문맥에 커밋
   ▼
엔진 라우터 (결정론·exec 경로 wrap)          ← A2 function_agents[] 선언 소비
   ├─ (W) wrap 모드: 메인이 부른 기존 도구(KB_search_* 등)를 서브로 위임
   └─ (P) producer 모드: 스카폴드 GET operand를 서브가 산출 (= 기존 T2_SG_ISOLATE·구현 완료)
   ▼
기능 서브 (자체 메시지 리스트만·메인 문맥 미공유)
   getter 호출(A2 선언 도구)→env 결정론 실행→서브 문맥에 되먹임→최종 compact 반환
```
- **(W) wrap 모드가 이번 설계의 신규분**. 메인의 도구 스키마·행동 변화 불요(최소침습): 에이전트가
  `KB_search_bm25(query=...)`를 부르면 엔진이 그 query를 서브에 넘기고, 서브가 검색·정독 후
  **답+근거 원문 인용**만 반환 → 메인에는 그것만 남는다. 20K 덤프가 메인에서 사라진다.
- 기존 producer 모드는 그대로(변경 0). 두 모드가 같은 서브루프 부품(생성·getter 되먹임·trace)을 공유.

### A2 선언 스키마 (banking 1호기 = policy_qa)
```json
"function_agents": [
  {"name": "policy_qa",
   "mode": "wrap",
   "wraps": ["KB_search_bm25", "KB_search_dense", "shell"],
   "getter_tools": ["KB_search_bm25", "KB_search_dense"],
   "instructions": "You answer ONE policy/procedure question against the knowledge base. Search, read, then answer concisely.",
   "return_contract": "Return JSON: {\"answer\": <직접 답>, \"quotes\": [<근거 조항 원문 무편집 발췌 1-3개>], \"found\": true|false}",
   "max_rounds": 4, "max_getter_calls": 6,
   "quote_grounding": true,
   "temperature": 0}
]
```
- `wraps`에 `shell` 포함(banking의 shell=KB 브라우징) — 단 getter로는 KB_search만 제공: 서브가 문서 전문
  `cat` 대신 검색-정독하게. (shell 원호출의 query 추출 불가 시 → 폴백=원 실행.)
- `quote_grounding=true`: 반환 quotes가 **실제 KB 문서의 substring인지 엔진이 결정론 검증**(§2e `_norm_ground`
  재사용) — 불일치 quote는 드롭·전부 드롭이면 폴백. 서브 환각이 메인에 "근거"로 못 들어간다([[03b]]).

## 3. ★게이트 증거 계약 (WEV 상호작용 — 필수 설계점)
- WEV·gate.observe·`_rebuild_gate_state`가 스캔하는 것은 **메인 `state.messages`의 role=tool 내용**이다.
- 서브로 격리된 기능의 출력에 게이트 증거(예: dispute의 `txn_...`+`RESOLVED` 공존)가 있으면, **반환의
  `quotes`(원문 무편집)에 실려 메인에 남아야** 한다. 요약-only 반환은 증거를 소실시켜 WEV가 정당 write를
  오차단한다(028 역회귀 경로).
- 규칙: **wrap 대상은 "자료 read"만**(KB·문서). **상태 read**(dispute status·거래 조회 등 env DB read)는
  wrap 금지 — 증거 원문이 통째로 메인에 남아야 하는 도구들이다. A2 `wraps` 목록이 이 경계를 선언하고,
  리뷰 시 write_evidence_specs와 교차 감사한다.

## 4. 함정 → 제약 (전부 실측 근거)
| 함정 (실측) | 제약 |
|---|---|
| §2d 결함1: 서브가 부하 재생산(75거래 통째→172K) | 서브 입력 cap: `row_fields` 절단·`max_getter_calls`·검색 k cap. 배치형 기능은 `max_batch`(=2 선례) 청킹 |
| §2d 결함2: 서브 환각(온도-분산) | temp 0 + `quote_grounding` 결정론 검증 |
| 021 실패 기전: 봉투 오류로 호출 증발 | 1라운드 `tool_choice=required`(§2c 선례 그대로) |
| 서브 실패/미선언 | **폴백 = 원 도구 그대로 실행**(거동 변화 0·T2_SG_ISOLATE 선례) |
| [[08]] 디버깅 공백(서브는 메인 궤적 밖) | `_isolate_trace` 계열 JSONL 의무(질의·라운드·반환·grounding 드롭 수) |
| 등대 모트: 게이트 자신도 역효과 | **Δspurious 계측 의무**: wrap이 좁혀서 놓친 정보로 실패한 케이스 카운트(아래 §5) |

## 5. 측정 계획 (무료 먼저·[[09]])
1. **무료 오프라인**: 계열 궤적(018~029)의 실제 KB 질의를 추출 → policy_qa 서브 단독 프로브(GPU0 :8140):
   반환 answer가 원 덤프-정독 답과 일치하는지 + quotes grounding 통과율. 기준: 질의 정답률 손실 0.
2. **무료 시뮬레이션**: 029/028 기록 궤적에서 KB 덤프를 wrap-반환(전형 크기 ~1K)으로 치환한 컨텍스트 재계산
   — 예상 절감 확인(029: 53.5K→~3K chars = 총 −46%).
3. **유료 스모크(사용자 승인 후)**: 022 단독(최대 페이로드 태스크·CWE 3연속) — 성공 기준: CWE 0·reward 유지·
   최종 컨텍스트 30%↓. 대조: dedup만 vs dedup+wrap(단일변수).
4. Δspurious: wrap-arm에서 실패한 태스크의 실패 원인 중 "서브 반환에 없던 정보가 필요했다" 분류 카운트
   (0이어야 통과·아니면 return_contract 보강).

## 6. 구현 계획
- **P0 (다음)**: `t2_scaffold_get.py`에 `_sub_wrap()` — `_sub_formalize`서 공용 부품(생성 루프·getter 되먹임·
  trace) 추출 재사용. exec 경로(`exec_augment` 또는 scaffold_get exec2)에서 wraps 매칭 시 위임.
  플래그 `T2_FN_ISOLATE=1`(기본 OFF·단일변수 arm 보존). 오프라인 배선 테스트(`test_sg_isolate.py` 선례).
- **P1**: banking `policy_qa` A2 1호기 → §5 측정 → 022/계열 확대.
- **P2**: [[14]] E-PLAN 합류(discovery read의 서브화)·타 도메인 ABox-swap 전이 확인([[05]]: 엔진 불변·A2만 추가).

## 7. 미결 (이 설계 scope 밖·별도 검토)
- **args 재전송**(13.3K): `get_reward_discrepancies`가 transactions를 param으로 받는 A2 선언 구조 —
  "직전 출력 참조형" 선언은 [[16]] GET 루프와 충돌 여부 검토 후 별도 제안.
- 기능 서브의 **중첩**(서브가 서브를 부름)은 금지(1단·복잡도 통제).
