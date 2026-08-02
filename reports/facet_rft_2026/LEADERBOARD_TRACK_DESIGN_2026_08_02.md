# 리더보드 트랙 설계 — 목표 바·비교 규격·점증 실행 계획 (2026-08-02)

> **사용자 지시(2026-08-02)**: ⑴*"우리는 gpt5.5와 겨뤄야 한다. 단지 custom인 것만 다르다."* ⑵*"특허에서는
> pass^1~^4와 reward에서 gpt-5.5보다 모두 나아야 한다."* ⑶*"반드시 alltools로 가라"*(이전 지시 미이행 교정)
> ⑷*"nt=2 32 task로 구조 만들고 nt=4로 확장하면서 97로 확장하라."*
> **상태 = 규격 확정·계획 사전등록 · 실행은 ax32 아크 종결 후.** 메모리 포인터 = [[54]].
> 규율: 이 문서의 수치는 전부 **1차 출처 직독**(제출 JSON·릴리즈 노트·소스 코드). 재유도 금지.

## 1. 목표 바 — GPT-5.5 (banking_knowledge)

| 지표 | 값 | 출처 |
|---|---|---|
| pass^1 | **46.39** | `submissions/gpt-5-5_sierra_2026-05-05/submission.json` |
| pass^2 | **36.94** | 동일(보드는 "—"로 미표시 — **원본에는 있다**) |
| pass^3 | **31.19** | 동일 |
| pass^4 | **27.84** | 동일 |
| cost | 1.988 | 동일 |
| 설정 | reasoning **xhigh** · agent GPT-5.5 · user-sim gpt-5.2(**reasoning_effort: low**) · **nt=4** · `alltools` · **v1.0.1** · 2026-05-06 | 동일 |
| 범위 | **banking 단독 제출**(airline/retail/telecom null) | 동일 ⇒ **우리도 banking 단독 제출 가능** |

**pass^k 붕괴 = 46.39 → 27.84(−18.6pp)**. pass^1은 모델 능력이 지배하지만 **pass^2~4 붕괴는 일관성·준수
축**이고 그것이 결정론 스캐폴드가 파는 물건이다([[45]] scale-invariant compliance).

### 1b. 경쟁 좌표 — Custom 트랙 유일 항목

| | Distyl ButtonAgent |
|---|---|
| pass^1~^4 | **31.19 / 21.47 / 16.19 / 13.40** |
| 구성 | **GPT-5.4 high reasoning** + 자체 검색(Mixedbread `mxbai-wholembed-v3`·698 문서 벡터스토어) |
| 설정 | 97 전수 · nt=4(seed 300) · user-sim gpt-5.2 · **tau2 v0.2.1-dev** · 프롬프트 미수정 선언 |

읽을 것 셋: ⑴**커스텀 스캐폴드가 반드시 이기지 않는다**(31.19 < Standard GPT-5.2 32.2) ⑵frontier+custom도
pass^4서 **13.40으로 붕괴**(57% 손실) = 우리 강점 축의 공백 ⑶버전이 v0.2.1-dev라 **우리가 1.0.1로 내면
Custom 트랙 최신·최정직 항목**이 된다.

## 2. 비교가능성 조건 (전수 — 하나라도 어기면 인용 불가)

| # | 조건(축자/요지) | 우리 현재(ax32) | 처리 |
|---|---|---|---|
| 1 | **v1.0.1 이상** — *"banking_knowledge scores are not comparable across this release"* | **1.0.0**(HEAD `5ebebbe`) | 업그레이드 |
| 2 | **태스크 필터 금지** — *"No task filtering via `--task-ids` or `--num-tasks`"* · banking 전수 **97** | `--task_ids` 32 | 97로 확장(점증) |
| 3 | **4+ trials** | nt2(pass 체인) | nt4로 확장 |
| 4 | **`retrieval_config` 필수** · 보드 상위 15위 전부 `alltools`/Terminal · **bm25 항목 0개** | `bm25` | **alltools 고정** |
| 5 | **standard = 무수정 기본 스캐폴드** ⇒ 우리는 **Custom** | — | Custom 제출·methodology.notes 필수 |
| 6 | **reasoning effort 보드 표시** + **user-sim reasoning_effort**(GPT-5.5=low) | 미지정 | 러너에 인자 추가 |
| 7 | 전 도메인 권고이나 **단독 제출 실증 있음**(GPT-5.5) | banking 단독 | 그대로 가능 |

**☠검색 설정이 순위를 지배한다(결정 증거)**: 같은 GPT-5.2가 **alltools 32.2% vs qwen_embeddings 12.6%**
= 19.6pp. 참고 좌표 Qwen3.5-397B-A17B = 9.8%(text-emb-3-large).

`alltools` 구성(소스 직독 `retrieval.py`) = **BM25 + dense(OpenAI `text-embedding-3-large`) + shell(읽기전용)**
· 전용 프롬프트 `all_tools.md` · **`DEFAULT_RETRIEVAL_VARIANT`(=기본값)** — 우리가 명시적으로 bm25로
덮어써왔다. 대안 `alltools-qwen`(OpenRouter `qwen3-embedding-8b`). OpenAI 키 `/home/woori/.openai_key` 실재.

## 3. ☠v1.0.1 재채점 — 우리에게 유리한 벤치 버그였다

릴리즈 *τ-bench 1.0.1 — banking_knowledge Grading Fixes*(7/22) 6건 중 **#329가 결정적**:
축자 *"call_discoverable_agent_tool unconditionally appended a row … **including reads** … any extra
defensive read by the agent (e.g. checking eligibility before opening an account, verifying claims before
filing a dispute) diverged the hash and **zeroed the DB reward**"* ⇒ **우리 스캐폴드가 장려하는 검증-읽기가
벌점으로 계상되고 있었다**(WRITE_EVIDENCE 선행-read 강제 계열). 영향권 실측(우리 실패 궤적):
p1 005·026·027, p2 020·026·028 — 모두 `call_discoverable_agent_tool` **읽기** 포함.
그 외: #397 숫자 정규화(25↔25.0 해시) · #403 거래 최신순 · #402 T077-086 gold 실현가능화 ·
#388 Platinum 요율 문서 모순(제목 4% ↔ 본문 10%) · #404 task_074 $8.00→$14.50.
재채점 효과 **+0.47~9.02점 · pass→fail 뒤집힘 0**.

### 3b. 우리 frontier 기준선이 STALE — 갱신 의무

| 모델 | 우리 캐시(2026-07-12) | v1.0.1 보드 | 차이 |
|---|---|---|---|
| gpt-5-5 | 0.374 | **0.464** | +9.0 |
| gpt-5-4 | 0.307 | 0.394 | +8.7 |
| gpt-5-2 | 0.247 | 0.322 | +7.5 |
| distyl(custom) | 0.312 | 0.312 | 0(미재채점) |

⇒ `sim_results/banking_perstep_frontier_2026_07_12.txt`와 [[47]] 인용부를 **v1.0.1 기준으로 갱신**해야
한다(논문·덱에서 이 표를 쓰고 있으면 수치가 낡음). per-step 재도출도 같이.

## 4. 실행 계획 (점증 — 사용자 지시 ⑷)

| 단계 | 내용 | 산출/판정 | 비용 |
|---|---|---|---|
| **A** | ax32(bm25·nt2·32) 완주·수확·판정 | 이번 아크 종결·per-case 기전 | 진행 중 |
| **B1** | **v1.0.1 업그레이드** + 로컬 패치 재적용(`message.py coerce_arguments` — vLLM JSON-문자열 인자→dict·로컬 서빙 필수) + 배터리 회귀 | 버전 정합 | 0 |
| **B2** | **기존 궤적 재채점**(모델 호출 0) | **#329 오염분 계량 = 레버 효과 vs 벤치 버그 분리** | 0 |
| **B3** | 러너에 **`--user_reasoning_effort`**(기본 low) 추가 · go_stack **alltools 전환** · **P2 재설계 표기**(bm25 전용 신호) | 설정 정합 | 0 |
| **C** | **alltools 스모크 6태스크·nt1** | 프롬프트 전환 정합·레버 발화·임베딩 연결·컨텍스트 | ~$0.4 |
| **D** | **alltools × 32 × nt2** ← *"구조 만들기"* | bm25 32와 짝 비교 = **검색 설정 효과 귀속** | ~$4 |
| **E** | **64 × nt2** | 확장 안정성 | ~$8 |
| **F** | **97 × nt4 = 388 sim** | **제출 후보**(Custom·banking 단독) | ~$25+임베딩 |

⚠D·E는 태스크 집합이 97이 아니므로 **내부 진척 판정용**이고, **보드 비교 주장은 F에서만** 가능하다.
⚠비용은 전부 **[D] 추정** — B 단계에서 ax32 실청구액으로 갱신 후 [[09]] 승인.

## 5. 승리 조건 (특허 주장 기준·사용자 지시 ⑵)

**pass^1 ≥ 46.39 ∧ pass^2 ≥ 36.94 ∧ pass^3 ≥ 31.19 ∧ pass^4 ≥ 27.84 ∧ reward 우위** — 전부 동시.

- **pass^2~4가 우리 유리 축**: 결정론 게이트 = 시행 간 변동 억제. Distyl 붕괴(31.19→13.40)가 공백의 실증.
- **pass^1 46.39가 진짜 관문**: 32B 로컬 + alltools로 어디까지 가는지가 D 단계 1차 신호.
- **`reward` 미결**: GPT-5.5 제출 JSON에는 `pass_1..4`와 `cost`만 있고 `reward` 필드가 없다 —
  **reward 정의·소재 확인이 선결**(evaluator avg reward가 pass^1과 동일한지·부분점수 포함 여부). B 단계 항목.

## 6. 미결·리스크

1. **reward 정의 확인**(§5) — 특허 주장 문구가 여기 걸린다.
2. **alltools 프롬프트 전환의 레버 영향** — 105종이 bm25 프롬프트 위에서 저작됨. **P2는 bm25 hit/score
   전용이라 재설계 확정**. C 단계 스모크가 게이트.
3. **reasoning 모델 전환은 보류**(사용자: 없으면 현 32B로 간다). 로컬에 `Qwen/QwQ-32B-AWQ` 실재 —
   D·E가 frontier 대비 유망하면 그때 스모크로 판단(원장 연속성·컨텍스트 리스크 때문에 기본은 현 32B).
4. **컨텍스트 예산** — alltools는 문서 다량 회수라 ctxover 이력(C205/C208①)과 겹친다. C에서 계측.
