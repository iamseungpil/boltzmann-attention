# banking per-step 다층 실패 분해 — 균일 연산-loop의 실증 (2026-07-16)

> 사용자 지시: 첫-원인 아닌 **per-step 다층 전수**(여러 단계 문제 다 세고 다 해결)·**에러대처 포함**·**reach/discovery/dispute가 같은 경계문제로 치환→outer/inner loop 전 워크플로 확장**.
> 스크립트 `bank_perstep_decomp.py`(DB-basis 실패 4262·infra제외·17궤적·[[08]] over-action 감사 교차검증). 로컬 무료.

## 0. 한 줄
banking 실패는 **다층(78.3%)**이고, 모든 층이 **{FIND/COVERAGE/GET-⋈/COMPUTE/GATHER-ASK/OVER}=C92 연산-오분류의 per-step 인스턴스**다. 지배 = **under-action/discovery 경계(COVERAGE 40.7%+FIND 27.2%=68%)**이며, **종료는 100% user_stop**(몇 층이 깨졌든 갭 안고 조기중단). ⇒ dispute-특화·단일-연산 컨트롤러 무효. **전 워크플로 per-step 균일 연산-loop + H_min 종료**가 정답(사용자 통찰 실증·C90/C92 정합).

## 1. 다층 구조 (첫-원인 귀속 폐기)
- sim당 실패 층수: 1층 **21.7%** vs **≥2층 78.3%**(6+층 25.5%). 별도 pure-DB(action-check 없음) 18.6%=오프라인 blind(C93).
- ⇒ **"다 해결" 필요**: 단일-연산 fix(dispute-only·COMPUTE-only)로 통과 가능한 건 ≤22%. 대다수는 여러 층을 *동시에* 닫아야 pass.

## 2. 연산 타입 전체 빈도 (전 16387 층·다층·[[08]] 교차검증)
| 연산 (C92·[[16]]) | % | 의미 | 레버 |
|---|---|---|---|
| **COVERAGE** | **40.7%** | 필요한 write 미수행 | 아우터 loop coverage-track + H_min |
| **FIND-discovery** | **27.2%** | 필요한 read/열거 미수행 (reach) | 강제 열거(FIND) |
| GATHER-ASK | 21.3% | write 필드 enum/reason/값 틀림 (⋈/의미) | inner: ASK/그라운딩 |
| GET-⋈ | 4.2% | 엔티티 id 틀림 (잘못된 참조) | inner: GET/⋈ |
| COMPUTE | 3.7% | ABox-compute 필드 틀림 | inner: 결정론 COMPUTE |
| OVER-ACTION | 2.9% | gold 없는 진짜 DB-write | suppress (게이트 금지축) |
- **under-action/discovery(COVERAGE+FIND)=68%** = 압도적 지배 = 사용자 "reach 최대 병목" 실증·C80/C52 정합.
- **★over-action 교차검증**: 초판 21%는 절차도구(log/KB검색/unlock/shell/escalation) 오탐 → denylist 후 **2.9%**, proxy X=1,Y=0=**3.6%**(C93 §3.5)와 수렴 = 신뢰.

## 3. 공존·연쇄 (같은 경계문제로 치환)
- 공존 Top: COVERAGE+FIND 1241 · COVERAGE+GATHER 863 · FIND+GATHER 626. ⇒ 경계 연산들이 *함께* 실패(discovery 안 하고·cover 안 하고·의미 오분류) = 단일 under-action 근원.
- **연쇄 신호**: read-miss(FIND) + 하류 write 문제 동시 = **33.9% sim** ⇒ discovery 선행성(열거 안 함→write 틀림/누락). reach가 dispute·write "이전" 병목이라는 기존 분석과 정합.

## 4. ★종료거동 = H_min의 load-bearing 근거
- **전 층-수(1~6)에서 종료 = 100% user_stop**(4262/4262). 즉 sim이 1층 깨지든 6층 깨지든 agent는 *항상* 갭을 안고 조기중단(max_steps·crash 아님).
- ⇒ 메타-실패 = **조기종료(under-action)**. 몇 개를 틀리게 했느냐 이전에, **끝까지 진행 안 함**이 지배. **H_min 종료-게이트(잔여 갭>floor면 continue)가 유일하게 이 메타층을 닫음**([[07]] control-not-prompt·C92 문제2).

## 5. ★통합 결론 (사용자 통찰 실증)
- reach·discovery·coverage·dispute-args·over-action = **분리된 문제 아님**. 각 gold 스텝에서 **해소연산을 오분류**한 동형 인스턴스(C92). 
- **outer loop(enumerate+coverage)과 inner router(per-item operator)를 특정 액션이 아니라 전 gold DAG의 *모든* 스텝에 균일 적용**하면 = 하나의 loop:
  ```
  for each residual gap in gold-DAG (H_min>floor):
      op ← classify{GET(알려진 id) | FIND(열거/discovery) | COMPUTE(결정론) | ASK(비결정)}
      execute(op); update coverage; suppress over-action
  continue until H_min floor (모든 갭 닫힘)
  ```
- outer/inner 구분은 *스케일*(across-item vs within-item)일 뿐 같은 primitive. 78% 다층·68% under-action/discovery·100% user_stop = **이 균일 loop이 전 층을 닫아야 pass**를 실증.
- **컨트롤러 정정(C93 강화)**: sim-level arm 분류(재앵커)도 여전히 거침 — **per-step DAG-walk + operator-classify + H_min**이 정본 아키텍처. dispute-only(C93 폐기)·sim-arm(부분)을 대체.

## 6. caveat
- read/write = 이름-prefix 휴리스틱(ABox producers 비어 대체·도메인일반). 절차도구 denylist=감사 기반.
- GATHER-ASK는 enum/의미 갭을 뭉뚱그림(⋈-경계 vs 단순 gather 미분리·inner router 세분은 별도).
- action-check 밖 pure-DB 18.6%는 per-step 분해 불가(C93 blind·DB-replay/live 필요).
- 층수=미충족 args-row 수 근사(같은 도구 다중 gold 액션=다층 정당). proxy ~90% tight 상속.
