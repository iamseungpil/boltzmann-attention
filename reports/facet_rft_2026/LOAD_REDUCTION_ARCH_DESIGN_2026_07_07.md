# 부하-감축 아키텍처 설계 — context-isolation + 결정론 dispatch smoke (2026-07-07)

> **위치**: `CLEAN_NT4_FAILURE_FORENSIC_2026_07_07.md`(6기전 분류) + `LEARNED_WING_MECHANISM_DESIGN`(reachability)의
> 후속. 사용자 제안(멀티에이전트: 짧은-단계 전문 agent + context-유지 agent + formalize-전담 agent)을
> **thesis-정합 최소형**으로 환원해 검증하는 설계.
> **불변**: [[05]] scaffold 도메인-일반·A2만·[[13]] 학습/scale 최후·scaffold 최소·[[03]]#8 forensic>mechanism·
> smoke 먼저·[[09]] 무료(offline)먼저·[[10]] selector/verifier=결정론·LLM=formalize만·[[00]] boundary-translator.

---

## 0. 한 줄
잔여 실패의 통일된 정체 = **competence–performance 갭(부하 하 실행붕괴)**. 부하를 **두 원천**으로 갈라,
각 원천이 **비-학습·비-scale 개입**으로 복구되는지 **offline(무료)→live-smoke** 2단계로 판정한다.
사용자의 멀티에이전트 직관을 최소형으로 검증하고, 성공 시에만 full-build로 escalate.

## 1. 가설 (forensic-도출·엄밀 정의)
작업 = 하위결정 $s_1..s_H$. $p_{iso}(s)$=격리 정답률·$p_{traj}(s)$=궤적내 정답률. **부하 = $p_{iso}-p_{traj}>0$**·
과제성공 $\approx \prod_i p_{traj}(s_i)$(compounding). 부하는 **두 원천**(CLEAN_NT4 scale-절단으로 실증):

| 원천 | 기전 | scale | 격리로 복구? | 개입 |
|---|---|---|---|---|
| **(A) 위치/컨텍스트** | M2 state/⋈참조·M3 loop·M5 attention-dilution | reducible(14B악화) | ✅ (iso 100%) | **E1: context-isolated formalize 서브콜** |
| **(B) 조합/완결** | M1 coverage·M4 goal-drift·M6 over-action | **invariant(32B≈14B)** | ❌ (컨텍스트 줄여도 안됨) | **E2: 결정론 완결 dispatch/gate** |

**핵심 주장**: (A)는 짧은-컨텍스트 formalize 서브콜(=사용자의 "전문 agent")로 복구·(B)는 결정론 오케스트레이터
(=사용자의 "context-유지"를 **LLM 아니라 코드로**)로 복구. **context-유지자를 LLM으로 두면 scale-invariant 실패를
그 LLM이 그대로 상속**(∴ 결정론 필수·[[10]]).

## 2. 무엇을 테스트하나 + GO/NO-GO
사용자 3분할의 thesis-정합 매핑:
- 짧은-단계 전문 agent → **E1**(위치부하) · formalize-전담 = E1의 서브콜 역할(LLM=formalize) · context-유지 →
  **E2**(결정론 dispatch·코드) · selection/verify = 기존 결정론 게이트.
- **GO(full multi-agent/오케스트레이터 build)** = E1이 (A)-버킷을·E2가 (B)-버킷을 **over-block=0로 유의 복구**.
- **NO-GO** = 복구 미미 → 잔여가 순수 위치/조합부하 아님(= 진짜 formalize-capability 또는 경계) → 학습-wing(G2)으로.

## 3. E1 — context-isolated formalize 서브콜 (위치부하·M2·M5)

### 3.1 기전
결정점(write-tool이 *선택 인자*를 담음: 변형 `new_item_ids`·⋈ `order_id`)에서, **궤적내 선택을 신뢰하지 않고**
동일 모델에 **짧은 clean 컨텍스트 서브콜**을 발행: "후보집합 C + 사용자 목표 G → 어느 것?". 서브콜 출력을
인자로 사용. = 위치부하를 컨텍스트 격리로 제거.

### 3.2 훅·구현 스케치 (기존 코드 재사용)
- 훅 = `t2_gate_patch.gen_gated`(생성-레벨·이미 tool_call intercept). write-tool + 선택인자 감지 시 서브루틴.
- **후보 C 추출 = 도메인-일반**: 이미 fetch된 record(`exec_augment`의 `_rec`/present 데이터)서 변형목록·주문목록.
  = present가 보여주는 그 집합. 신규 도메인지식 0.
- **목표 G 추출 = 대화의 user turns**(agent가 이미 보는 것·`_regen_last_user`+reason). **eval/gold 미접근.**
- 서브콜 = `operand_probe_n100.ask()` 패턴(단일턴·temp0·60tok)·프롬프트 = "Given candidates {C} and the customer's
  goal {G}, output ONLY the chosen id." **모델이 formalize+select**(scaffold는 컨텍스트 scope만 조작·답 미주입).
- 결과 사용 = 해당 tool_call 인자를 서브콜 선택으로 set(differ 시). = 사용자의 "전문 agent가 답 준다".

### 3.3 anti-cheating 제약 (필수·[[05]])
1. 서브콜은 main agent가 **가진 정보만**(present 후보 + 대화목표)·**gold/eval/DB-정답 미접근**.
2. 서브콜 = **모델의 formalize**(scaffold가 argmax 계산 안 함·정답 안 박음). 컨텍스트 격리 ≠ 답 주입.
3. 프롬프트·추출 = **도메인-일반**("select-among-candidates given goal"·retail 토큰 0). `grep "if domain"=0`.
4. 도메인 추가 = A2-swap만(후보-추출 spec = present_specs 재사용·신규 0).

### 3.4 측정 (offline 먼저·무료)
- **Phase A (offline·gpt-4.1 0)**: clean-nt4 `results.json`의 (A)-버킷 실패 궤적에서 결정점을 뽑아 E1 서브콜 replay →
  **선택이 gold로 flip하는 비율**. flip률 높음 = 위치부하 확증·E1 유효. (operand_probe를 궤적-결정점에 정조준.)
- **Phase B (live-smoke·local user-sim·gpt-4.1 0)**: E1 배선 후 (A)-fail-set N~15 + **control set(passing 15)**
  로 live run(Claude/deterministic user-sim) → pass 복구 + **over-block(control 회귀)=0** 확인.
- 지표 = per-case([[08]])·pass^1 점추정 금지·gold-write diff.

## 4. E2 — 결정론 완결 dispatch/gate (조합부하·M1·M4)

### 4.1 기전
**명시 universal 양화사**("all/both/every my pending orders") 요청에서: (i) 술어 formalize(LLM: "pending orders
of user")→(ii) **결정론 enumerate**(DB서 조건-매칭 엔티티집합 = working-set)→(iii) **완결 gate**: 모델이
종료/transfer 시도 시 미커버 엔티티 있으면 deny("아직 {remaining} 미처리"). = coverage(M1)를 *루프/게이트가 보장*,
LLM 완결능력에 안 맡김. (NEXT_LEVERS coverage-controller·handoff 0.2와 동일 계보.)

### 4.2 구현 스케치
- 훅 = 게이트 kind 신설 `coverage`(gate_interpreter)·gate deny 경로 재사용(`gen_gated`의 deny+regen).
- working-set = **A2-구동**: `coverage_spec`{entity=orders·predicate_source=user_id+status filter}·**결정론 enumerate**.
  술어(pending/특정속성) = 대화서 LLM formalize(양화사+속성)·enumerate = 코드. [[05]] 준수(도구/정책 미하드코딩).
- 완결 판정 = write 관측(`gate.observe`) vs working-set 차집합. covered=∅까지 종료-deny.

### 4.3 over-block=0 (대칭 크레딧·[[03]]#9)
- **명시 universal 양화사 + 올바른 엔티티레벨에만** 발화. 암묵 scope("고쳐줘"·단수) = **residual**(gate 미발화).
- 술어 mis-formalize 위험 → 애매하면 gate off(보수적). control set(비-universal passing)서 **추가 실패=0** 필수.

### 4.4 측정
- **Phase A (offline)**: clean-nt4 coverage-버킷 궤적서 "완결 gate가 미커버를 잡았을까 + passing엔 오발화 안 하나"
  건별 판정(무료·궤적 정독).
- **Phase B (live-smoke)**: E2 배선·coverage fail-set + control → 복구 + over-block=0.

## 5. 측정 프로토콜 (공통)
- **비용([[09]])**: Phase A 전부 offline(저장 results.json·gpt-4.1 0). Phase B = local user-sim(Claude/det·gpt-4.1 0).
  **gpt-4.1 full run = E1/E2 GO 확정 후 *확인 1회만*·사용자 승인.** 탐색용 full-run 금지.
- **양 scale**: 14B·32B 둘 다(scale-invariant 버킷[M1] 복구 = thesis 증거). smoke = nt=1·num_tasks 소.
- **per-case([[08]])**: 집계→결론 금지·gold-write diff + 궤적 정독. flip/복구/over-block 건별.
- **control set**: passing 태스크로 회귀(over-block) 측정 = 대칭 크레딧([[03]]#9).

## 6. 결정 기준 (GO/NO-GO)
| 결과 | 해석 | 다음 |
|---|---|---|
| E1 offline flip 高 + live 복구 + over-block0 | (A)=위치부하 확증·격리로 복구 | 오케스트레이터 서브콜 build |
| E2 offline 포착 + live 복구 + over-block0 | (B)=완결부하·결정론 dispatch로 복구 | coverage-controller build |
| 복구 미미(flip 低·live 무변) | 잔여≠순수 부하(진짜 formalize-capability or 경계) | 학습-wing G2로·부하-arch 보류 |
| over-block>0 | 게이트/서브콜이 정상 태스크 훼손 | 발화조건 축소·재smoke |

## 7. 규율 자가감사
- **[[05]]**: E1 후보-추출·프롬프트·E2 enumerate = 도메인-일반·A2-구동. 결정질문 1(도메인특화 순증?)=No·2(유동판단
  동결?)=No(선택은 여전히 모델)·3(scaffold가 도메인행동 수행?)=No(컨텍스트 scope만·답 미주입). `grep if domain=0`.
- **[[13]]**: full 멀티에이전트 = scaffold-grow(최후). ∴ **smoke 먼저·offline 먼저**·GO 시에만 build. scale/학습 아님.
- **[[03]]#8**: E1/E2는 forensic 6기전에 *정합*(충돌 아님)·elegant멀티에이전트를 forensic-확증 없이 build 금지.
  #9: E1(복구)에 증거 요구하면 E2(over-block)에도 동일·shaped≠closed(offline≠live)·실측前 banking 금지.
- **[[10]]/[[00]]**: E1 서브콜=LLM formalize(boundary translator)·E2 enumerate/완결=결정론(selector/verifier). 역할분리 정합.
- **[[09]]**: Phase A 전부 무료(offline replay)·Phase B local user-sim·gpt-4.1은 최종확인 1회.

## 8. cheating-surface 자가리뷰
- **E1이 답을 박나?** No — 모델이 clean 컨텍스트서 선택. scaffold=scope 조작만. (단 후보-추출이 gold-필터면 치팅 →
  후보=present 전집합·필터 금지 강제.)
- **E2 enumerate가 gold를 읽나?** No — DB+formalized 술어서 도출(eval criteria 미접근). gold-set 사용 = 치팅 → 금지.
- **over-block로 pass 부풀리나?** control set 회귀=0 강제로 차단.
- **부하-복구가 실은 make-or-break rig?** E1은 ⋈-resolver 신규 추가 아님(모델이 여전히 매칭)·[[05]] join-resolver 금지 준수.

## 9. 사용자 리뷰 대기 (open questions)
1. **E1 override 강도**: 서브콜 선택으로 (a)인자 replace vs (b)"reconsider" 피드백 후 재생성. (a)=사용자의 "전문 agent가
   답 준다"에 충실·(b)=덜 침습적. 권장=(a)(측정 깔끔)·단 anti-cheating §3.3 준수.
2. **E2 술어 formalize 주체**: 양화사+술어를 (a)별도 LLM 서브콜 vs (b)main agent 발화서 파싱. 권장=(b)(서브콜 최소).
3. **scope**: E1·E2 동시 vs E1(위치·최대버킷) 먼저. 권장=**offline Phase A를 둘 다 먼저**(무료·빠름)→flip률 보고 build순서.
4. full multi-agent build는 **offline+live-smoke GO 후**로 확정(지금 build 아님)·동의?

## 10. 시퀀싱
1. **Phase A offline (무료·즉시)**: E1 결정점-replay flip률 + E2 완결-gate offline 판정. 도구=`operand_probe` 확장 +
   궤적 정독 스크립트(scratchpad). 병행 = 지금 도는 operand_probe(14B)가 E1 Phase-A의 일부(변형/⋈ iso flip).
2. 리뷰(flip률·over-block offline) → E1/E2 배선 여부·순서 결정.
3. **Phase B live-smoke (무료 user-sim)**: 배선 후 fail-set+control·양 scale·per-case.
4. GO → 오케스트레이터/coverage-controller build → gpt-4.1 확인 1회(승인).
5. NO-GO → 학습-wing G2(reachability)로 피벗.
