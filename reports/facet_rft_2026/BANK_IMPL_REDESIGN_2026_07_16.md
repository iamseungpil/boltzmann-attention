# BANK 구현 설계 — 정본 2-트랙(outer/inner)에 exhaustion→fail 추가 (2026-07-16)

> **이 문서는 새 아키텍처가 아니다.** 정본 = `BANK_TWO_TRACK_DESIGN_2026_07_16`(Track A outer/inner + Track B inner F3 slot) + `BANK_EPLAN_ALLACTION_IMPL_DESIGN_2026_07_16`(t2_eplan 구체배선) + C101(배선 완료).
> 이 문서 = 정본 구조 **그대로 두고**, 사용자 지적 1건(미명시 빈틈)만 추가한다: **미근거 write의 pass-through → fail/포기 선언.**
> 재작성 트리거(2026-07-16): 표류 중단·최근 결과 왜곡 금지. **아래 §1은 정본 재확인(재결정 아님)·§2가 유일 delta.**

---

## 1. 정본 재확인 (이미 결정남 — 재론·왜곡 금지)

### 1.1 loop = outer / inner 2층 (정본·C90·C94·TWO_TRACK §3·line 80)
```
Track A = outer/inner 결정론 loop:
  ┌ OUTER (across-item · coverage/discovery/reach)
  │   · FIND-discovery : 미조회 read 강제 (not-surfaced 60% · REACH)
  │   · COVERAGE       : required − done (surfaced-not-done 40% · H_min-track)
  │   지배 = under-action 68% (C94) · 종료 100% user_stop (조기종료 메타실패)
  └ INNER (per-item · per-operand)
      · COMPUTE  : 정책-계산(liability/amount_difference) 결정론 (C81)
      · GET-⋈    : id·참조 decidable filter (reference_filter 82% · C78)
      · classify : F3 enum slot  ← Track B (§1.2)
      · ASK      : user-원천/data부재 (경계)
  H_min 종료 (전 갭 닫힘)
```
- **Track A = outer/inner 결정론 loop. Track B = inner의 F3 slot.** (TWO_TRACK line 80·왜곡 금지)
- 개입 = 생성-레벨·**write 강제 금지**([[14]]·[[16]])·read/discovery만 강제.
- 오프라인 상한(❷ 리뷰 교정·정본): **HARD-only 9.9% · +SOFT(COVERAGE) 12.0% · +F3(B) 29.3%** (관측 38.3%). 상한≠실현.

### 1.2 F3 enum = 이미 결정남 (category=풀림 / reason=못 풂·SFT) — 재결정 금지
| F3 유형 | 결정 (정본·C99·C100·TWO_TRACK §9) | 처방 |
|---|---|---|
| **dispute_category**(사실-도출) | **one-shot 프롬프트로 풀림** (zero 55% → **few-shot 81.7%**·induction) | **SOFT one-shot 대조(실제예+prior-conflict 반례)·SFT 불필요·더 쌈** = inner slot 프롬프트로 해결 |
| **dispute_reason**(강한-prior 서사) | **어떤 프롬프트도 무효** (35%=majority·anti-prior·few-shot 반례·conf-gate 전부 실패·[[42]] 천장) | **HARD = Track B SFT+prior억제 DPO/NPO**. 단 **작은 slice**(지배=coverage/compute)·우선순위 낮음. **중간 stopgap = online-H_min bounded-ASK**(확신-오파일링→질문1개·C100·net-safe) |
- **∴ two prescriptions = 이 표가 전부**(사용자 지적): category→one-shot(soft가 되는 유일 경우)·reason→SFT(soft 무효 확정). **soft를 무차별로 쓰지 않는다.**
- soft 프롬프트가 되는 조건(S)= **격리·한 결정점·사실-도출**뿐(C42/C67). in-vivo 강한-prior엔 무효.

### 1.3 배선 상태 (C101·이미 완료)
- outer(a)(b)(c) 배선 완료: 파서·entity_key=transaction_id·bulk-reader=enumerator·디스패처 unwrap. L1/L2 discovery deny·CP5 coverage 리마인더 banking 발화 실증. test_eplan ALL PASS(92).
- inner: reference_filter(⋈·배선·전이검증)·compute keystone(오프라인 755replay 90.9%·**라이브 미검**).

---

## 2. ★유일 delta — 미근거 write의 pass-through 폐기 → fail/포기 선언 (사용자 지적·빈틈)

### 2.1 빈틈 (정본 두 문서 미명시)
- 두 정본 어디에도 **deny cap(T2_EPLAN_DENY_CAP=4) 소진 시 거동**이 없다 → "write 강제 금지"가 암묵적 **pass-through**를 함의 → **미근거 write가 그대로 커밋**(2026-07-16 관측: task_040/041 날조 txn_id가 cap 후 통과).
- in-vivo 실증(2026-07-16): soft deny 받은 32B는 교정 대신 ①동일반복 ②표면변형 ③옆-날조 → cap 후 통과 = [[42]] 천장 in-vivo.

### 2.2 원칙 추가 (도메인일반·[[05]]·C44 선례)
정본 규율에 한 줄 추가:
- **write-강제-금지**(안 시킨 write 강제 emit 금지) — **유지**.
- **+ write-통과-금지-when-ungrounded**(신설): operand가 outer(discovery)·inner(⋈/compute/classify) 어디서도 grounded 안 되고 예산 소진이면 **그 write를 커밋하지 않는다**.
  - 소진 분기 = pass-through **아니라** → (i) **GET-폴백**(getter 있으면 스캐폴드가 조회 공급·C44 §3·deny-and-hope 아님) → (ii) **bounded ASK**(필드당 1개·H_min·C100) → (iii) 그래도 불가/거부면 **fail 선언**(honest abstain·NL "이 항목은 확인 불가로 처리 못 함"·write 미커밋·조기 user_stop 대신 명시적 미완).
- **근거(C44)**: 출처-루프의 확정 semantic = "ASK가 날조를 대체·소진 시 GET-강제 폴백"(67→0%·Δspurious 0). 즉 **미근거 통과 금지는 이미 C44에 있고, outer E-PLAN 배선이 이 exhaustion semantic을 빠뜨렸다**(그래서 pass-through).

### 2.3 모순 없음
- "write 강제 금지"=원치 않는 write 강제 emit ✗ / "통과 금지"=미근거 write 허용 ✗. 둘 다 = **grounded write만 통과·나머지는 GET폴백→ASK→fail.** write를 *만들지도* *조용히 통과시키지도* 않는다.
- Δ-계측(모트·정본 유지): over-block=0(정당 write를 fail로 오종결 0·C44 실증)·Δspurious≤0.

### 2.4 구현 (최소 변경·엔진 리터럴0)
- `t2_eplan_patch`/`t2_gate_patch`의 deny-cap 소진 분기: `pass-through` → `exhaustion_handler`(GET폴백 시도 → 없으면 ASK 마커 → 소진 시 abstain NL·write drop). 도메인일반(getter 존재=A2 source_map).
- **★설치 전제(2026-07-16 dark 버그 재발방지)**: 이 레버들은 `unified()` 호스트에 있음 → `T2_GATE_REGEN=1 T2_GROUND=1` 없으면 plain apply()로 전부 dark. 실행 config에 필수.
- 단위테스트: 미근거 write → GET폴백/ASK/fail 분기(pass-through 0)·grounded write 무회귀·over-block 0.

---

## 3. 검증·순서 (정본 §4 유지 + 2.4)
1. **무료 설치검증 스모크**: `T2_GATE_REGEN=1 T2_GROUND=1 T2_EPLAN=1 T2_RESOLVE=1 T2_COMPUTE=1` → `UNIFIED regen ON`·GET폴백·**fail 선언 마커** 발화 확인(pass-through 0)·thrashing(soft) 아님.
2. **Ⓐ 오프라인 R3 게이트**(무료·최우선): 43 floor replan recall vs gold. 미달 시 밤샘 금지.
3. **오프라인**: 040/041 날조가 GET폴백/fail로 잡히나(통과 0).
4. **[[09]] 유료**: nt≥3(변량 평균화·nt=1 금지)·floor 인터리브·gpt-5.2 user-sim. 탐색 유료런 금지·확인용만.

## 4. 이 문서가 바꾼 것 / 안 바꾼 것
- **안 바꿈(정본 유지)**: outer/inner 2층·Track A/B·F3 결정(category one-shot/reason SFT)·write강제금지·상한 봉투·배선(C101).
- **바꿈(유일 delta)**: 미근거 write **pass-through → GET폴백→ASK→fail** (§2). + 설치 config 필수조건(T2_GATE_REGEN/T2_GROUND) 명시.
