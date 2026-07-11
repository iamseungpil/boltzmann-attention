# 차세대 레버 2종 설계서 — NOTICE-PERGATE · FORMALIZE-EXEC (2026-07-11)

> **상태 = [D] 설계서 v1.1 (리뷰 반영·2026-07-11). 현 스택 동결 중(사용자 2026-07-11) — 스택 편입·라이브 arm은 S4/S5 완료 후.**
> v1.1 변경: ①§2.6 V0 결정점 수확을 전-456-sim으로 확장(희소 타입 n≥30 구조적 불충족 해소+통과-sim over-fire 계측 동시 획득) ②§1.1③ 산출-키 하류 소비자 감사·G4 레거시 키 dual-emit 추가 ③CENSUS §3b(동적 `<pm>` 주석)를 G8로 supersede 명기(이중 구현 방지) ④상태 표기 사실 정정(원판도 05d71c27로 커밋됨 — "커밋 0" 기재는 오기였음).
> (RESEARCH_MASTER §4 E-XGRAMMAR 행과 동일 규율: "스택 동결 — S4/S5 후". 두 레버 모두 지금은 등재·설계 고정만.)
> 파생: 레버1 = S1c 보고의 기확정 변경안 정식화(`test_notice_gate.py` B1-B4 실증 위) · 레버2 = §1.5 Q3-compound 경로
> + C56 처방("formalize→결정론 직렬화") + C61 E-ISO(형식화-부하 실재)의 설계-구체화.
> 규율: [[05]]/[[10]]/제1원리(반대편 계측 필수)/[[08]](결론은 per-case)/[[09]](무료 검증 先·유료는 승인 후 최소 scope).

---

## §0. 공통 — 채널-축 좌표 · 우선순위 · 현 스택 간섭

### §0a. 채널-축 좌표 (taxonomy (a)~(f))
개입-채널 축 = "개입이 에이전트 루프의 어느 지점에서 작동하는가". 정본 (a)~(f) 표는 특허 taxonomy 부록 X(로컬 전용·[[32]])
소관이며, repo에 영속된 앵커는 RESEARCH_MASTER §4 E-XGRAMMAR 행의 **"(f)=디코드-시점 제약(현 스택 미포함=전 개입이 생성-후)"**
하나다. 그 앵커와 현 스택 구현으로 재구성한 좌표(부록 X 대조는 착수 시 확인 항목):

| 채널 | 정의 | 현 스택 점유 |
|---|---|---|
| (a) 프롬프트/정책 텍스트 | 무효 확정([[42]]·C30·C41) | — (죽은 채널) |
| (b) read-증강 주석 | tool 출력에 결정론 주석(replay-safe) | calc/nested/present_specs |
| (c) 생성-레벨 deny→피드백 재생성 | 비커밋 작업버퍼서 deny+복구절→재생성 | gate regen·prov regen |
| (d) silent 제자리 치환 | 인자만 교체·대화/턴 완전 불변 | P-A GROUND·P-B switch·P2 원리-디폴트 |
| (e) 격리 서브콜 | 동결 문맥+후보 → 판단만 반환(히스토리 밖) | `_t5c_disamb_subcall`(t2_gate_patch.py:779) |
| (f) 디코드-시점 제약 | guided decoding | 미점유(E-XGRAMMAR [D]) |

- **레버 1 NOTICE-PERGATE = (c)** — 기존 notice kind가 이미 쓰는 채널 그대로. 변경은 채널이 아니라 **엔진 판정 입력의 입도**
  (스칼라 1개 → per-gate callable). 채널 신설 0.
- **레버 2 FORMALIZE-EXEC = (e)+(d) 합성** — 격리 서브콜이 *형식화*하고 결정론 실행기가 계산, 전달은 (d) silent 치환
  또는 (b)-변형 후보-주석(§2.3 채널 선택은 Δ측정으로). C10("부작용은 scope/채널에서 온다")·C62("정보는 무해·전달 기전이
  유해") 계보의 직계 — 격리에서 검증된 것을 격리된 채로 소비.

### §0b. 우선순위 권고 — **NOTICE-PERGATE 먼저, FORMALIZE-EXEC V0(무료 격리)는 병렬 가능**
1. **NOTICE-PERGATE 선행 근거**: ① 변경 최소(파일 3·행 ~20·기확정 변경안 = S1c 보고) ② 검증 인프라가 이미 있다
   (`test_notice_gate.py` [B] B1-B4가 현행 한계를 고정 — 갱신만 하면 회귀망) ③ **banking 전이 선행조건**: banking A2에
   2번째 notice(GB2형)를 넣는 순간 현행 first-notice 배선은 B2(교착)/B3(기존 게이트 파괴)를 재현 — 엔진 한계가 A2-swap
   전이([[05]]·[[11]])를 직접 막고 있는 유일 지점 ④ 실패 시 손실 작음(A2서 G8 제거 = 원상복구).
2. **FORMALIZE-EXEC 후행 근거**: E1′ 하향 이력(C23)이 있어 **격리 형식화-정확도 선측정(V0·무료) 통과 전 스택 편입 금지**.
   단 V0 자체는 로컬 무료(E-ISO C 프로토콜 재사용)라 S4/S5 대기 중에도 준비·측정만은 가능(스택 불변·[[09]] 정합).
3. 기대 payoff(정직·상한): NOTICE-PERGATE = t57형 1 task(확률적·§1.5)+banking 공존 enable / FORMALIZE-EXEC = C64
   C클래스 ≈6 task(t20·t37·t79·t71) 중 형식화-가능분. 크기는 FORMALIZE 쪽이 크나 확실성·비용은 NOTICE 쪽이 압도.

### §0c. 현 스택과의 간섭 지점 (착수 시 회귀 체크리스트)
| 간섭 | 내용 | 처방 |
|---|---|---|
| notice deny ↔ 예산 | G8 추가 시 cancel-class write가 문구 미송신이면 1회 deny(+num_errors) — regen 채널 예산 semantics(C53 승계) | Δtme ≤ 0 게이트·§1.5 census 선행 |
| `T2_GATE_KINDS` 필터 | notice kind 화이트리스트 동작 불변이어야 | 단위 회귀 |
| compliance 소비자 | `t2_compliance.py:35`·`t2_gate.py:40`·`t2_gate_r2_verdict.py`·`t2_passk_autopsy.py` 전부 first-notice 가정 | §1.2③·구 export는 호환 보존 |
| FORMALIZE ↔ P-B DISAMB | fire 지점이 같은 결정점(confirm-write 인자) — 이중 발화 금지 | 관할 규약(§2.4): formalize 先·불가/UNSURE시 DISAMB 폴백·단일 fire |
| FORMALIZE ↔ calc 주석 | CALC-EXT(`argmax_where`)와 재료 중복·서브콜 전사에 `[COMPUTED FACTS]` 혼입 | T5C rev3 N11 필터 승계·관할: 정적(문맥-무관 제약)=calc / 동적(문맥-의존 제약)=formalize |
| FORMALIZE ↔ constraints 게이트 | P-B의 축퇴-스위치 revert 안전장치(T5C V1 U3/U5)는 formalize 치환에도 동일 적용 | 치환 후 게이트 재검사 경로 공유 |

---

## §1. 레버 1 — NOTICE-PERGATE: notice 엔진 최소변경 → 다중 notice 게이트

### §1.0 문제 고정 (배경 = `test_notice_gate.py` 상단 주석·B1-B4 [M])
현행 notice 판정 입력이 **게이트별이 아니라 스칼라 1개**: 호출부가 `_notice_text` = gates 중 *첫* notice의
notice_text(`t2_gate_patch.py:59-60`)를 뽑아 그 한 문구의 송신 여부만 계산(`:210`/`:1196`/`:1314` →
`_transfer_msg_sent(:337)`/`_regen_transfer_sent(:1061)`), `gate_interpreter.check()`의 notice 분기(:209-211)는
그 스칼라를 전 notice 게이트에 공용. `t2_compliance.py:35`·`t2_gate.py:40`도 동일 first-notice 가정.
⇒ notice_text가 다른 2번째 게이트(G8 환불-목적지·applies_to=cancel)는 **A2만으로 배선 불가**:
- B2: G4가 첫 notice면 — 환불 고지를 보내도 cancel 영구 deny(교착·over-block).
- B3: G8을 앞에 두면 — G4 transfer가 환불 문구 기준으로 판정 + compliance TRANSFER_MSG 오염(기존 게이트·계측 파괴).
- B4: `check()` 시그니처 자체에 게이트별 통로 없음.

### §1.1 확정 변경안 (S1c 보고의 정식화 — 최소침습 3점)
**① 엔진 (`gate_interpreter.py:209-211` — notice 분기만)**
```python
if kind == "notice":
    sent = transfer_msg_sent(g["notice_text"]) if callable(transfer_msg_sent) else transfer_msg_sent
    if sent is False:
        return False, g["id"], render_recovery(g)
```
- callable이면 **per-gate 평가**(그 게이트의 notice_text로 송신 여부 계산) / 스칼라(bool·None)면 **현행과 바이트-동일**
  (None=skip·False=deny·True=allow). 시그니처·파라미터 수 불변(B4 프로브는 의미 갱신).
- 도메인 분기 0·notice_text는 A2 데이터 — 엔진 헤더의 "도메인 리터럴 0" 불변식 유지.

**② 호출점 3곳 (`t2_gate_patch.py`) — 문구-매개 클로저 전달(커링만·기존 함수는 이미 text-인자형)**
| 위치 | 현행 | 변경 |
|---|---|---|
| `apply()` gated `:210` | `tms = _transfer_msg_sent(self, a2["_notice_text"])` | `tms = lambda text: _transfer_msg_sent(self, text)` |
| `apply_gate_regen` gen_gated `:1196` | `transfer_sent = _regen_transfer_sent(state.messages, a2["_notice_text"])` | `transfer_sent = lambda text: _regen_transfer_sent(state.messages, text)` |
| `apply_unified_regen` unified `:1314` | 동일 | 동일 커링 |
- `_denied_calls(:1072-1082)`는 값을 그대로 `check()`에 관통시키므로 무변경.
- 비용: 클로저는 notice 게이트 수 × check 호출마다 메시지 스캔 — 현실 게이트 2~3개·스캔은 기존과 동일 O(msgs).
  필요시 클로저 내부 `{text: bool}` memo 1줄(턴-로컬)로 상수화. `_domain_a2`의 `_notice_text`(:59-60)는 스칼라
  소비 잔존처(진단 스크립트) 호환용으로 보존하되 **신규 소비 금지** 주석.

**③ 측정 leg (`t2_compliance.py:35` · `t2_gate.py:40`) — first-notice → applies_to 기준 선택**
- `domain_constants`: `"TRANSFER_MSG": first-notice`(구 export·호환 보존) + 신규 `"NOTICE_GATES": [notice 게이트 전부]`.
- `violations_of_sim`의 G4 의미론(위반 = notice-도구 실행 ∧ 문구 부재)을 per-gate 루프로: 각 notice 게이트 g에 대해
  `applies_to`(＋`applies_when`은 엔진 `_gate_applies` 재사용) 도구가 실행됐고 g.notice_text가 어시스턴트 발화 전체에
  부재면 해당 게이트 위반. 산출 키는 게이트 id별(`G4_TRANSFER_MSG`·`G8_REFUND_NOTICE`) — 기존 G4 숫자와의 연속성은
  retail 단일-notice에서 자동 보장(회귀 검증 항목).
- `t2_gate.py:40` `TRANSFER_MSG`·`transferred_then_notice(:86)`는 deprecated-호환 export로 유지(외부 분석도구
  `t2_gate_r2_verdict.py`·`t2_passk_autopsy.py`가 소비) — 이들은 retail G4 전용이라 per-gate화 불요·주석만.
- **★하류 키-소비자 감사 (v1.1·U단계 체크리스트)**: 산출 키가 게이트-id별(`G4_TRANSFER_MSG`…)로 바뀌면
  `violations_of_sim` 출력 키를 *직접* 읽는 집계(make_figs_results류·리더보드 파이프·과거 비교 스크립트)가 깨질 수 있다
  — t2_gate.py export 보존만으론 부족. 처방: ① 구 키 `TRANSFER_MSG`를 G4 값으로 **dual-emit**(과거 수치 연속성)
  ② 소비처 grep 감사(`TRANSFER_MSG` 검색)를 U단계 필수 항목으로.

### §1.2 하위호환 보장 논거
1. **스칼라 경로 바이트-동일**: ①은 callable 분기 추가뿐 — 기존 테스트([A] A1~A6·B4 시그니처)와 스칼라 호출자
   (`RetailGate`·validate·진단 스크립트) 동작 불변.
2. **단일-notice 도메인 동등성**: notice 게이트가 1개면 per-gate 평가 ≡ first-notice 평가(같은 text·같은 판정) —
   retail 현행 456-sim 결과 재현이 곧 회귀 증명(오프라인 replay로 무료 확인 가능).
3. **A2 미변경 시 무영향**: G8을 넣지 않는 한 게이트 집합 동일 → 신규 deny 0. 롤백 = A2서 G8 1줄 제거.
4. 플래그 불요(엔진 입도 수정이지 새 개입 아님) — 단 G8 자체는 A2 diff로 toggle되므로 arm 정의가 곧 스위치.

### §1.3 표적
- **retail t57형 — `G8_REFUND_NOTICE`** (초안 = `test_notice_gate.py:47-56`에 보존): kind=notice·
  notice_text="Per policy, the refund for a cancelled order always goes back to the original payment method…"·
  applies_to=[cancel_pending_order]. 실측 기전(C64 census·CENSUS_LEVERS_DESIGN §3b): 조건체인 끝("gift card 환불
  안 되면 취소도 하지 마라")에서 에이전트가 취소 실행+허위 발화 — G8은 취소 *전* 정책-사실 고지를 강제 →
  user-sim(체인 보유)이 "그럼 취소하지 마라" → no-op = gold. **에이전트 판단 경유·write 강제 0**(deny+ask만·§1.5 Q5 준수).
  한계 정직(§3b 승계): user-sim이 재확인 안 하면 못 닫음 — 커버는 확률적·Δ로 실측.
  **★§3b supersede (v1.1)**: `CENSUS_LEVERS_DESIGN §3b`의 동적 `<pm>` confirm-주석안은 **G8 정적-문구 notice
  게이트로 대체**(같은 표적 t57·이중 구현 금지). 정적 문구가 §1.4 "notice_text 동적 값 금지" 규약과도 정합 —
  CENSUS §3b에 역방향 표기 필요.
- **banking GB2와의 공존**: banking A2의 notice 게이트(GB2)와 향후 2번째 notice가 같은 도메인에 서도 B2/B3 없이 동작
  — 전이(E-XFER-bank 재개) 전 엔진 선행조건. per-gate화 후 스모크(UNI_OK 동형)로 확인.

### §1.4 반대편 계측 (제1원리 — notice도 over-action 역효과 가능)
- **notice 과다 = 발화-마찰**: 정확 문구 강제는 턴·에러 예산 소비(deny 1회/미송신 write). 계측 = **Δtme ≤ 0**·
  deny 발화수(`[POLICY GATE G8_*]` 마커)·pass→fail flip(Δspurious ≤ 0).
- **★pass-sim 발화 census 선행(무료·착수 게이트)**: 기존 456-sim 궤적(COMP/floor) replay로 "G8이 있었다면 deny했을
  지점" 전수 — **현재 passing인 cancel sim에서의 deny 빈도 = over-block 상한**을 스택 편입 전에 수치로. GO 조건:
  passing-cancel deny가 낮고(문구 송신으로 1-deny 후 회복 가능한 수준) 표적(t57형) deny가 실재.
- 문구 자체의 창-오염(C43 재료화)은 정적 정책-문구 1줄이라 낮음 — 단 notice_text에 동적 값 금지(§3b 초안 그대로
  정책-사실 정적 문구만) 를 A2 규약으로 명기.

### §1.5 검증 계획 (단위 → 오프라인 → 표적 nt=1 · §7c 사이클 규율)
| 단계 | 내용 | GO |
|---|---|---|
| U | `test_notice_gate.py` 갱신: [A] 불변 + [B]를 per-gate 기대로 반전(B2′ 환불고지 송신→cancel allow · B3′ 순서-무관 · B5 G4/G8 독립 판정 · 스칼라 하위호환) + compliance per-gate 단위 | 전부 PASS·기존 [A] 무회귀 |
| O | 오프라인 replay census(§1.4): retail 단일-notice 재현 동일성 + G8 가상-deny 전수 | 재현 100%·over-block 상한 수치 확보 |
| T | 표적 nt=1: t57 + transfer-게이트 회귀(t-transfer 포함 sims) + banking 스모크(GB2) — **S4/S5 후·승인 후** | t57 개선 ∧ Δtme≤0 ∧ Δspurious≤0 ∧ 위반0(G4 불변) |

### §1.6 [[05]] 3질문
| Q | 답 |
|---|---|
| 엔진=도메인일반? | ✅ callable per-gate 평가 로직뿐 — 문구·적용도구 전부 A2 데이터·도메인 리터럴 0 |
| A2만 추가? | ✅ G8 1줄(notice_text·applies_to)·엔진/scaffold 구조 불변 |
| 도메인행동 대행? | ✅ 아님 — 고지 송신·취소 여부 판단은 에이전트 몫(deny+복구절 유도만·write 강제 0) |

---

## §2. 레버 2 — FORMALIZE-EXEC: E1′형 격리 직렬화 (compound criterion)

### §2.0 문제 고정과 근거 사슬
- **§1.5 Q3**: compound/계산형 능력은 thinking이 아니라 **Q1(결정론 실행)으로 회귀** — "형식화(LLM)→결정론 실행(argmin/filter)" [P].
- **C56 [M]**: 동-scale thinking은 |C|≥2 선택을 못 산다(base .145 = QwQ .143) ⇒ 기준-형식화형 레버 = **formalize→결정론
  직렬화**가 유일 잔여. 체계핵 t71 = "최근 주문" 기준-오적용 4/4(user-sim 오확인 고착 포함).
- **C61 E-ISO [M]**: 형식화-부하 실재 — ITEMS A .10 → C(열거+형식화) **.44** · PAYMENT .12→.38. 격리가 여는 질량이 실측됨.
- **CENSUS_LEVERS_DESIGN §2a V0 [M]**: CALC-EXT 정적 주석의 구조적 한계 실증 — t20 Running Shoes MISS(gold=주문과
  같은 size 9 중 최고가 = **제약값이 주문-문맥 의존이라 정적 spec 불가**). 이 자리가 정확히 FORMALIZE-EXEC의 표적.
- **C23(E1′ 하향)과의 차별점 — 명기**: E1′는 (i) 정보-빈약 프로브(C13 각주 — p_iso가 에이전트 실제 보유 정보와 미스매치)
  (ii) DB-기준 성공-write 선택편향 슬라이스에서 F2 −4로 payoff 작아 보임 (iii) 격리→치환 인프라 부재 시점의 설계였다.
  지금은 ① **DISAMB subcall 인프라 재사용**(`_t5c_disamb_subcall` t2_gate_patch.py:779·fire 배선 :955/:1468 —
  격리·전사·파싱·silent 치환·전예외 no-op이 전부 기성) ② **결정론 실행기 재료 기성**(CALC-EXT `argmax_where`/
  `argmin_where` op가 `gate_interpreter.compute_facts:431-450`에 이미 구현) ③ **정보-맞춘 선측정 프로토콜 확보**
  (E-ISO C — 실 궤적 결정점 재생) ④ C56이 thinking 대안을 실측 폐쇄. ⇒ 하향 사유 3개가 전부 해소된 재상정이다.

### §2.1 설계 골자 (3단 직렬화)
결정점 = confirm-write 도구의 **변형-선택 write 인자**(P-B와 같은 fire 채널: hints 매칭·값 문맥-실재·후보 ≥2 —
추가 조건 §2.4 관할 규약).

**① LLM 격리 서브콜 — 기준의 형식화** (`_t5c_disamb_subcall` 변형·FORMALIZE_SYS):
동결된 대화 전사(+해당 인자의 후보 record들)를 주고 "이 선택의 기준"을 **도메인-일반 어휘의 JSON**으로 반환:
```json
{"op": "argmax" | "argmin" | "filter",
 "field": "price",
 "constraints": [{"field": "size", "value": "9"}, {"field": "availability", "value": true}]}
```
- 어휘 규약: `op` 집합 = 엔진 상수(도메인일반) · `field`/`constraints`의 이름·값 = 서브콜이 **대화·기조회 record에서
  추출**(A2에 필드 리터럴 불요 — record의 실제 key와 대조 검증만). 형식 위반/파싱 실패 = UNSURE = no-op.
- `"op": "none"`(기준 자체가 형식화 불가) 및 `"unresolvable"`(기준은 있으나 실행 재료 부재) 분기 포함 — §2.2③.

**② 결정론 실행기 — 에이전트-기조회 변형 목록 위 실행**:
- 입력 = `_parse_tool_outputs(msgs)`서 해당 인자의 **후보 record 전체**(P-B `_candidate_records`의 record 동봉 확장
  — 규칙0: 에이전트 자신이 조회한 tool 출력만·DB 접근 0).
- 실행 = `compute_facts` op 커널 재사용/일반화: constraints로 filter → op(argmax/argmin/filter)를 field에 적용 →
  결과 후보 id(들). 동률 = 전부 반환(단일 아니면 치환 금지·주석/DISAMB 폴백).
- 실행 불가(제약 field가 record에 없음·후보 0) = no-op + `unresolvable` 처리.

**③ 결과 전달 — 채널 선택은 Δ측정으로 (V0 A/B)**:
- **(d) silent 치환**(1안): 실행 결과가 단일 id ∧ 원값과 불일치 ∧ A2 화이트리스트(`formalize_sub_args`·
  `disamb_sub_args` 동형·B2 선례) → `_subst_arg_value`로 제자리 치환. 대화 불변·replay-clean·T5C 원리 4조 승계.
- **(b′) 후보-주석**(2안): 결과를 비커밋 작업버퍼 리마인더(CP5 채널 동형)로 — "형식화된 기준 X의 결정론 답 = id Y" —
  선택은 에이전트 유지(유동성 최대·단 C61 CP 재주입 오염·나비효과 소폭 잔존).
- V0(§2.5)서 두 채널의 fix/break를 같은 결정점 집합에 대해 오프라인 대조 → 높은 net·break=0 쪽 채택(타입별 상이 허용).

### §2.2 표적 (C64 C클래스)
| task | 기준 | 형식화 예 | 비고 |
|---|---|---|---|
| **t20** | 주문과 같은 size 중 최고가(신발) | argmax(price) + constraints[size=9(문맥 추출)] | CALC-EXT 구조적 MISS의 정확한 보완 — 정적 spec이 못 담는 문맥-의존 제약을 서브콜이 추출하면 실행기는 기존 argmax_where와 동형 |
| **t37** | 예산 이하 조합 | filter(price ≤ B)·후보별 가격표까지 | 조합최적화 자체는 스코프 밖 유지(CENSUS §2a 경계 승계) — 실행기는 재료(필터·합)까지·조합 선택은 에이전트 |
| **t79** | 다른 레코드와 attr-match("같은 색") | filter(color = X) — X는 서브콜이 기준-레코드서 추출 | cross-record 제약 = 정적 calc 불가 칸 |
| **t71** | "가장 최근 주문" | argmax(date) → **unresolvable**(retail 전 tool 출력에 날짜 필드 0건 — CENSUS §2a V0 [M]) | ★**ASK 위계 연동**: producer/field 부재 판정 = 결정론 → C48 위계상 ASK가 정답 — 서브콜 결과 `unresolvable`이면 치환·주석 대신 **ASK-유도 피드백**(비커밋)으로 라우팅. 형식화가 "실행 불가"를 *판정*해 주는 것 자체가 t71형의 정확한 처방(현행은 오적용 고착 4/4·C56) |

### §2.3 [[10]]/[[05]] 역할 배분
- [[10]] 정합: **NL→formalize = LLM / concrete 실행 = offload / 검증·실행 = 결정론** — 분담 원칙의 전형 그대로.
  선택 '기준'의 해석은 LLM에 남고(유동성 동결 아님), 기준이 확정된 뒤의 *계산*만 결정론이 대행 = Q3 대행 아님.
- [[05]] 3질문: 엔진 = op 어휘+실행기+JSON 검증(도메인일반·필드 리터럴 0) / A2 = `formalize_sub_args` 화이트리스트
  (+fire할 write-class·기존 `_confirm_write_tools` 도출 재사용) / 대행 ✗ — 조회·write 생성 0·기조회 데이터 위 계산만.

### §2.4 관할 규약 (P-B DISAMB·CALC와의 서로소 분할 — §0c 간섭 해소)
1. **fire 순서**: 결정점에서 ① 사용자 발화/전사에 비교·최상급·제약 술어가 있는 "기준-형식화형"으로 서브콜이 판정
   (`op != none`) → FORMALIZE-EXEC 경로 ② `op: none`(순수 ⋈·기준 없음) → **P-B DISAMB 폴백**(기존 열거-선택)
   ③ `unresolvable` → ASK-유도. **한 결정점 = 서브콜 1회**(FORMALIZE_SYS가 형식화·선택 폴백 판단을 한 콜에 —
   서브콜 비용 배증 금지).
2. **CALC-EXT와의 분할**: 문맥-무관 제약(정적 spec 가능) = calc 주석(기존) / 문맥-의존 제약 = FORMALIZE-EXEC.
   같은 결정점에 calc 주석이 이미 정답을 준 경우(t20의 MATCH 3종) 서브콜 결과와 일치해야 정상 — 불일치 = 계측 이벤트.
3. 서브콜 전사 필터 = T5C N5+N11 승계(`[COMPUTED FACTS]`·`[OPERAND DISAMBIGUATION]` 등 마커 제외 — augment-전 원문만).
4. 치환 후 게이트 재검사·축퇴-스위치 revert(T5C V1 안전장치) 공유. 전 예외 no-op·세션 서브콜 상한 공유.

### §2.5 반대편 계측 (제1원리)
- **형식화-오류 고착(t61형)**: 오형식화 → 결정론 실행 → silent 치환이면 오답이 *확신을 갖고* 고착(user-sim 오확인
  고착과 복리·C56). 계측 = switch 전건 로깅 + gold 대비 정오표 + **Δspurious ≤ 0**(정답-write를 뒤집은 수) 1급 게이트.
  방어 = "불일치 시만 치환"(P-B 비대칭 안전성 승계) + V0 타입별 보수 임계.
- **서브콜 비용**: fire 수·토큰·latency(짧은 생성 1~2회/fire·상한 공유) — Δtme ≤ 0.
- **gold∉C(3.7%·C55)**: 실행기는 후보 밖을 만들 수 없음(닫힌 후보 위 계산) — 단 gold가 후보 밖이면 치환은 무익·
  break 가능 → V0서 gold∉C 슬라이스 분리 계상(T5C B5 편향 명기 승계).
- 주석 채널(2안) 선택 시: 창-오염(C43 재료)·CP 재주입 경로(C61 20%) — 주석은 결정점 직전 1회·비커밋으로 한정.

### §2.6 검증 계획 (격리 재현 → 오프라인 단위 → 표적 nt=1 · §7c 사이클 규율)
| 단계 | 내용 | GO |
|---|---|---|
| **V0 격리 재현(무료·S4/S5 대기 중 가능)** | **★수확 범위 = 전 456 sim(v1.1)** — C클래스 실패 궤적만이 아니라 통과-sim 포함 전체서 기준-형식화형 결정점 수확(E-ISO C 프로토콜 재사용·정보-맞춘). 이유: ① C클래스만으론 희소 타입이 n≥30 **구조적 불충족**(t79 = 1 task×4 trial ≈ 결정점 4개 — 실력이 아니라 표본 부족으로 영구 편입 불가) ② 통과-sim 결정점 = gold 기지 + **over-fire(오발화) 실측** = Δspurious 예고 계측 무료 획득. — ① **형식화 정확도 선측정**: 서브콜 JSON을 per-결정점 gold-기준과 EM 대조(op/field/constraints 분해 채점) ② 실행기 오프라인 적용 → 채널 A/B(fix/break) 대조 ③ 타입별 P(정답\|불일치) — T5C B5 정량 GO 동형(차이>0 ∧ CI 하한>−0.05 ∧ n≥30/타입·**n<30 타입은 pooled 추론+타입별 기술통계 병기로 판정**(Wilson CI 정직 보고)) | 통과 타입만 `formalize_sub_args` 등재 — **불통과면 스택 편입 금지(C23 재발 방지 게이트)** |
| V1 단위 | 실행기 op×constraints 전수(동률·필드부재·빈후보·파싱실패=UNSURE)·치환 왕복·기존 플래그 회귀 | 전부 PASS |
| V2 표적 nt=1 | t20·t37·t79·t71 + 무회귀 대조군(t0·t28형) — **S4/S5 후·승인 후·opt-in env(`T2_FORMALIZE=1`)·stderr 마커 `[T2_FORMALIZE]`** | per-task 복구 ∧ Δspurious≤0 ∧ Δtme≤0 ∧ 위반0 |

---

## §3. 상태·다음 행동 (동결 준수)
- 본 문서 = **[D]·커밋됨(05d71c27·v1.1 별도 커밋)**. 동결의 대상 = **스택 편입·라이브 arm·A2 라이브 변경** —
  오프라인·무료 준비물(단위·census 스크립트·V0 측정)은 스택-불변이라 동결과 양립(§0b.2와 동일 논리·레버1 census도 동형).
- **⚠️작업트리 관찰(2026-07-11 리뷰 시점)**: `gate_interpreter.py`(§1.1① 문자 그대로)·`t2_compliance.py`·`t2_gate.py`
  수정 + `notice_pergate_census.py`가 작업트리에 **이미 존재**(미커밋·본 리뷰가 만들지 않음). 레버1 U/O 단계가
  진행 중인 것으로 보임 — **출처·승인 여부는 사용자 확인 필요**(스프린트 공지 범위는 t81/t2_eplan_patch였음).
  확인 전까지 해당 파일 커밋 금지.
- **착수 시점 = S4/S5 후**(현 스택 동결·사용자 2026-07-11). 착수 순서 = §0b(NOTICE-PERGATE → FORMALIZE-EXEC·
  단 FORMALIZE V0 격리 측정은 무료·스택-불변이라 대기 중 병렬 가능).
- 착수 시 선행 확인: ① 특허 taxonomy 부록 X의 (a)~(f) 정본과 §0a 재구성 대조 ② banking A2 GB2 notice 스펙 실물 확인
  ③ RESEARCH_MASTER §4 큐 등재(두 레버 모두 — E-XGRAMMAR 행과 동일 형식·[D]·동결 표기).
