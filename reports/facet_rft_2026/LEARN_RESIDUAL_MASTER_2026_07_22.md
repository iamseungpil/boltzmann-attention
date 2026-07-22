# LEARN 잔여 통합 정본 (2026-07-22)

> **이 문서가 learn 축의 단일 정본이다.** 상위=`RESEARCH_MASTER.md`(등대). learn 관련 설계문서 9개를
> 통합하고, **실험으로 종결/흡수/강등된 것은 §1로 격리**(재개 트리거만 보존)·**살아있는 잔여만 §2에
> 활성 유지**한다. 규율: **[[11]]** 학습은 학습벤치(synth)서만·banking=eval 전용(ABox-swap 전이) ·
> **[[12]]** 표현/구조 다양성 필수 · **[[13]]** 흡수순서 scale→learn→scaffold(최후) · **[[05]]** 엔진 도메인일반 ·
> **[[09]]** 무료검증 우선 · Δspurious≤0 · gold-independence.
>
> **통합 대상 문서(9)**: E6PRIME_GATHER_LEARN_DESIGN·C38_INDIST_GATHER_RESULT·E11_GATHER_BEFORE_ACT·
> INFER_CALIBRATION_LEARN_DESIGN·VALUE_GROUNDING_PLACEHOLDER_LEARN_DESIGN·COMPLETION_EVIDENCE_LEARN_DESIGN·
> LEARNED_WING_MECHANISM_DESIGN·SEMANTIC_NONSCALE_METHODS·GENERALIZED_SCAFFOLD_ARCHITECTURE(§4c 흡수 판정).
> 각 문서는 **보존**(원 provenance)·본 정본이 **활성 상태 판정의 정본**.

---

## 0. 한눈에 — learn 지형 (2026-07-22 실측 확정)

**핵심 발견: "learn 잔여"의 대부분은 이미 종결(실험)·흡수(결정론 scaffold)·강등(전제붕괴)됐다.** 개입레버를
`(구조=A2/엔진) + (의미=learn/ASK)`로 분해한 뒤(GENERALIZED_SCAFFOLD), **의미-선택(disamb)은 ≥2→ASK로,
값-grounding은 근거요구 게이트로 결정론에 흡수**됐다. 남는 활성 learn 잔여는 **3개뿐**이며 전부 "판단/계획"
층이다. rall11(§2bu)이 "cap-소진·환각접미사·검색마찰"의 기계적 잔여를 대거 소멸시켜 **잔여를 learn 경계로
밀어붙인 것**이 이 정리의 계기다.

```
제외(종결/흡수/강등·§1)                          활성 learn 잔여(§2)                learn 못 닫는 경계(§3)
─────────────────────────────                    ──────────────────────           ────────────────────
완료-증거 날조    [게이트FAIL·NOGO]               ① formalize(FIND) 정확도         ⋈ systematic 8/8동일오답
disamb-select    [≥2→ASK 흡수]                    ② 순서-계획(054 crossover)        genuine 표현-애매성(over-ask)
값-grounding     [WRITE_ARG_GROUND 흡수]           ③ gather 재시험(D1~D4 대기)       state-track scale-gated≥13B
집계/argmax      [CALC/결정론]                                                       └→ 7B는 결정론 controller
INFER-calib      [강등·095=gather·097미부활]
gather-cfbsynth  [측정무효·base 0.98]
present / E9     [폐기 / 환경집행 죽은레버]
```

---

## 1. 제외 — 실험으로 종결·흡수·강등된 축 (재개 트리거만 보존)

| 축 | 판정 | 결정적 실측(verbatim) | 재개 트리거 | 원문서 |
|---|---|---|---|---|
| **완료-주장 evidence** | **[종결-NOGO]** | 게이트 자동발사 `0/36=0%`(합격선 30%·banking 라이브 54%)=**합성이 실패 재현 못함→gradient 0→학습 불가**. +논문 코어 하차(4겹). | "왜 0인지(방아쇠·다양성) 규명" 선행 시에만 | COMPLETION_EVIDENCE |
| **INFER-calibration**(값 후보-선택) | **[강등·보류]** | 095 부하-격리 `p_iso=0/9`(격리해도 오답)·presence 선검사: gold 값이 문맥에 **아예 없음** ⇒ 095=INFER 아니라 **gather 미완**. **097로도 미부활**: 097 잔여=grounding[scaffold완료]+산술[결정론 L1]이지 후보-오선택 아님. | 격리된 **INFER-증거 태스크**가 실측 확보될 때(=값이 문맥에 실재하는데도 오선택) | INFER_CALIBRATION |
| **disamb / criterion-variant 선택** | **[scaffold흡수]** | GENERALIZED_SCAFFOLD §4c: GET→FIND→**1개면 사용·≥2면 ASK**로 INFER(유효후보 중 추측=오류원) 삭제 ⇒ ASK-calibration이 결정론 규칙으로 소멸·틀린추측0. | — (흡수 완료) | LEARNED_WING §0.5·GENERALIZED_SCAFFOLD §4c |
| **값-grounding**(인자 값 실재) | **[scaffold흡수]** | `T2_WRITE_ARG_GROUND`(fix4)=선언 기록-값 인자가 도구출력∪user발화에 실재해야 write → **031 승격(2/5→3/5)의 축**. operand grounding A2 `14/14 PASS`+라이브 발화. | — (게이트가 집행·학습 불요) | VALUE_GROUNDING §1·RATE §2af/§2bu |
| **집계·argmax·most-recent(도구有)** | **[CALC/결정론]** | well-defined 쿼리는 DB에 결정론적으로 존재·`check_rebate/closure/cli_eligibility` scaffold(§2bu fix9). LLM INFER 금지. | — | GENERALIZED_SCAFFOLD §8 |
| **gather-before-act(cfbsynth 데이터)** | **[측정무효]** | cfbsynth가 *"I don't have the id"* 큐를 **150/150(100%)** 제공(tau2=120중 1건)+규칙 명시 ⇒ base 이미 **0.98**=gradient 없음. DPO off-policy(지지집합 밖)·SFT 첫행동 lookup 2000/2000=퇴화정책(tme 13→25). | **§2.3 재시험**(결손-보존 데이터)=**살아있는 축 §2.3으로 이관** | C38·E6PRIME |
| **present(정보 주입)** | **[폐기]** | pass +4.7pp 사지만 order조회 2.62→0.48(5.5×억제)·미조회날조 5.6%→10.4%. C31: read→act 지워 **학습신호 파괴**. | — (baseline=floor 확정) | E11 §1 |
| **E9 id-날조 차단** | **[죽은레버·환경집행]** | 환경이 이미 거부(C12 `93/93`). | — | E11 §7 |

---

## 2. 활성 learn 잔여 (make-or-break 순 — 이것만 살아있다)

### 2.1 formalize(FIND) 정확도 — ★유일 semantic 잔여·최우선

**정의**: LLM이 NL 제약 → predicate로, NL 값 → 유효집합 내 정확한 항으로 변환하는 충실도([[10]] 생성기 몫).
GENERALIZED_SCAFFOLD §4d/§5가 "**유일 semantic 잔여·make-or-break**"로 확정.

**실패 기제(왜 scaffold가 못 잡나·[S])**: ≥2→ASK 규칙은 **엔진이 GET/FIND 산출집합에서 ≥2를 셀 때만** 발동한다.
그런데 **오/과엄격 형식화가 유효집합을 1개(틀린 것)로 붕괴**시키거나 **유효집합 *밖* 단일값**을 confident-emit하면
≥2 트리거가 안 걸려 **silent-wrong**이 통과한다. evidence-quote 게이트는 값-날조는 잡지만 predicate-오형식화는 못 잡음.
- 렌즈: t106 `INFER("smaller"→XL)` — 모델이 1값 confident-emit·GET-집합에 애매성이 후보로 안 뜸 → ASK 미발동.
- rall11 부분귀속: **038 처방-선택**(dispute↔statement-credit↔close·"eligible circumstances ONLY in..." KB doc_017)
  = 조건 formalize(어느 절차가 이 상황에 적용되는가)의 한 형태. 038은 close·credit·dispute를 오선택하며 이 축을 실증.

**learn 경로**: 학습벤치서 NL제약→predicate·NL값→유효집합-항 매핑을 **다양성**([[12]])으로 SFT. 출력에 `{value,
evidence_quote}` 동반(§4b)해 엔진이 quote∈source를 검증(값-충실도는 흡수·형식화 정확도만 남김). on-policy rejected(C38).

**상태**: 설계 확정·데이터 미제작. 권위본=GENERALIZED_SCAFFOLD §4c-d/§5.

### 2.2 순서-계획 (order-dependent write / plan-first) — ★신규(054)·crossover 경계

**정의**: 한 대화의 여러 요청 중 **요청 A의 부작용이 요청 B를 막는** 경우, 실행 전에 상호작용을 따져
심사-완결형을 먼저 배치하는 스킬. = frontier PASS 공통 습관(§2bg: "절차 정독→플랜→연속 완주").

**실측(054·[S]·§2bu·§2bt)**: env 히든룰 "pending dispute/replacement면 CLI approve 거부"(tools.py 확정).
고객이 "dispute 먼저"를 시키는데 gold 순서는 CLI-완결→replacement→dispute. **dispute를 먼저 접수하면 그 sim의
CLI는 회복 불가.** 이 규칙은 **KB 698문서·도구설명·에러문구 어디에도 없음**(census·역grep 0)=agent-가시 채널 전무.
- **scaffold 불가 판정**: env소스→A2 이식=[[03b]] cheating. scaffold 살 수 있는 것 다 삼(액션 4/17→**16/17**)·잔여=이 순서 1건.
- **frontier도 대부분 못 풂**(§2bg: 054 frontier 전멸·최고 1/4). 그 1/4이 사는 것=①은행 실무 **상식-prior**(scale)+②**계획-먼저 습관**.

**learn 경로**: 054를 가르치는 게 아니라([[11]] 도메인-타깃 금지) **"순서-의존 write-쌍" P-primitive를 학습벤치에**
구성(요청 A 부작용→요청 B 차단 구조·다양한 도메인 표면)하고, "다중 요청 수령 시 실행 전 상호작용 검토→심사-완결
우선 배치" 행동을 SFT로 설치. frontier의 ②습관을 작은 모델에 이식. **①상식-prior 자체는 scale 몫**(정직한 한계).

**상태**: 신규 축·rall11 실측 확보·설계 초안 단계. E-PLAN([[14]])의 learn 표적과 직결(controller=순서 집행·learn=순서 판단).

### 2.3 gather-before-act 재시험 — 데이터 재설계 대기(결손-보존)

**정의**: 값이 없으면 write 전에 조회하는 스킬. cfbsynth로는 **측정 무효**(§1)이나 **진짜 결손 위에서 시험된 적 없음**
= 살아있는 잔여(C38 §3). E6′ 설계 골격 유효·데이터만 C38 §4(D1~D4)로 교체.

**재개 데이터 처방(D1~D4·C38 §4)**: D1 *"I don't have X"* 큐 제거(결손을 사용자가 명시하지 않게)·D2 규칙문장 제거
(base가 0.98로 이미 obey하지 않게)·D3 **on-policy rejected**(우리 32B를 실 결정점서 샘플·off-policy DPO=likelihood
displacement 실패)·D4 rejected 3종 확장{예시값·발명형 id·조합형 placeholder}(32B 실패양식=발명 48/93이지 예시복사 아님·C39).

**직렬 사슬(scaffold↔learn 협업)**: E11-a 결정론 gather 게이트(위반→강제read→올바른write)가 **감독신호(라벨)를 생성**→
그 궤적이 learn 감독신호→내면화 후 게이트 제거. 게이트와 learn은 경쟁 아니라 직렬(E11 §5). E11-a는 Phase B/C 본실행 대기.
- rall11 연결: **097 coverage**(4계좌 중 1개만 apply·값도 95 vs gold 12)=gather/plan 미완+산술(L1)의 복합→E-PLAN/gather.

**상태**: 데이터 재설계 처방 확정(D1~D4)·미제작. 권위본=C38 §4·E6PRIME.

---

## 3. learn으로 "못 닫는" 경계 (scale/ASK — 완결성 위해 명시·[[13]] 경계)

learn 잔여가 아니라 **learn·scaffold 둘 다 못 닫는** 것. 논문 crossover의 반대편([46]).

- **⋈ systematic 동일오답**: self-consistency **+0%·8/8 만장일치 오답**(분산0·오답=mode)=RL-unreachable(LEARNED_WING §7·
  C88). voting=분산 레버라 systematic엔 vacuous. → **verify-or-ASK / scale / 경계 수용**.
- **genuine 표현-애매성**: t106 "one size smaller"=XL?S? — 사용자 발화 자체가 미결정. ASK로 경계 처리는 안전하나
  **over-ask 비용 미측정**(GENERALIZED_SCAFFOLD §4c caveat).
- **state-tracking 학습설치 = scale-gated ≥13B**(SEMANTIC_NONSCALE §1.2: Code Llama 7B 13.7 vs Llama-2 7B 15.0)
  → **7B τ²는 learn 대신 결정론 controller로 우회**([[13]] 순서의 외부 근거). genuine load-induced semantic crossover는
  문헌서도 미검=**우리 실험이 채울 whitespace**(살아있는 semantic 영역이나 방법이 아니라 실증이 whitespace).

---

## 4. 공통 방법·데이터 규율 (모든 활성 축)

1. **데이터 타당성 게이트(선행 필수)**: base 32B **실 실패 확인** → 합성이 그 실패 **재현 확인**(banking 실측률 대비) →
   통과해야 학습. COMPLETION_EVIDENCE가 이 게이트에서 FAIL(0/36)=재현 실패로 종결된 것이 규율의 실증.
2. **학습벤치서만·ABox-swap 전이**([[11]]): banking=eval 전용. 도메인-타깃 학습 금지.
3. **다양성**([[12]]): 단일 템플릿 SFT=표면매핑 역전이. 표현·구조 다양성이 전이 필수.
4. **on-policy rejected**(C38 D3): off-policy DPO=지지집합 밖 마진=likelihood displacement. 우리 32B 샘플에서 rejected 추출.
5. **흡수 순서**([[13]]): scale→learn→(최후)scaffold/A2. 본 정본의 §1 흡수분이 이 순서의 역방향 검증(먼저 scaffold가
   기계적 잔여를 흡수→learn 표적이 판단/계획으로 정제됨).

---

## 5. 우산 개념 (논문 프레임·정본 서사)

- **citation-carrying tool use**(VALUE_GROUNDING §7): 모든 도구-인자 값+사용자-대면 주장에 인용 동반 = 논문 코어 서사.
  실체는 **입력측(값 grounding)=scaffold 흡수 / 출력측(완료-evidence)=종결 / 형식화 정확도=활성 learn**으로 분해됨.
- **crossover(모트·[46])**: 054가 실물 실증 — scaffold가 살 수 있는 것(4/17→16/17)과 살 수 없는 잔여(발견불가 의존성=
  상식-prior+계획습관=scale/learn)가 실측으로 갈라짐. "scale-불변 잔여 + non-scaling 레버"의 양면.
- **위계**: VALUE_GROUNDING(우산) ⊃ {INFER_CALIBRATION=입력측[강등], COMPLETION_EVIDENCE=출력측[종결]}. 세 문서 중
  활성 learn 잔여는 없고(전부 §1), 우산 개념과 재개 트리거만 생존.

---

## 부록 — 통합된 9문서의 활성 상태 매핑

| 원문서 | 활성분 → 위치 | 사문(死文)분 → §1 |
|---|---|---|
| GENERALIZED_SCAFFOLD_ARCHITECTURE | §2.1 권위본(formalize 잔여)·§4c 흡수판정 | — (현행 LOCK) |
| E6PRIME_GATHER + C38 | §2.3(데이터 D1~D4 재시험) | cfbsynth 측정=무효 |
| E11_GATHER_BEFORE_ACT | §2.3(E11-a 게이트=감독신호) | present 폐기·E9 죽은레버 |
| LEARNED_WING_MECHANISM | §3(⋈ RL-unreachable)·reachability=RL 필요조건 원리 | criterion-select=≥2→ASK 흡수 |
| SEMANTIC_NONSCALE_METHODS | §3(scale-gate ≥13B·crossover whitespace 인용앵커) | self-critique/neuro-symbolic=인용금지 |
| VALUE_GROUNDING_PLACEHOLDER | §5(우산·논문 프레임) | grounding=scaffold 흡수·학습코어 보류 |
| INFER_CALIBRATION | — | §1(강등·095=gather·097 미부활) |
| COMPLETION_EVIDENCE | — | §1(종결-NOGO·게이트 0/36) |
