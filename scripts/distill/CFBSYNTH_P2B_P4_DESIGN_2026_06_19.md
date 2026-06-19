# CFBSYNTH 설계 — CFB-참조 추상 synth로 P2b(연속홉)+P4(선택) 학습 (2026-06-19)

> **자립 문서**(리뷰용). 권위 = `PRIMITIVE_COVERAGE_MATRIX_2026_06_15.md`(P1-P9 정의·§87-88 커버리지)·`fc_randomize_fetchable.py`(v6 fetch-강제 메커니즘)·`ma/synth_to_nativefc.py`(synth 생성 선례). 메모리 = [[00-thesis]]·[[03-anti-drift]]·[[05-fixed-vs-variable]]·[[11-transfer-direction]]·[[12-diversity-required]].
> 구현(설계 후 검증) = `ma/synth_fetch_nativefc.py`·`tau2/build_solo_data_cfb.sh`·`build_solo_train_cfb.sh`.

## 0. 문제 (전수 진단 → primitive 매핑)
retail 실 e2e(step5 full-40)서 **tool 호출 63% 에러·그중 73%="Order not found"(405회)·52/80 sim 날조.** 전수 궤적: 에이전트가 `get_user_details`(주문목록 생산)를 건너뛰고 **order_id를 날조**(`#W0000000`)→get_order_details 호출 실패. resolve_selection은 19회 실행·**100% grounding 실패**(상류 미fetch라 도달조차 못 함).

**primitive 매핑**(권위본 정의): retail `get_user_details→[목록]→order_id 선택→get_order_details(order_id)` =
- **P2b** = gather-for-arg(2-hop): lookup 출력 → 하류 인자 (연속홉).
- **P4** = select-from-output: 관찰 리스트서 옳은 항목 추출 (cardinality>1).
- 날조 = **¬P1** (실패모드·P1 위반).

**근본원인**(권위본 §87-88): τ² 필요 = {P1,P2b,P3,P4,P5,P6,P7,P8}. **P2b·P4는 오직 CFB만 공급.** 그런데 solo LoRA = synth+sopbench+taskbench·**CFB 제외** → P2b/P4 미학습 → 연속홉 fetch 붕괴. ("flow 풀렸다"=P1/P3/P5/P8 / "operand서 막혔다"=P2b/P4 — 둘 다 사용자 history와 정합.)

## 1. 접근 — 왜 CFB 직접이 아니라 *추상 synth*
- **CFB 직접 = 표면매핑 위험**([[12-diversity-required]]·[[11-transfer-direction]]): 단일-도메인 CFB 궤적 SFT는 표면 어휘를 암기→역전이(C8 synth1.00→τ²0.03·facet3 역전이 선례). CFB의 P2b/P4 전이도 "v7 eval 전 미확정"(권위본 §72).
- **추상 synth = 구조만**: P2b/P4의 *도메인-무관 dataflow 구조*를 합성. 익명 툴/필드·per-traj randomized id·표현 다양성 → 표면 0·구조만 학습. synth content-op(§28 held-out 1.00)이 이 방식으로 작동한 선례.
- **"가장 추상적인 부분만"**(사용자 지시) = 모델이 배우는 것 = **"없는 값은 생산 tool 호출→그 출력서 *복사*·리스트면 기술된 것 선택·날조 금지"**라는 *규율 SHAPE*뿐. 도메인 술어(어느 getter가 무엇 생산)는 학습 아님(§9 참조).

## 2. 추상 구조 (생성기 설계)
```
user: "<action> the item where <attr>=<val>. My <key>=<v>. I don't have the <id>."   (id 미제공)
 → [P2a/P1] getter(key=<user서 복사>)
 ← tool: [{id:<rand>, attr:<x>}, {id:<rand>, attr:<val>}, ...]   (리스트·하나만 attr 일치)
 → [P4] attr=val record 선택  → [P2b/P1] consumer(arg=<그 id, 출력서 복사>)
```
- **id randomize(per-traj)**: gold id가 *오직* getter 출력에만 존재 → 암기 불가 → **복사만 가능**(P2b 강제·v6 fc_randomize_fetchable 메커니즘 계승).
- **익명 툴/필드명(per-traj)**: 구조만·lexical 암기 차단.
- **다양성**: hops∈{2,3}(3홉=getter1 출력→getter2 키→consumer)·list_n∈{1..5}(1=순수 P2b·>1=P2b+P4)·attr/id/val 포맷 변주.
- **consumer arg명 ≠ id 필드명**: schema 기술로 매핑(표면 단축 차단).
- 출력 = native-FC(synth_to_nativefc 동형)·`_meta.bench=cfbsynth`.

## 3. ★Stratum 분리 — COPY vs COMPUTE (resolve_selection과 충돌 없음·thesis 핵심)
앞 진단서 "synth(resolve)가 'id emit 마라'를 가르쳐 flow grounding 간섭"이 우려였음. 해소 = **두 stratum은 *다른 연산***:
| | cfbsynth (P2b/P4-filter) | resolve_selection (content-op) |
|---|---|---|
| 연산 | **COPY**: 관찰된 값 복사 | **COMPUTE**: 카탈로그 위 선택 계산 |
| 예 | 주문목록서 그 order_id | 가장 싼/N번째 item (argmax/rank) |
| 깊이 | 얕음(매칭·복사)=in-head 가능 | 깊음(235B도 0.02@N50)=offload |
| 모델 행동 | **id를 emit**(복사) | id emit 안 함(엔진 grounds) |
| primitive | P2b·P4(filter)·P1 | P4(argmax/rank)·content 8-op |
- **경계**: P4가 *단순 매칭(filter)*이면 COPY(학습)·*순서/집계(argmax/rank)*면 COMPUTE(offload). 둘은 P4의 두 갈래라 모순 아님 — cfbsynth=filter측만·resolve=argmax측만. **모델은 "값이 관찰됐으면 복사, 계산이 필요하면 op 명명"을 *구분* 학습.** (이게 두 데이터 합본의 정당화.)

## 4. Thesis-정합 자가심사 ([[03-anti-drift]] 규칙7)
- 학습=도메인-일반 규율(P2b/P4 SHAPE·copy-grounding)·전이=ABox-swap·측정=실 e2e. ✅ ([[00-thesis]])
- base+bespoke 우회 아님(학습된 TBox). ✅
- synth=통제 벤치로 유지·τ²만 안 봄. ✅ (규칙5)
- 고정={TBox+Scaffold}/변경={ABox}: "어느 getter가 값 생산"=도메인 사실=ABox(§9). ✅ ([[05-fixed-vs-variable]])

## 5. 치팅/역전이 공격면 자가심사
- **암기 우회**: id randomize → 복사만 가능. ✅
- **lexical 단축**: 툴/필드/arg명 per-traj 익명·arg명≠id명. ✅
- **표면 템플릿 역전이**(C8 교훈): 표현·hop·list_n·포맷 다양성으로 단일템플릿 회피. ⚠️ *충분한가*는 실 e2e 전이가 판정(미확정).
- **P4 cheating**(엔진이 대신 고름): cfbsynth는 모델이 *직접* 선택·복사(offload 아님)=의도. ✅
- **provenance 보존**: user 발화에 있는 값(key)은 user-제공이라 fetch 강제 안 함(v6 규율). ✅

## 6. 측정 (실 e2e만·[[03-anti-drift]] 규칙)
- **★FAB(not-found 날조)율** = order/item "not found" tool 에러 수·sims_fab/n. = order_id 날조 직접 proxy(근본지표).
- **fetch-first**: get_user_details 등 producer 선행 호출 비율.
- **pass^1** vs base 0.205 (분산 ±2-3·multi-trial).
- **A/B**: solo_lite(cfb 없음·baseline) vs solo_cfb(cfb 추가)·1-변수. ckpt-at 곡선으로 망각도 동시.

## 7. Falsifiable 예측 + GO/NO-GO
- **H**: cfbsynth 추가 → FAB율 baseline 대비 ↓(fetch-first ↑) → resolve_selection이 grounding할 상태 도달 → pass ↑.
- **GO**: FAB율 유의 감소(예 sims_fab 65%→<40%) + pass Δ≥3-4(또는 적어도 FAB↓로 resolve grounding 성공률↑).
- **NO-GO/반증**: FAB율 불변 → 추상 synth가 P2b/P4 전이 실패(C8식 역전이 재발) → ABox 의존맵+결정론 가드로 선회(§9·학습 아닌 제공).

## 8. 위험 / 미결
- **전이 미확정**: 추상 P2b/P4가 retail 2-홉에 전이한다는 보장 없음(권위본: CFB조차 미확정). 이 실험이 *첫* 검증.
- **A2 의미층 잔존**: "order_id는 get_user_details가 생산"이라는 *도메인-의미 의존*은 추상 synth가 못 가르침(어느 getter인지=도메인 사실). cfbsynth는 *구조*(있으면 fetch·복사)만. 의미 의존은 §9.
- **합본 균형**: cfbsynth 6000 = synth/taskbench와 비등 → 과대표현 시 다른 primitive 희석 가능(곡선서 감시).
- **forgetting**: 데이터 추가가 망각 재유발 가능(ckpt-at 곡선·mid-layer 옵션 보유).

## 9. 학습/결정론 경계 — A2(어느 getter)는 ABox로
cfbsynth가 가르치는 것 = **구조적 규율**(P2b/P4 SHAPE·copy-no-fabricate). 가르치지 *못하는* 것 = **도메인-의미 의존**(retail서 order_id의 생산자=get_user_details). 후자 = 도메인 사실 → **ABox 의존맵**(value-type → producer-tool)으로 제공([[05-fixed-vs-variable]]). + **결정론 provenance 가드**(상류출력/user에 없는 id로 호출 시 거부·resolve 패턴을 order_id로 확장)로 backstop. ⇒ 분담: **구조 규율=학습(cfbsynth) / 의미 의존=ABox / 집행=결정론 가드.** 사용자 다층 모델(operand 규칙 하위루틴)의 구현 형태.

## 10. 리뷰 안건
1. COPY/COMPUTE 경계(§3)가 P4를 깨끗이 가르나, 아니면 filter/argmax 혼동 위험?
2. A2 의미의존을 ABox로 빼는 게(§9) 충분한가, 아니면 일부 학습 필요?
3. cfbsynth 다양성(§2)이 C8 역전이를 피하기에 충분한가 — 실 e2e 전 알 수 없음(H7 GO/NO-GO가 판정).
4. 합본 비율(6000)·forgetting 곡선 감시 지점.
