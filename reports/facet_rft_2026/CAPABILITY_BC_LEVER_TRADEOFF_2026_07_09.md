# 능력 재조합(BC) · 레버 · 트레이드오프 · 최적화 — 정본 설계서 (2026-07-09)

> ⚠️ **이름 폐기 안내**: 이 문서의 BC0-7 코드는 폐기됐다. **통일 정본 = `UNIFIED_TAXONOMY_2026_07_09.md`**(서술형 이름·3축·4도메인). 이 문서는 트레이드오프 매트릭스·최적화 논리의 상세 근거로만 참조하되 이름은 통일 정본을 따른다.

> 관찰 실패기능 G1-G9를 **원인+구제**로 근본능력 BC0-7로 재조합하고, 레버별 트레이드오프와 비용-knee 최적화를 정리.
> ★규율: **G=측정층·BC=처방층**(C54). BC는 [D]설계·§1.5 라우팅의 명시화(새 프레임 아님). [D]/[?]를 [M]처럼 쓰지 말 것.
> 근거: 등대 §1.3·1.5·1.6·§5.1 · `TAU2_FRONTIER_..._MASTER` · `DOMAIN_TRANSFER_STATUS_AND_PLAN`.

## 1. G → BC 재조합 (원인축)
| G (관찰) | → BC (근본능력) | 유형 | 오프로드? |
|---|---|---|---|
| G1 COVERAGE | BC1 ENFORCE(닫힌술어 집행) | decidable | ✅ 결정론 |
| G3 VERIFY | BC1 ENFORCE | decidable | ✅ |
| G5 SCOPE | BC1(precond) + BC4/BC0(순수scope) | **split** | ◑ |
| G2 REACH | BC2 ASSEMBLE(선행자원 조립) | **load** | ✅ controller |
| G6 OPERAND | BC3 COMPUTE + BC4 GROUND | **split·도메인별** | ◑ |
| G7 REFERENCE⋈ | BC4 GROUND(의미 참조) | semantic | ✗ 경계 |
| G4 PERSISTENCE | BC6 STOP-JUDGE + BC1 | **split** | ◑ read-only |
| HORIZON | BC5 SUSTAIN(복리 p^H) | horizon | ◑ scale/분해 |
| G9 GUIDANCE | BC7 DELEGATE(dual-control) | 위임 | ◑ scaffold |
| (전부의 상류) | BC0 TRANSLATE(NL↔형식) | 경계 | ✗ 모델 몫 |
- **쪼개지는 3개(G6·G4·G5)**: 한 기능에 결정론-오프로드 조각 + 능력/경계 조각 공존 → 조각마다 다른 구제. **G→BC split은 도메인별 per-case**(C54: G6 retail=BC4 vs airline=BC3).
- 3계층: **오프로드(BC1·BC2)** = scaffold(도메인-불변·C52) / **레버로 사는 능력(BC3·BC5)** = thinking·scale / **환원불가 경계(BC4·BC0)** = 모트.

## 2. 레버 분류 (= 어느 BC를 사고 무엇을 파나)
| 레버 | mechanism | 사는 BC | 비용항 |
|---|---|---|---|
| **scaffold-족** {게이트·controller·calc·provenance} | 결정론 오프로드 | BC1·BC2·BC3계산·날조 | **S**(~0·재사용) |
| **thinking** | test-time compute | BC3 기호 | 추론 토큰 |
| **learn(T)** | training | BC4/BC3 설치(미검증) | **T**(1회·무망각) |
| **scale(N)** | parameter | BC5 horizon+reach절반 | **N**(per-req) |
| **위임/fleet** | = N 선택구매(전술) | (N) | (N배치) |
| **ABox(A)** | 데이터 공급(능력 아님) | — | **A**(도메인 반복) |
| **경계/ASK** | 수용 | BC4·BC0(레버없음) | 0 |
- 비용순 **S ≺ T ≺ A ≺ N**(§5.1·[[13]]). 위임=N 싸게. thinking=N쪽(추론).

## 3. 트레이드오프 매트릭스 (제1원리: 산다↔판다·§1.3)
| BC | 솔루션 | 산다(+) | 판다(−) | 조건 | 등급 |
|---|---|---|---|---|---|
| BC1 | 결정론 게이트 | 준수·완결·scope 위반 0(pass비용0) | write강제 시 over-action·over-block·게이트 자신 over-action(C4d) | decidable∧미집행 | [S] |
| BC1 | provenance 게이트 | 날조 67→0·e2e +3.3pp | 재발화 예산(微) · ⋈ 못닫음 | 값 출처검증 | [M]GO |
| BC2 | 결정론 controller | reach·부하감축 | 턴예산·과열 tme | load(p_iso>p_traj) | [M] |
| BC3 | calc/offload | 정확 집계·max(상수) | 무엇 셀지 formalize | decidable·조회레코드 | [P]/[S] |
| BC3 | thinking | 기호(격리 .864·Qwen-think F2 0.4%) | **F4·F5 매도**(전궤적)·length크래시 | 기호·결정점격리 | [M] |
| BC4 | 경계-map/ASK | 정직(날조안함) | coverage(기권·p<.5 유해) | 순수참조 | [S]부분 |
| BC4 | learn | 미검증 | 망각·likelihood displacement·off-policy | 데이터 타당성후 | [?]/[M-neg] |
| BC4 | thinking·투표·CoT | +0%(8/8) | — | — | [S-neg]dead |
| BC5 | scale/fleet | horizon(p^H) | 비용R·fleet 3조건·저ROI | horizon-binding | [S-lit]/[M] |
| BC5 | 결정론 분해 | per-step 평탄 | 오버헤드·합성만증명(Hanoi) | 장기절차 | [S-lit] |
| BC6 | persistence 게이트(read) | 조기포기 억제 | **write강제 금지**(abstain→act=p<.5 유해) | 판별상한내 | [M]/[D] |
| BC0 | (환원불가·모델) | formalize·NL생성 | mis-formalize(검증기도 못잡음) | 항상 | [S]경계 |
> **부작용 없는 레버 = BC1 게이트(read-only)·calc(decidable)뿐.** 나머지는 측정된 상쇄로 합성해야 순이득.

## 4. 최적화 (비용-제약·커플링 비분리·경계 바닥)
$$\min \text{비용 s.t. pass}\ge\tau$$
- **알고리즘(탐욕·비용순)**: ①측정 G-질량 → ②G→BC 라우팅(splitter per-case) → ③BC별 S≺T≺A≺N 최저비용 배정(적용조건 통과분) → ④파는 BC 커플링 차감(순<0 기각) → ⑤BC4·BC0 = 잔여 하한(경계).
- **성질**: (a)커플링=비분리(교차항·§1.3) (b)적용게이팅=라우팅(§1.5) (c)경계 바닥=환원불가 잔여=모트 (d)read-only 하드제약.
- **solver 아님·결정 스캐폴드**: 계수 대부분 [M]/[D] → 출력 = 도메인별 레버 우선순위 + 예측 잔여바닥 + 측정필요 [D] 교차항 목록. Phase 4가 계수 채움.
- **좋은 성질(C52)**: BC1·BC2 scaffold=도메인-불변→고정비 1회 amortize(ABox swap). BC3·BC5=도메인별. ⇒ 최적배치 = 불변 scaffold 1벌 + 도메인별 능력레버 = **TBox+ABox 구조**.

## 5. 워크드 예 (retail·측정 G)
G1 52%→BC1 게이트(회복) · G6 30%→BC3 calc(무료)+BC4 thinking(−coverage) · G2 18%→BC2 controller · G4 10%→BC6 게이트 · **G7 9%→BC4 없음=바닥** · G5 5%→precond(env)/경계.
→ **retail 잔여 하한 ≈ BC4(G7+G6변형+G5scope) ~15-20%**; 나머지 ~70%는 scaffold+thinking 회복. = 준수-drop 모트 정합.

## 6. 상태·다음
- BC/레버/트레이드오프 = **[D] 설계**(측정 backbone = C51·C52·C54). 처방 전이 실측 = `DOMAIN_TRANSFER_STATUS_AND_PLAN` Phase 3.
- 특허: B(레버 partition·knee) · A(능력분해→오프로드·교차도메인 불변). 덱: 프레임 슬라이드.
