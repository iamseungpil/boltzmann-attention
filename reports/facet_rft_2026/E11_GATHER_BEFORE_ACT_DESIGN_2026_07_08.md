# E11 — gather-before-act 게이트 · present 폐기 (2026-07-08 밤)

> 상위 = `RESEARCH_MASTER.md`. 근거 = `DB_ONLY_HARDCORE_FORENSIC_2026_07_08.md` §10 (C29~C31).
> 사용자 확정(2026-07-08): **present는 gather의 효과를 임시로 근사해 본 프로브였다. 정식 스택에서 폐기한다.**
> 재현: `scripts/distill/tau2/dbonly_forensic.py`, 본 doc §1 표는 `data/simulations/on_n32int8_*` 4-arm.

---

## 0. 한 줄
**날조는 gather 실패의 하류 증상이고, present는 그 실패를 *제조*한다.** 정보를 주입하지 말고 조회를 강제하라.

## 1. ★present 장부 (32B-int8 동일 base · t3 · clean sim만)

| arm | sim | pass^1 | db_pass | **order 조회/sim** | product 조회/sim | **미조회 날조** | over-action sim |
|---|---|---|---|---|---|---|---|
| floor (맨몸) | 342 | 0.547 | 0.596 | **2.62** | 0.79 | **18 (5.6%)** | 20 |
| present 단독 | 337 | 0.576 | 0.626 | 1.29 | 0.77 | 22 (6.9%) | 11 |
| g15 단독 | 329 | 0.550 | 0.578 | 2.44 | 0.78 | 19 (7.0%) | 8 |
| **present+g15** | 323 | **0.594** | **0.678** | **0.48** | 0.76 | **38 (10.4%)** | 8 |

- **산다**: pass^1 **+4.7pp** · db_pass **+8.2pp** (present+g15 vs floor).
- **판다**: order 조회 **2.62 → 0.48 = 5.5× 억제** · 미조회 날조 **5.6% → 10.4% = 1.9×**.
- **기전**[D]: present는 *주문*을 떠먹여 읽기 습관을 죽이지만 *변형 id*는 주지 않는다(product 조회 0.79→0.76 불변).
  모델은 그 빈자리를 **날조로 메운다**(`6117189161_cheapest`, `6117189162`).
- ⇒ **present는 frontier 격차의 83%를 차지하는 실패(C29)를 스스로 만들어낸다.** 국소 최적이며 진짜 수리를 막는다.
- ⇒ **C31**: present는 궤적에서 `read → act`를 지워 **learn-wing(P4·C7)의 감독 신호를 파괴**한다.

> **결정: present 폐기.** 이후 모든 arm의 baseline = **floor**.

## 2. 왜 scaffold-차단이 아니라 gather인가 (§1.5 절차 정직하게 다시 밟기)

| 단계 | 판정 |
|---|---|
| Q1 술어가 decidable한가 | **yes** — "이 operand를 산출한 선행 read 결과가 있는가" |
| Q1b 이미 집행 중인가 | **부분적으로 yes** — 환경이 *유효하지 않은 id*를 **93/93 거부**(C12) |
| ⇒ | **"날조 차단" 게이트는 죽은 레버**(E9 NO-GO 재확인). 환경이 이미 한다. |
| 그러나 | **"행동 전 조회" 술어는 미집행**이고, 이것은 *차단*이 아니라 **행동 유도** |
| Q4 scale이 사는가 | **비단조** — 7B 38.8% → 14B 7.0% → 32B(+present) 14.4% / floor 14B 9.2% → 32B 6.7% · frontier **0.0~0.3%** |

⇒ 우리 사다리 안에서 scale은 7B→14B 구간만 산다. 32B→frontier의 **불연속(6.7% → 0.0%)** 은 open big-tier 외삽으로 못 메운다([EST]).
**남는 길 = (a) 결정론 gather 게이트 (b) gather-before-act를 학습(TBox·[[00]]·[[11]])** — 그리고 둘은 경쟁이 아니다(§5).

## 3. 술어 (도메인-일반)

> **write 호출의 모든 entity-id·free-text operand 값은, 그 값을 산출한 선행 *도구 출력*에 문자열로 존재해야 한다.**
> (사용자 발화도 provenance로 인정 — 사용자가 새 주소를 불러주는 경우.)

A2(도메인별)는 **매핑 하나뿐**: `{write 인자 → 그 타입을 열거하는 read 도구}`

| write 인자 | 열거 read |
|---|---|
| `new_item_ids` / `item_ids` | `get_product_details(product_id = 해당 item의 제품)` |
| `order_id` | `get_user_details` |
| `payment_method_id` | `get_user_details` |
| `address1`/`city`/`state`/`zip` | `get_order_details` 또는 `get_user_details` |

[[05]] 준수: 엔진(술어·집행)은 도메인-일반, A2(매핑)만 도메인별. present처럼 **도메인 내용물을 주입하지 않는다**.

## 4. 두 변종 (제1원리: 둘 다 무언가를 판다)

| 변종 | 동작 | 산다 | 판다 |
|---|---|---|---|
| **E11-a (hint)** | write 거부 + **어떤 read를 부르라고 지시** | 에이전트 주도성·학습신호 보존 | 턴 |
| **E11-b (supply)** | 해당 read를 **자동 실행**해 결과를 반환 | 정보 보장 | 읽기 주도성(present의 병) |

**E11-a 우선.** 환경의 기존 거부 메시지(`Variant not found`)는 *무엇을 해야 하는지* 말하지 않는다 —
E9 PhaseA에서 날조 12/15가 거부 후에도 실패한 이유가 이것일 수 있다(가설·E11-a가 검정).
E11-b는 E11-a가 실패할 때의 fallback이며, **성공해도 C31(학습신호) 비용을 진다.**

## 5. scaffold ↔ learn (경쟁 아님)

[[13]] 흡수 우선순위는 **scale → 학습 → (최후) scaffold**다. 그런데 여기서 두 경로는 **직렬**이다:

1. **결정론 gather 게이트가 위반 지점을 정확히 라벨링한다** (E11 발화 = 지도 라벨).
2. 그 궤적(`위반 → 강제 read → 올바른 write`)이 **E6 learn-wing의 감독 신호**가 된다.
3. 학습이 내면화하면 게이트를 떼고 **전이는 ABox-swap**([[11]]). 이것이 **P4**의 실증.

⇒ **present로는 이 사슬이 불가능하다** — 주입은 `read → act`를 남기지 않으므로 배울 것이 없다(C31).

## 6. 측정 계획

### Phase A — 오프라인 검출 (무료·완료)
기존 궤적에 술어를 적용. clean sim만.

| | E11 발화 sim | **db_fail 중 발화 = 상한** | db_pass 중 발화 = **Δspurious 위험** |
|---|---|---|---|
| **32B floor** | 50 / 456 (11.0%) | **40 = 8.8pp** | 10 |
| 32B +present스택 | 44 / 456 (9.6%) | 23 = 5.0pp | 21 |

floor 발화 사유: `new_item_id 변형목록 미조회` 30 · `payment_method_id` 26 · `address1` 21 · `order_id` 4.
※ present 스택에서 address 발화가 21→5로 주는 것은 present가 주소를 주입하기 때문이다 — 즉 present는
**주소 provenance를 사고 변형 provenance를 판다**(30→64). 총 발화는 비슷하고, **db_pass 발화(무해한 날조)만 늘린다**.

### Phase B — 기전 스모크 (무료·GPU0 14B)
게이트가 올바른 지점에서 발화하고 over-block=0인지. **agent=14B**(GPU 여유상 32B 불가·Step3가 GPU1 점유).
목표: 크래시 0 · 발화 위치 per-case 일치 · 정상 궤적 오발화 0.

### Phase C — 본 측정 (Step3 종료 후·GPU0에 32B-int8)
**arm: floor vs floor+E11-a (vs floor+E11-b)**. user-sim은 먼저 14B(무료), 결론 선 뒤 gpt-4.1(승인·[[09]]).

**GO 조건** (등대 §4 공통 + 본 레버 고유):
- per-case 복구 ∧ over-block 0 ∧ **Δspurious ≤ 0** (db_pass 10 sim에서 파손 0)
- **턴 예산 초과 0** (제1원리: 강제 조회는 턴을 판다)
- **reads/sim이 frontier 쪽으로 이동**(floor 2.62 → ?; frontier order 조회는 훨씬 많다)

## 7. 폐기·정정 기록
- **present**: 정식 스택에서 제외. 과거 결과(C4a `present+g15 +12.3pp`)는 **유효하나 국소 최적**이며, 그 이득은
  **날조 1.9× 증가와 읽기 5.5× 억제를 대가로 산 것**임을 함께 인용해야 한다.
- **E9′(free-text provenance)**: E11에 흡수(주소는 같은 술어의 특수경우).
- **E9(id 날조 차단)**: 죽은 채로 유지. 환경이 이미 집행(C12).
