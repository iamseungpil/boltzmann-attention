# A′ 회귀 포렌식: aprime1→aprime2 t86·t95 pass→fail 판정 (2026-07-11)

**질문**: aprime1→aprime2 사이 유일한 코드 변경(①`T2_EPLAN_DENY_CAP=4` ②품목-나열 힌트 `_enum_items`, 커밋 `00fa5d23`)이 t86·t95의 1.0→0.0 하락을 유발했는가, 아니면 user-sim 분산인가. [[08]] 전수 궤적·마커 추적으로 판정.

**판정 요약**:

| task | 판정 | 한줄 근거 |
|---|---|---|
| **t86** | **(b) 분산 — 확정** | 두 변경 모두 발화 0 (deny 0회·enum_items=0·cap 이벤트 0). 첫 분기=user 발화(msg 23·직전 컨텍스트 23개 메시지 완전 동일). user-sim이 오답 포인터(#W7739115) 단정 + 에이전트 커버리지 환각으로 write 미수행. |
| **t95** | **(b) 분산 우세 — cap은 실패 *모드*만 변경·결과 불변 [M]** | 첫 분기=user 첫 발화(msg 1·직전 컨텍스트=인사 1개 동일). user-sim이 허위 사실("두 노트북이 같은 주문") 단정→에이전트 frame-lock. cap 소진(deny 4회 전부 무시)이 인과 사슬 말단에 있으나, counterfactual(구코드=deny 무한루프)도 fail — 실패 결과는 불변. |

**권고**: cap=4 유지 · enum 힌트 조정 불요 (근거 §5). 전체 점수는 4/10→4/10 동률 (t81·t47 상승 ↔ t86·t95 하락).

---

## 0. 데이터·방법

- 궤적: `reports/facet_rft_2026/sim_results/t5c_aprime{1,2}.results.json.gz` (tasks 키 gold 내장·seed=300·user-sim=openrouter/gpt-4.1 temp=0.0·양 run 태스크셋 동일 10개, 8-동시 실행)
- 마커: 동 디렉토리 `*.stderr.log.gz`. 태스크 인터리브 → **주문ID(#W…)로 귀속** (태스크별 주문ID 전집을 궤적에서 추출, 전 태스크 간 서로 소 확인). ID 없는 walk 라인은 ledger replay의 (n, m, unexamined) 삼중치 일치로 귀속.
- replay: `scripts/distill/tau2/eplan_iso_probe.py --mode predicate` (현 작업트리=aprime2 코드) — 양 run 궤적에 적용.
- 분기점: C60 프로토콜 — 정규화 turn(role·content·tool_calls args) 비교로 최초 불일치 인덱스·주체 판정.

## 1. t86 (Yusuf Hernandez: 재킷 modify + 기본주소를 "주문 속 DC 주소"로 변경)

### 1.1 write-diff (gold 대조)

- gold: `modify_pending_order_items(#W2466703, 9385662952→8733974883)` + `modify_user_address(565 Maple Drive, Suite 501, Washington DC 20307)`
- aprime1 (1.0): 두 write 모두 **exact match**.
- aprime2 (0.0): `modify_pending_order_items` exact match ✓ / **`modify_user_address` 미수행(MISSING)**. db_match=False. 종료=user_stop.

### 1.2 첫 분기점 [S]

msg 0~22 **완전 동일** (동일 modify write·동일 assistant 문면 포함). **msg 23 = user 발화에서 최초 분기**:

- A1[23]: "…I want the Washington DC one from my order history, not the Dallas address. Can you check again and update it…"
- A2[23]: "…I want my default address set to the Washington DC one you have on file from my past orders. I'd rather not share the details again…"

동일 컨텍스트 입력에 다른 출력 = user-sim 비결정성(gpt-4.1 API, temp=0.0이어도 재현 미보장). **분기 주체 = user.**

### 1.3 이후 인과 사슬 (aprime2 정독) [S]

1. [26]~[27]: 에이전트가 주소 재요청 ↔ user "밝히기 싫다" 반복.
2. **[29] user: "It should be the address from order #W7739115"** — 허위 단정(그 주문 주소=Denver). gold DC 주소는 **#W1994898**에 있음(aprime1 [35]서 확인). user-sim scenario에는 주문 지정 정보가 없는데 지어냄 = confabulation.
3. [30]~[32]: 에이전트 #W7739115 조회→Denver 확인→재질문.
4. [33] user "check all my past orders" → [34] 에이전트 `get_user_details`만 호출 후 **[36] "I've reviewed your order history, and … none … are in Washington DC" = 커버리지 환각** (#W1994898·#W2166301·#W6832752 미조회. examined={#W2466703, #W7739115}뿐 — replay 확인).
5. [37] user 포기 → write 미수행 → fail.

대조: aprime1 [27] user는 "I'm not sure which order it is"라고만 하고 전수 조회를 시킴 → 에이전트가 4개 주문 순회 → #W1994898서 DC 발견 → write 성공.

### 1.4 변경분 관여 검사 [S]

- **E-PLAN deny**: 양 run 로그에서 t86 귀속(주문ID #W1994898/#W2166301/#W2466703/#W6832752/#W7739115 언급) deny 라인 **0건**. cap 이벤트 0.
- **enum 힌트**: 양 run t86 전체 user 발화에 `_enum_items` 적용 → **0** (발화 없음). walk_n=1·gap 없음(양 run) → walk도 불발.
- **replay (aprime2 코드 → aprime1 궤적)**: 두 write 시점 모두 fire "−" (침묵) — **신코드가 aprime1 pass 경로에 개입했을 발화 = 0. 변경은 t86에 완전 불활성.**
- [24] "I encountered an issue because I need your explicit confirmation…"은 confirm-gate(양 run 동일 코드) 흔적 — 변경분 아님.

### 1.5 판정: **(b) 분산 확정**

변경 코드의 어느 경로도 발화하지 않았고(마커·replay 이중 확인), 첫 분기는 동일-컨텍스트 user 발화. 실패 기제 = user-sim 오답 포인터 + (base 모델의) 커버리지 환각. 우리 레버와 무관.

## 2. t95 (Lei Wilson: 노트북 2대 exchange — 실제로는 **두 주문에 1대씩**)

### 2.1 write-diff (gold 대조)

- gold: `exchange(#W2905754, 3478699712→9844888101)` + `exchange(#W4073673, 2216662955→9844888101)`
- aprime1 (1.0): 두 write 모두 **exact match**.
- aprime2 (0.0): 성공 write **0건**. 유일 시도 = `exchange(#W2905754, item_ids=[3478699712, 3478699712], …)` → **env 오류 "Number of 3478699712 not found"** (한 주문에 같은 item 2개 시도). #W4073673는 조회조차 안 함(examined={#W2905754, #W3826449}). 종료=user_stop(휴먼 이관 후).

### 2.2 첫 분기점 [S]

**msg 1 = user 첫 발화** (직전 컨텍스트 = 동일 인사 1개):

- A1[1]: "I just received **a couple of laptops** …" (복수 즉시 계시)
- A2[1]: "I just received **a laptop order** …" (단수 프레임)

**분기 주체 = user** (변경 코드는 user-sim에 아무 경로로도 닿지 않음).

### 2.3 이후 인과 사슬 (aprime2 정독) [S]

1. [7] user: "it's for two 15-inch laptops that I just received. **Both** need to be exchanged" — 수량은 계시했으나 단일 주문 프레임 유지.
2. [10]~[14]: 에이전트 #W3826449·#W2905754 조회, #W2905754에 15-inch 노트북 발견 → "이 주문에 노트북들이 있다"로 오독. **#W4073673 미조회.**
3. [20]·[22]·[24] assistant: "It appears there was an error due to an oversight…" ×3 — **은닉 regen 라운드의 E-PLAN L2 deny 수신 흔적** (deny는 최종 메시지에 안 남음). stderr: `L2 deny: unexamined siblings #W4073673` **4회** (lines 644·717·854·863 — #W4073673는 t95 전유 ID). deny 피드백 문면은 매회 **정답 경로를 명시** ("You listed record(s) #W4073673 but have not read their details yet — call get_order_details for them first"). **에이전트는 4회 전부 불응** — `get_order_details(#W4073673)` 호출 0회 (frame-lock: user의 "같은 주문" 단정을 신뢰).
4. line 864: `deny cap 4 reached` → **cap 소진(변경 ①)** → 5번째 write 시도 통과 → [24] 중복 item_ids write → **env가 거부(ERR)** — DB 오염 없음.
5. [27] user: "I received both laptops together in the same delivery, so **they should be under the same order**" — **허위 단정 강화** (DB상 두 주문에 분산·scenario에 "같은 주문" 정보 없음 = confabulation).
6. [31] user 에스컬레이션 요구 → [32]~[36] 휴먼 이관 → fail. (참고: 마커상 walk는 user_stop 시 발화함 — line 1019 `walk gap: qty=2 executed=0 unexamined=1` = t95 말단 상태와 삼중치 정확 일치 — 리마인더까지 갔으나 회생 실패.)

대조 aprime1: 동일 L2 deny가 **1회**(line 777) 발화 → 에이전트 **즉시 순응**(#W4073673 조회) → 두 주문 각각 exchange → pass. 같은 레버·순응 여부만 다름.

### 2.4 변경분 관여 검사

- **enum 힌트**: t95 전체 user 발화 `_enum_items`=**0** (양 run — "i7 processor, 8GB RAM, 1TB SSD"는 digit 세그먼트가 run 절단·설계대로 미발화). **변경 ②는 t95 불관여 확정 [S].**
- **deny 횟수 대조**: aprime2가 더 많음(4 vs 1). 단 L2 술어 자체는 무변경 — 횟수 차이는 코드가 아니라 *에이전트 불응*(user 허위 프레임 하류)의 결과.
- **cap(변경 ①)의 인과 위치 [S]**: cap 소진이 사슬 말단에 존재(마커 864 → 중복 write 방류). 단 방류된 write는 env가 거부 → 실패 형태는 "오답 write"가 아니라 "gold write 미수행" — cap 없어도 동일.
- **counterfactual(구코드 = cap 없음) [M]**: deny가 무한 지속. 근거 — (i) 이 sim에서 정답 order-id·도구명을 명시한 deny 4회 전부 불응, (ii) 동형 사례 t103(aprime1): 동일 L2 deny 수십 회에도 순응 0→max_steps 소진 fail, t27: user 포기 fail (커밋 00fa5d23의 cap 도입 동기 그 자체), (iii) t95 user는 이미 조급 상태("move quickly"·"sorted out quickly")·[31]서 에스컬레이션 요구 = deny 지속 시에도 user_stop/이관 경로. **구코드에서도 fail — cap은 결과를 바꾸지 않고 실패 모드만 변경(deny-루프 소진형 → env-오류+이관형).**
- **replay (aprime2 코드 → aprime1 궤적) [S]**: 두 write 시점 모두 fire "−" — 신코드가 aprime1 pass 경로를 바꾸지 않음 (aprime1 deny 1회 < cap=4 → cap 불활성·enum=0).

### 2.5 판정: **(b) 분산 우세** (cap 관여는 실패-모드 한정·결과 불변 [M])

근본 원인 = user-sim 첫 발화 분기 + "같은 주문" confabulation → 에이전트 frame-lock·deny 4회 불응. 변경 ①은 인과 사슬 말단에서 실패의 *형태*만 바꿈. (a) 부작용 확정 요건인 "변경-유발 개입이 실패를 만들었다"는 성립하지 않음 — 개입(방류)의 산출물이 env에서 기각되어 DB 무영향이었고, 개입이 없던 세계(구코드)의 종점도 fail. 순수 (b)와의 차이는 counterfactual이 [M]등급(동형 사례 외삽)이라는 점 — [S]로 승급하려면 유료 재run이 필요하므로 [[09]]상 비권장·불요.

## 3. 교차 검증: 나머지 8태스크 및 상승분

- 전체: aprime1 4/10 (61·17·86·95) → aprime2 4/10 (61·17·81·47). **하락 2 = 상승 2, 합계 동률.**
- **t81 0→1**: 변경 ②의 설계 표적 그대로 — aprime2서 enum_items=6(6품목 나열) → L1 deny(line 311 인접)+walk(line 550: qty=6·m=1) → 두 write 완수. 변경의 양(+) 방향 실증.
- **t47 0→1**: deny 4회(L1 1+L2 3, lines 311/406/549/677) 후 **cap 소진(line 678)** 상태로 **pass** — cap-방류가 회생 경로가 된 대칭 사례(모드 변경이 양방향임을 시사) [M].
- **t103**: aprime1 deny 수십 회·201 msgs·max_steps fail → aprime2 deny cap 후 54 msgs·user_stop fail — cap이 무한루프를 설계대로 종결(결과는 동일 fail·비용만 절감).

## 4. 증거 원장 (등급)

| # | 증거 | 등급 |
|---|---|---|
| E1 | t86/t95 write-diff·종료사유·action_checks (양 run) | [S] |
| E2 | 첫 분기점: t86 msg23·t95 msg1 = user 발화, 직전 컨텍스트 동일 (정규화 turn 전수 비교) | [S] |
| E3 | 마커 귀속: t95 L2 deny 4회(#W4073673 전유 ID)+cap(line 864) / t86 deny 0 (양 run 주문ID 전집 서로 소) | [S] |
| E4 | ledger replay(신코드→aprime1 궤적): t86·t95 모두 전 write 시점 침묵 = 신코드는 pass 경로 불변 | [S] |
| E5 | enum_items=0 (t86·t95 전 user 발화·양 run) | [S] |
| E6 | user-sim confabulation: t95 "same order"(DB=두 주문)·t86 "#W7739115"(DB=Denver) | [S] |
| E7 | counterfactual "구코드도 fail": deny 4회 불응 + t103/t27 동형 + user 조급/이관 요구 | [M] |
| E8 | t47 cap-방류 후 pass = 모드 변경 양방향 | [M] |

## 5. 권고

1. **cap=4 유지.** t95는 cap 값 문제가 아님 — 정답을 명시한 deny를 4회 무시한 에이전트가 6~8회에 순응할 근거 없음(t103서 수십 회 무순응 [S]). 상향은 t103형 비용만 재유입. 하향(예 2~3)도 근거 없음 — aprime1 t95는 deny 1회 순응 pass였고, t47은 4회차까지 간 뒤 회생.
2. **enum 힌트 무조정.** 표적(t81) 적중·비표적(t86/t95 포함 전 태스크) 발화 0 — 456-census의 과발화 0 판정과 일치. 이번 하락과 무관.
3. **이번 하락은 레버 회귀 신호가 아님** — 단일-trial run 간 user-sim 분산(특히 **confabulation형**: 시나리오에 없는 사실을 단정해 에이전트를 오도)이 지배 항. A′ 계열 pass 집계 해석 시 trial=1 점수의 태스크별 ±는 이 분산 대역 안 [M]. 결론이 필요한 태스크는 (유료 재run 전에) 본 문서 방식의 궤적·마커 포렌식이 선행 절차 [[08]][[09]].
4. (관찰·큐 밖) t86 aprime2의 "order history 다 봤다" 커버리지 환각과 t95 frame-lock은 기존 분류(관찰=집계-단정형)의 재출현 — 별도 레버 논의는 등대 실험큐 절차로.

*작성: 2026-07-11 포렌식 세션. 재현 스크립트: 분기점/write-diff·마커 귀속·enum 검사=세션 scratchpad(일회성·본문에 결과 영속), replay=`scripts/distill/tau2/eplan_iso_probe.py --mode predicate --results <gz> --task {86,95}`.*
