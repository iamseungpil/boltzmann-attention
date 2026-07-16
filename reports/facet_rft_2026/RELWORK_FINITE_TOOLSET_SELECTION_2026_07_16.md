# RELWORK — 유한 도구집합 선택 실패: 선행 지형·whitespace 판정 (2026-07-16 딥리서치)

> 출처: deep-research `wf_9a27ec8c-763` (101 에이전트·19소스·주장 25건 3표 적대검증: 18확정·7기각·0미검).
> 질의 맥락 = 이 세션 관측 A/B/C. **⚠️관측 C는 질의 제출 後 `ASSERTION_PROVENANCE_ARMS_DESIGN` §11로
> 거짓 전제 판명**(A2 도구가 `_f`로 주입돼 스키마가 모델에 도달한 적 없음) — 관측 C 관련 결론은 전부 그 기준으로 재해석(§5).
> 관측 A(`by_phone*` 날조·네이티브 도구)와 B(가용 대안 `by_name` 미선택)는 유효.

## 0. 한 줄 (whitespace 판정)
현상 명명·기전 가설·knowing-doing 격차는 **선점**됐다. 남는 whitespace는 **좁은 정식화 하나** —
**"결정론 게이트가 날조 *실행*은 0으로 막지만 *재선택*은 사지 못한다"(차단≠회복)** — 단, 인접 3편
(회복률을 재는 논문들)을 정독하기 전엔 최종 판정 보류(§4).

## 1. 선점된 것 (인용·양보)
| 축 | 선행 | 내용 | 등급 |
|---|---|---|---|
| **현상 명명** | `2412.04141`(ICML25)·`2406.20015`(ToolBeHonest·EMNLP24)·`2601.05214`(Amazon) | 비존재 도구명 날조 = "tool type / function selection hallucination"으로 분야 표준 등재 | [S-lit] |
| **기전(prior-override)** | `2606.07555` "Priors Persist Through Suppression" | lexical-prior 강도가 간섭 예측(β=+0.114, p<10⁻⁹, n=7,744)·**명시적 국소 규칙도 prior를 못 끔**(11모델 1B-9B × 4 conflict family 전부 Δ>0) — C30/[[42]]의 통제된 판박이 | [M]·Stroop 도메인 유비 |
| **knowing↔doing** | `2606.20661` KAPRO/KAware | Knowing(메타인지 probe)과 Acting(호출) 분리 측정·**DirGap +33.67**(claude-sonnet-4.5-think: 불필요를 알고도 호출) | [M] |
| | `2605.14186` | FOK/JOL 신호는 산출되나 제어에 안 쓰임 — "제어 인터페이스 부재"로 정식화 | [M]·도구선택 벤치 아님 |
| **frontier 잔존** | `2406.20015` | ToolBH: Gemini-1.5-Pro **45.3** · GPT-4o **37.0**/100 — "큰 파라미터가 보장 못 함" | [M]·2024 frontier |
| **학습 완화의 정체** | `2412.04141` | SFT→DPO로 날조 61.5→18.8%·unmatched 91.1→**0.0%** — 단 **0%는 기권으로 달성**(tool call 3.3→0.8): "더 잘 고르게"가 아니라 "안 고르게" | [M]·8B 단턴 |
| **추론강화 역효과** | `2510.22977` "The Reasoning Trap" | RL/SFT/inference-time 추론 강화가 도구 환각을 **증폭**(Qwen2.5-7B NTA 34.8→90.2%·prompt로는 90.2→87.5% ≈ 무효)·무관 태스크(수학) 강화도 전이 | [M] |

## 2. 우리 규율과의 정합 (인용 시 짝지을 것)
- `2606.07555` "명시 규칙도 prior 못 끔" ↔ **C30**(policy.md:18에도 날조 91건)·**C47**(예시+금지문 무효)·**[[42]]**.
- `2606.07555` §5.4 활성화 패칭: override 복구는 **prior 억제가 아니라 문맥 target 유지 채널**에서 옴
  ⇒ **차단(억제)과 회복(대안 유지)이 다른 채널**이라는 우리 분리의 유일한 기전적 지지([D]·유비).
- `2412.04141`의 "0% = 기권" ↔ **C21**(o4-mini는 기권형)·**A4**(frontier는 기권, 우리는 틀리게 행동).
- `2510.22977` 추론강화 역효과 ↔ 등대 §1.3(레버는 하나를 사면 하나를 판다)·**C4b**(thinking 순 0).
- scale은 간섭을 줄이나 0으로 못 만듦(`2606.07555` logparams β=−0.32·1B-9B) ↔ **[[45]]** scale-invariant 잔여.

## 2b. ★★포지셔닝 (2026-07-17 사용자 프레임) — "`2606.07555`는 기전, 우리는 해법"
그 논문이 보인 것: ① 금지문·명시 규칙은 prior를 못 끈다(**억제 채널 폐쇄**) ② 패칭 복구는 prior 억제가 아니라
**문맥 target 유지**에서 온다(**회복=target 채널**) ③ 단 그들의 개입=activation patching = **배포 불가**.
⇒ **우리 FIND/GET/INFER/ASK 선택 루프(C45 출처선언 + C48 호출가능성 위계·[[16]] LOCK)가 target 채널의
배포 가능한 결정론 구현이다**: 금지 대신 모든 operand/도구 결정을 유효 출처 4개로 라우팅·검증기는 결정론
(producer 호출가능 ∧ INFER 선언 → 무효 → 재발화·GET 폴백).
- 행동 수준 재현(이 세션): 차단 문구만(억제) → 같은 날조 4~5회 반복 / **유효 대안이 도구 집합에 존재**
  (`verify_identity`·`by_name`, target) → 회복 → task_019 사상 첫 pass. C45(날조 67→0%·over-block 0)와 동형.
- 논문 문장: *기전 논문이 억제-측 레버의 실패를 내부 증거로 보였다면, 우리는 그 채널 분리를 32B·다중턴
  tool-use에서 행동 수준으로 재현하고, target 채널의 배포 가능한 구현(4지선다 루프+대안 공급)과 그 측정을 제시한다.*
- ⚠️정직 경계: (i) 패칭→루프 매핑은 **유비 [D]**(1B-9B lexical ↔ 32B tool-use·동일 기전 증명 아님)
  (ii) "대안 공급" = **A2로 도구를 집합에 등록**이지 런타임 도구 추천(사후-redirect·폐기됨) 아님 — [[16]] 선택유도/강제 경계.

## 3. 우리 관측 중 선행이 안 다룬 것 (모트 후보)
1. **in-band active contradiction**: *같은 궤적 안에서* "그 도구는 없다"고 언어화한 **직후** 그 도구를 재호출
   (bank_dreq_20260716_2140 sim1 `[20][24]`→`[22][28]`). KAPRO=elicited probe·`2605.14186`=수동 방치와 구별.
2. **차단 후 재선택 회복률의 실측**: 게이트가 실행을 0으로 막은 뒤 무엇이 회복을 사는가 —
   우리 데이터: TOOLGATE 차단만으로는 4~5회 반복(t019d/g), **유효 대안 도구(`verify_identity`) 제시 후 회복**(2230 런).
3. **32B급·다중턴·유한집합 전부-제시** 조건의 정본 수치 자체가 부재(선행은 1B-9B Stroop·8B 단턴 API).

## 4. ✅whitespace 최종판정 (2026-07-17·인접 3편 병렬 정독 완료) — **3/3 전부 미선점(c)**
| 논문 | 실체 (정독 결과) | 판정 | 반드시 인용·차별화할 지점 |
|---|---|---|---|
| **`2506.21967`** "More Vulnerable than You Think" (PKU+Huawei) | 회복 평가 실재하나 **차단이 아니라 합성 과거-스텝 주입**(정답 궤적 절단+실존 도구의 실제 응답 부착). 비존재 도구·차단 문구·반복 emit·ablation 전무. ToolBench 단일쿼리 ReAct·32B 없음(7-20B, 70B급) | **(c)** | §4.2 *"agents can often identify and correct mistakes by **choosing a new appropriate tool**"* — 그들 설정은 **대안 상시 존재** 조건이었고 그때 도구선택 오류는 빨리 교정(Δ<8%). **우리 verify 벽 = 대안 부재 조건서 4~5회 반복 → 대안 공급 시 회복** — 두 조건이 "회복=대안 존재"의 양쪽을 맞물려 지지. 대조 시 설정 차이 명시(rebuttal 안전) |
| **`2604.16706`** AgentProp-Bench (단독 저자) | Interceptor = 병렬 모니터 3층 → **기권(abstain)**·피드백/재시도 루프 아님. −23.0pp의 대부분 = 기권 33.5%로 산 것·GPT-4o-mini만 유효·절대치 human-unvalidated 자인. 파라미터 오염 주입(비존재 도구 아님)·단일턴 chain·전부 폐쇄모델 | **(c)** | §5.3 *"Rejection and recovery are **independent capabilities** (Spearman ρ=0.126, p=0.747)"* — 워딩 유사·조작화 상이(내생 거부 vs **외부 게이트 100% 공급**·오염 후 정답도달 vs **재선택**). 그들=무상관 **관찰**, 우리=**기전 분리+인과 레버 식별**(대안 도구) |
| **`2606.31307`** "When the Database Fails" (SIGDIAL26·=Guided-Retry) | **tool-calling 아님**(단턴·DB실패 합성주입·재시도 루프 없음). 환각=응답 내용 날조(가짜 예약번호). 딥리서치 라벨 "tool hallucination 42-50%"는 **오라벨** — 인용 시 "DB-grounding content hallucination"으로 정정 필수 | **(c)** | *"prompt-level recovery is helpful, **but not sufficient on its own**"* + Phi-3선 구조화 지시가 **역효과**(35.2/37.0% 잔여) = [[42]]의 단턴·콘텐츠판 방증 |

⇒ **좁은 정식화 확정 미선점**: "결정론 게이트가 비존재-도구 emit의 *실행*을 0으로 차단해도 재선택은 회복되지 않으며
(4~5회 반복·in-band 부재 언어화 직후 재호출 포함·궤적 실측 [M]), 회복은 차단 문구가 아니라 **유효 대안 도구의 존재**와
동행한다". 세 인접 논문 모두 이 구조(외부 결정론 차단 채널 + 대안 조작 + 다중턴 + 반복 emit 계측)가 부재.
- ⚠️**정직 경계([[08]] 자기감사)**: "대안 존재 → 회복"은 현재 **관찰 [P]**다 — 대안(`verify_identity`) 추가가
  버그픽스들(§11 `_f`·§12 PROV)과 **같은 스택 변경에 동반**되어 격리되지 않았다.
- **★키스톤 실험(단일 변수·2026-07-17 사용자 교정으로 축소)**: 같은 스택서 **차단 문구 고정** + **대안 도구
  (`verify_identity`)의 A2 유/무만 toggle** → 검증 벽 재선택 회복률 짝지은 대조. 이것이 통과해야 [P]→[M].
  ⚠️금지문-강도 arm은 **불필요**(제거) — 금지문 무효는 이미 확정: **C30**(정책 금지에도 날조 91)·**C47**(태그+금지문 9→8)·
  **이 세션 궤적**(TOOLGATE 피드백 상존에도 4~5회 반복·부재 언어화 직후 재호출) + 외부 재현(`2606.07555` 래퍼 4종·
  Relign 베이스라인). 재조작은 기증명 재증명 = 비용 낭비.
- 잔여 체크 1건: 기각된 `2605.14038`(cognition→action probe·0-3) 재검토는 관측 B(anchoring) 논증 시점에.

## 4b. ✅DR2 전수 지형 판정 (2026-07-17 수령·`wf_1190739b-1e4`·4주장 (a)완전선점/(b)부분/(c)미선점)
> 출처: deep-research 2차(5축 분해·후보 30편·전문 deep-read 24편·적대검증은 `2607.01641` 4표 완주).
> ⚠️수령 상태: 세션 단절로 마지막 주장 투표 2건(voter 2/3·3/3) 미완 — 단 같은 논문에 **독립 검증 4건 전부
> refuted=false·high 일치**(정적분석 전용·hallucination 어휘 0·대안-도구 조작 없음)라 판정 유지. deep-read
> 판정은 축자 quote 동반이나 적대투표 미실시 = **[M-lit]**(인용 시 C104 ⑥ 규율: 원문 정독 필수).

| 주장 | 판정 | 핵심 선행 (양보·인용) | 잔여 미선점 (우리 delta) |
|---|---|---|---|
| **1 차단≠회복 + 대안-도구 인과 레버** | **(b)** | perseveration 계량(유효 도구 대상): `2605.08477`("동일 도구+동일 인자 반복률" metric·SH 66.9% vs FH 13.9%·**"try a different tool" 피드백에도 5연속 반복**·단 constrained decoding으로 비존재-도구 emit 자체가 구조적 불가) · MIRAGE `2507.21017`(FIH-repetitive 3/6회 스냅샷·judge·Qwen2.5-32B HR .324·완화실험 없음) · SciAgentGym `2602.12984`(Loop Escape 35.7%) · ★**ToolMaze `2606.05806`**: 대안-도구 topology를 통제변수로 조작(C1 무대안 vs C2 대체·PRR 33.40→50.54%)·"switch to alternative"=정상 회복 규범화 — **단 실재 도구의 주입-오류 세팅**(비존재-도구 게이트 차단 아님·도구는 목록에 상존) · AgentAbstain `2607.10059`(도구 toggle=minimal perturbation·DV=기권·**Post-hoc가 아니라 회복 아님**) · `2607.01641`=정적분석 전용 **[S] 미선점 확정**(4표) | "**날조 비존재 도구**의 결정론 차단+명시 피드백 후 동일명 재-emit 계량 + **A2 대안 공급 단일변수 toggle**"은 여전히 부재. §20 키스톤 기각과 정합: ToolMaze(그들 세팅선 대안→회복↑) vs 우리(Δ=0·perseveration↓만) = **세팅 대조 foil**로 사용 — 논문 프레임=레버-부작용 유지 |
| **2 완료 날조 + 구조 이벤트 게이트** | **(b)·명명·벤치·rate는 사실상 (a)** | ★**False Success `2606.09863`**: **같은 τ²-bench** 9,876궤적·"false success"=실패의 45–48%(single-control)·모델별 13~79%·텍스트 탐지기 AUROC 0.83–0.95 — **단 post-hoc triage only·개입 0**·LLM-judge 실패(≤0.65)는 우리 문제설정 지지 · PAE `2603.03116`(Execution Consistency·**confirmation numbers 날조**·Phantom Booking·LLM-judge post-hoc) · 서베이 `2509.18970`("execution/outcome hallucination" 명명 기존재) · NabaOS `2603.10060`(HMAC 영수증 교차검증·fabricated tool reference 94.2% — 구조-증거 게이팅 부분선점·단 claim 추출=**텍스트 파싱 필요**·탐지-only·합성주입 평가) · ToolFailBench `2607.04686`(Output-Fabrication=**도구 호출됨** 전제=구별) · AgentAbstain(Post-hoc Abstention 2.6%=거울상) | **"action-completion fabrication" 신규 명명 주장 철회**(관리표 행2 갱신). 잔여 delta = ①**라이브 개입**(탐지→regen 인과사슬·전 선행=post-hoc/탐지-only) ②**텍스트-파싱-0 순수 구조-이벤트 결정론 게이트**({GET∧후속 미호출∧사임 N회}) ③자연발생 다중턴서 게이트 유효성. 잔여 정독 1건: AgentLTL `2607.02599`(LTL 절차준수 **training** 포함— 완료-주장 게이팅 여부 원문 확인) |
| **3 in-band active contradiction** | **(c)에 근접한 (b)** | knowing-doing 정본 `2504.16078`(bandits·옳은 rationale 87%에도 58% greedy — tool-calling 아님) · HalluClear `2604.17284`(RH.2 reasoning↔action 모순 명명·계측 — **GUI·VLM-judge·스텝 내**) · When2Tool `2605.09252`(probe-기반=KAPRO 계열·"표현지식은 표출능력과 독립"=오히려 우리 delta 지지) · MIRAGE FIH rubric(인지-**부재** 측정=방향 반대) · AgentAbstain(missing verifier **인정 직후** $500 이체 관찰 — 전용 계측축 아님·기권실패 분류 일부) | tool-calling에서 "**부재 언어화 직후 그 도구 재호출**"의 전용 명명·same-trajectory 계량 = 미선점 유지(§3-1 그대로). 인접 5편 인용 필수 |
| **4 턴 간 인자 누적 + 비대칭** | **(b)** | ToolDial `2503.00564`(과제공간 선점: DST→tool-args·11,111대화·sub-70% — **직전-턴-only 유형 명명 없음·description A/B 없음**) · LLMs Get Lost `2505.06120`(sharded −39%·**loss-of-middle-turns** 명명=현상 일반형 선점·개입=user-side Recap/Snowball뿐) · IFEval-FC `2509.18420`(param-description 지시 준수 계량·단일턴·"자주 무시됨"·<80%) · TAFC `2601.18282`(description 최적화=방법 전제 선점·wording A/B 없음) · `2601.08070`(금지문 역효과=일반 LLM 수준) | ①**직전-턴-only 인자누적 실패**의 유형 명명·계측 ②A2 설명 "누적 명시" **A/B 대조(60→93.3%·n=60/arm)** ③금지문 무효 vs 긍정형 구성-지시 유효 **비대칭의 tool-arg 맥락 실측** = 셋 다 미선점. 잔여 정독 1건: ToolHaystack(historical noise→parameter hallucination·서베이 경유 [D]) |

- **행12(논문 코어 "인용-동반 도구사용의 학습") 선점확인**: 미선점 유지(잠정) — 최근접 = NabaOS `2603.10060`
  (런타임 영수증 **탐지**·학습 아님·텍스트 claim 추출)·AgentLTL `2607.02599`(**training이 제목에 있음** —
  단 LTL 절차준수이지 값-출처 주석 학습 아님·**원문 정독이 최종 확정의 유일 잔여**). C45(선행0 기확정) 위 신규층 유지.
- **부수 수확**: ToolMaze scale-저항(기본성능이 fault-tolerance보다 3.66× 빨리 성장)·AgentAbstain
  scale-독립(최고 59.5%·13/17이 50%미만) = [[45]] 인용감 2건 추가 · ToolFailBench Qwen2.5-32B 단일턴
  CTUR 82.68%(3위) = "다중턴 병리는 단일턴 무능으로 환원 안 됨" 방증.
- **원문 정독 필수 목록**(인용 前·C104 ⑥): `2606.09863`·`2606.05806`·`2603.10060`·`2607.02599`·ToolHaystack.

## 5. 관측 C 재해석 (질의의 거짓 전제)
- 질의에 넣은 "인자 설명에 'the records you read' 명시했는데도 되묻는다"는 **거짓** — `_f` 주입으로 스키마가
  모델에 도달한 적 없음(§11). 스키마 수정 후 **LLM이 23건을 정확히 formalize**(6/6 엔진 `-> 4`·gold 일치 10/10).
- ⇒ 딥리서치의 open question 4(관측 C의 clarification 선행)는 **추적 불요**. ToolDial(`2503.00564`)은
  다른 목적(진짜 인자 결손 시 clarify 정식화)으로만 유효.
- 관측 A/B 결론은 영향 없음(네이티브 도구·정상 주입).

## 6. 엔지니어링 시사점 (검토만·[[05]] 주의)
- PA-Tool(`2510.07248`) "모델을 도구에 맞추지 말고 **도구 스키마를 모델의 pretraining 명명관행에** 맞춰라"
  — `by_phone` 날조의 완화 후보로 흥미로우나, **tau2 네이티브 도구 개명은 벤치 비교성 파괴**라 우리 세팅선 불가.
  우리 A2 도구 명명에만 참고.

## 6b. ★정독 결과 반영 (2026-07-17·`2606.07555`·`2606.20661` 전문 정독)
### `2606.07555` (기전) — §2b 프레임의 등급·예측 확정
- 인과 실증 범위 = **1B-2B·반의어 계열·game-rule 래퍼·단일토큰**(Limitations 자인). 행동 결과는 1B-9B.
  ⇒ 우리 32B 도구선택 매핑 = **유비 [M]**·"원문이 지지"라고 쓸 수 있는 것은 *지시문이 간섭을 못 없앰*(Table 8:
  "Using only this glossary"도 Δ=2.06·78-81% 아이템 양성)까지.
- 이론 틀: **보수적 베이지안 갱신**(§6 — 새 증거는 할인·prior는 전액 유지) = "차단 피드백이 턴마다 할인되어
  prior 도구선택이 복원"되는 우리 궤적의 좋은 설명 틀.
- ★**검증 가능한 예측(=§4 키스톤 실험과 동일)**: *금지문 강도 조작 → 재호출률 무효과 / 유효 대안 가시성 조작 → 효과*.
  이항 대조·A2 toggle로 실행 가능([[16]] A2-거의무료). **이것이 통과해야 §2b "해법" 주장이 [M]으로 선다.**

### `2606.20661` (KAPRO) — 프로브 설계 이식 + whitespace 축자 확정
- **이식할 형식**(Appendix E.3): 실행금지 role + 문맥 + **후보 목록 명시** + 후보별 true/false JSON 강제.
  우리 결정점 프로브(`bank_op_operand_probe.py`)에 얹으면: 궤적 접두 그대로 + "실행 말고 각 후보의 유효/필요만 판정".
- **채점 이식**: E_know(프로브) vs E_act(같은 접두 실호출) vs T(게이트=결정론 GT) → 4분면.
  우리 관측(부재 언어화→호출) = **Kc/Aw(Alignment Issue)** 셀·DirGap로 부호화. 우리 GT가 결정론이라 논문(LLM 합성 GT)보다 깨끗.
- **우리 delta 표기**: 논문=task-level·사전·별도 세션 / 우리=**결정점-level·mid-trajectory·in-band 언어화 대조**.
  "KAPRO의 Knowing probing을 결정점 granularity로 세분화"로 인용.
- **whitespace 축자**: §7 *"does not itself propose training-time or inference-time interventions to close these gaps"*
  + E.2 zero-shot·greedy 확인. ⇒ **executive-control gap을 겨냥한 개입(우리 게이트/A2·대안 공급)이 그들의 명시적 공백.**
- 부수: KAS↔평균 호출 수 Pearson **r=−0.748**(RQ2) — over-action 비용 논증에 인용 가능. 30B급 오픈모델 수치 없음
  (Qwen3-30B-A3B는 주석 파이프라인 screener로만 등장 — 피평가 아님. 오인용 주의).
- ⚠️방법 경고: 초기 WebFetch 요약이 논문에 없는 한계 문장을 **지어냈음**(전문 대조로 판명) — 요약-기반 인용 금지·원문 정독 필수.

### `2510.22977` (Reasoning Trap·ACL26) — learn-wing 처방 (전문 정독)
- **딥리서치 요약 교정**: "late-layer 표상 붕괴"는 절반만 정확 — **붕괴(CKA<0.75)는 early/middle layer**(OOD 도구입력),
  **판별 신호(>0.14)는 late residual stream**(§5.1/5.2). 합쳐 쓰면 오인용.
- 정본 수치: think-then-act RL이 최악(NTA 41.4 vs **90.2**·direct RL 대비) · **R1-증류 SFT 단독으로도** 34.8→74.3 ·
  수학-only GRPO도 전이 상승(§4.2) · **Qwen3-32B도 Think On DT 50.7%** = scale이 이 축 못 닫음([[45]] 인용감).
- ★**learn-wing 처방** (우리 계획에 직결):
  1. **기권-정답(NTA/DT형) 음성 사례를 학습 분포에 처음부터 포함** — 유일 유효 완화(DPO 양방향 선호쌍)가 이것.
     사후 DPO 패치는 **utility −24%** 선례 ⇒ 사전 설계 우선. **D7(근접-오답 배치·음성사례)과 정확히 합치.**
  2. **think-형식 궤적 증류 경계** — 형식 자체가 오염원(74.3). [[12]] 다양성 요건과 별개의 추가 제약.
  3. **SimpleToolHalluBench(592문항·공개·github.com/albert-y1n/Reasoning_Trap)를 learn-wing 체크포인트
     회귀 게이트로 상설화** — 로컬 생성 무료·[[09]] 무료검증 범주. 100-step 추적 프로토콜 그대로 재사용([[41]]).
  4. C4b와의 관계: **같은 trade-off 계열·다른 기전**(그들=훈련시 가중치 변화·우리 C4b=추론시 예산) — 동일시 금지·
     "trade-off 계열의 독립 실증(훈련-레버 판)"으로 인용. 등대 §1.3 제1원리의 외부 정본 인용처.
- 현재 스택(32B-Instruct·RL 없음)엔 즉각 위협 없음 — **리스크는 learn-wing 가동 시점부터.**

### `2412.04141` (Relign·ICML25) — 전문 정독 (v1/v3 판본 차이 포함)
- ⚠️**판본 경고**: 61.5→18.8·91.1→0.0은 **v1 Table 1**·18.8은 **DPO-단독** 행(순차 SFT→DPO는 25.5).
  정본은 **v3(ICML 카메라레디·표 전면 교체)**: Toolllama 56.5→18.3 · Llama3.1 50.8→14.6 · Qwen2.5 49.2→17.6. 인용 시 v3 Table 2.
- **ChangeTools/TalkToUser의 실체** = 새 도구가 아니라 **기존 Finish 액션의 반환 분기**(v3 Table 9) — 전부 **terminal**
  (TalkToUser는 "묻는 것으로 과제를 끝냄"·user-sim 없음). ⇒ **다중턴 ASK-후-재개는 그들의 미해결 whitespace** = 우리 τ² ASK 분기가 사는 곳.
- ★**[[42]]/[[45]] 독립 재현**: 그 분기 프롬프트가 **베이스라인 포함 전 모델에 제공**되는데 7B는 환각 49-57%·GPT-4o만 8.3%
  ⇒ "행동 공간을 프롬프트로 열어줘도 소형은 못 쓰고 frontier는 쓴다" — 학습 없는 ASK류 추가 = 그들의 *베이스라인 조건*이지 *방법*이 아님.
- ★**learn-wing 수입품** (INFER-calibration·[[16]] 유일 잔여의 데이터 설계):
  1. **변조 연산자 2개**(도메인-일반): 도구셋 무관-교체(unmatched)·질의서 파라미터 은닉(missing)·배합 **4:3:3**(원본:교체:은닉 — 원본 40%가 과잉기권 방지).
  2. **스텝 DPO 3종 쌍**: `(A_correct>A_indecisive)`·`(A_indecisive>A_hallucinated)`·`(A_correct>A_hallucinated)` —
     특히 첫 쌍이 "정답 가능할 땐 ASK도 벌점"을 명시 학습 = **과잉기권 방지의 핵심**.
  3. APIBench OOD 전이(학습 없이 개선·v3 Table 4) = [[11]] 전이 방향의 인용 가능 지지.
- **기권 붕괴 방지는 부분적**: false-abstain calibration 지표 부재·UT/MP 소집합 통째 unsolvable 처리(일괄 기권=만점)
  ⇒ 우리 Δspurious≤0 계측(등대 §1.3) 같은 반대편 계측이 없음 — 적용 시 자체 보강 필수.

## 7. Open questions (딥리서치 산출·유효분)
1. in-band active contradiction을 정식화·계측한 선행이 정말 없는가 (`2605.14038` 재검토 포함).
2. §4 인접 3편의 회복률 측정이 우리 좁은 정식화와 얼마나 겹치는가.
3. 관측 B(정책 factor 열거가 후보 도구군을 좁히는 anchoring)를 도구선택 맥락서 계측한 선행 — 이번 조사서 0건.
