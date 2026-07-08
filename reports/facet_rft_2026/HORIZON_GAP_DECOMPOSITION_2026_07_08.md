# frontier 격차의 정체 — horizon 반증 · F2 국소화 (2026-07-08·무료·기존 궤적)

> 상위 = `RESEARCH_MASTER.md`. 동기 = 사용자 질문("open 모델로 frontier급 compliant 성취 가능한가·우리 접근이 타당한가").
> 데이터: `asmregen{32b,14b}`·`fl32b_floor`(우리) + tau2-bench 공식 frontier 궤적(claude-3.7·gpt-4.1·o4-mini·gpt-4.1-mini,
> 전부 gpt-4.1 user-sim·4trials). 스크립트=scratchpad `horizon`·`h34`. gpt-4.1 = 0.
> **horizon 정의 = task-내재 복잡도(gold action_checks 수)**. 에이전트 자신의 턴 수는 내생적(실패 시 길어짐)이라 사용 금지.

---

## 1. 검정한 가설
> **H0 (비관)**: 우리가 frontier에 못 미치는 잔여는 **F6 horizon**(per-step 신뢰도의 복리 $p^H$)이며, DR#2에 따르면 이는
> **scale만 사는 축**이므로 **scaffold로는 원리적으로 못 닫는다.**

## 2. 결과 — **H0 반증** [M]
### 2.1 pass by horizon (task-내재 H)
| model | H 0-2 | H 3-4 | H 5-7 | H 8+ | **p_step**(log-fit) |
|---|---|---|---|---|---|
| **ours 32B+scaffold** | 0.649 | **0.456** | 0.667 | 0.736 | 1.027 |
| 32B floor | 0.561 | 0.397 | 0.613 | 0.569 | 1.015 |
| 14B+scaffold | 0.561 | 0.397 | 0.631 | 0.722 | 1.042 |
| gpt-4.1-mini | 0.730 | 0.559 | 0.696 | 0.528 | 0.975 |
| o4-mini | 0.709 | 0.647 | 0.708 | 0.806 | 1.017 |
| gpt-4.1 | 0.743 | 0.765 | 0.774 | 0.639 | 0.983 |
| claude-3.7 | 0.764 | 0.794 | 0.821 | 0.750 | 0.997 |

- **$p_{step} \approx 1.0$ (0.975~1.042) — 전 모델.** 이 벤치서 **horizon 복리 감쇠가 관측되지 않음**. pass는 H에 단조감소하지 않음.
- **H 8+에서 우리는 경쟁력 있음**(gpt-4.1 대비 **+0.097**, claude-3.7 대비 −0.014).

### 2.2 격차는 **중간복잡도(H 3-4)에 집중** — 타 구간의 3~5배
| vs | H 0-2 | **H 3-4** | H 5-7 | H 8+ |
|---|---|---|---|---|
| o4-mini | −0.061 | **−0.191** | −0.042 | −0.069 |
| gpt-4.1 | −0.095 | **−0.309** | −0.107 | **+0.097** |
| claude-3.7 | −0.115 | **−0.338** | −0.155 | −0.014 |

⇒ **잔여는 horizon(scale 독점 축)이 아니라 기능(functional)이다. scaffold 사정거리 안에 있다.**

## 3. H 3-4 실패의 구성 — **F2 국소화** [M·소표본]
H3-4 task=17. 실패 sim: ours 37 / o4-mini 24. gold-write 미스매치 유형:
| 유형 | **ours** | **o4-mini** |
|---|---|---|
| **order_id (⋈ = F3 semantic 경계)** | 14 | **12** |
| **item/variant (F2 symbolic operand)** | **13** | **2** |
| missed_write(안 함/다른 tool) | 5 | 15 |
| other operand | 3 | 6 |

- **★⋈(F3)는 frontier도 거의 동일하게 틀린다**(14 vs 12) → **경계는 공유**·우리를 frontier와 가르는 요인 **아님**.
- **★우리의 *초과* 실패 = item/variant (F2 symbolic)** — 13 vs 2 (**6.5×**).
- (o4-mini의 지배 실패는 missed_write=행동 안 함 — 다른 실패양식.)

⇒ **frontier 격차 ≈ F2(기호적 operand·기준선택).** 그리고 F2는 우리가 **싸게 닫힌다고 이미 측정한** 축이다
  (criterion 격리정확도: 32B base .727 → prompted-CoT .795 → big-budget .807 → QwQ native **.864**).

## 4. ★재프레임 — 부작용은 레버의 *적용 범위*가 만든다 [D·논증]
F2의 닫개는 thinking인데, thinking은 persistence를 판다(B1·QwQ 12승=12패). 그래서 막혀 있었다. 그러나:
- QwQ의 persistence 붕괴는 **에이전트 루프 안에서** "정책상 못 하니 escalate"로 스스로를 설득한 결과다(`QWQ_FORENSIC §7b`).
- **"어느 변형이 이 기준에 맞나?"만 묻는 *격리된 sub-call*은 메인 에이전트의 escalation 정책을 건드리지 않는다.**
- MAKEORBREAK 실측: 변형-pick **GIVEN-SPEC = 88/88(100%)** · GOAL-only = 70%. ⇒ 모델은 **기준이 명시되면 실행은 완벽**.
  잔여는 **NL→기준 형식화**뿐.
- ⇒ **형식화는 격리 thinking sub-call이 사고, 실행은 결정론 compute가 오프로드** → **persistence 부작용 채널이 원리적으로 닫힘**
  (메인 에이전트는 사고하지 않음). = `LOAD_REDUCTION_ARCH` **E1 context-isolated formalize 서브콜**.

**일반 원리(가설)**: *레버의 부작용은 종종 레버 자체가 아니라 **적용 범위(scope)** 에서 온다. 전-궤적 적용은 부작용을 낳고,
결정점 격리는 그 채널을 닫는다.* → 제1원리의 정련: **"어디에 적용하는가"가 부작용을 결정한다.**

## 5. 함의 (사용자 3문 답)
1. **frontier급 compliant 가능?** 성공률은 아직 아님(0.640 vs o4-mini 0.693). **단 격차의 정체=F2**(닫히는 축)이고 horizon도
   ⋈ 경계도 아님. **보장(위반 0·난이도 무관)은 이미 frontier 초과.**
2. **부작용 없이 장점만?** 일반적으로 아니오. **단 (a) decidable 기능**(compliance·실행 compute)**은 공짜**, **(b) 부작용이
   scope에서 오면 격리로 닫힌다.**
3. **접근 타당?** 예 — 이 프레임이 아니었다면 "격차=horizon=못 닫음"이라는 **틀린 비관**에 빠졌을 것. 측정이 반증하고
   격차를 닫히는 축에 정확히 위치시켰다.

## 6. 원장 갱신 · 큐 재정렬
| | 변경 |
|---|---|
| **C5** (scale=horizon) | [S-lit] 유지 · **단 우리 벤치서 horizon 복리 미관측**(p_step≈1.0) → tau2-retail서 F6은 잔여 아님 |
| **신규 C9** | **frontier 격차 ≈ F2 symbolic operand**(⋈ 경계는 frontier와 공유) — **[M·소표본]** |
| **신규 C10** | **레버 부작용은 scope에서 온다**(전-궤적 thinking=persistence 매도 / 결정점 격리=채널 폐쇄) — **[D]** |
| **큐** | **E1′(격리 formalize 서브콜·F2) 신설·우선순위 상향**. 기존 E1(완결게이트·F4/F5)은 더 작은 잔여 + 자기-역효과 보유 → 후순위. |

## 7. 정직한 단서
- H 3-4 task=17·실패 sim 37/24 = **소표본**. [M]이지 [S] 아님.
- H = gold action 수 = 거친 복잡도 프록시(엔티티 수·⋈ 여부를 혼동).
- "격리 sub-call은 persistence 부작용 0" = **[D] 논증**(미측정). E1′가 검정.
- frontier 궤적은 공식 파일(우리 채점기 아님) — 실패 유형 분류는 동일 로직으로 재계산했으나 하네스 차이 잔존 가능.
