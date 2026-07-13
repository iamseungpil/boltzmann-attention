# banking action-required 라이브 probe 포렌식 — 레버 스펙 오정합 발견 (2026-07-13 LATE-2)

> 로컬 무료 유도 + 소액 유료 스모크(15 sims·nt1·gpt-4.1 user-sim·[[09]] 승인)로 **action-required 레버가
> banking에 오정합**임을 확정. 40-태스크 full probe는 착수 전 중단(스퓨리어스 발화만 확인·예산낭비).
> 정본. RESEARCH_MASTER §3 원장에 등재.

## 0. 한 줄
**핸드오프의 "banking act-vs-advise 36% = 최대 scaffold 레버"는 아티팩트.** banking gold action의 대다수는
`requestor='user'`(에이전트가 *직접 못 부르는* user-실행 도구)이거나, agent-실행이라도 `call/unlock_discoverable`
(발견체인)이다. action-required(=에이전트가 조언 대신 action_tool 호출 강제)는 (a) user-실행 도구엔
**구조적으로 발화 불가능한 요구**이고 (b) "에이전트가 apply 안 부름 + 마무리 텍스트"를 회피로 **오분류**한다.
실제 banking 실패 = **⋈ 오선택(잘못된 카드/도구) + 발견/reach**(C52 정합), act-vs-advise 아님.

## 1. 실측 (bankar_smoke5c·GR arm·5 표적 task·KB 정상)
- 구성: `--gate 1` + auth 게이트 + prov-rescue + **T2_RESOLVE=1**(action-required). infra 1/5·KB 에러 0(수정 후).
- **action-required 발화 9회 / 5 sims** — 라이브 배선 작동(리마인더 채널 정상). 단 **pass 0·행동 무변화**.
- 표적 5개 gold: task_001/003/007 = `apply_for_credit_card`(user-실행) · 005 = change/log_verif/transfer · 016 = submit_transaction(user-실행).

## 2. 궤적 정독 (task_003 전문·task_007 db_check)
- task_003: 에이전트가 KB_search·`give_discoverable_user_tool`로 카드 **발견·제안** → **user(role=user)가 apply_for_credit_card 실행**(msg15) → "application submitted"(msg16). 즉 **행동은 일어남**.
- 실패 원인 = **잘못된 카드**: gold `card_type='Silver Rewards Card'`인데 user가 `'Business Platinum Rewards Card'` 신청 → db_match=False. = **⋈/reference 오선택**(회피 아님).
- task_007 동일: gold `card_type='EcoCard'`·`action_checks[].requestor='user'`·`ACTION-EXEC role=user`.
- ⇒ 에이전트의 마지막 텍스트("Great choice!")는 **user가 이미 실행한 뒤의 마무리 인사** — 오프라인 estimate가 이걸 "deflection"으로 셌다.

## 3. 구조 증거 (floor 97 task gold action × requestor·결정론)
| action_tool | requestor | 건수 |
|---|---|---|
| apply_for_credit_card | **user** | 19 |
| submit_referral | **user** | 7 |
| submit_transaction | **user** | 1 |
| call_discoverable_agent_tool | assistant | 428 |
| unlock_discoverable_agent_tool | assistant | 275 |
| change_user_email | assistant | 1 |

- **user-실행**(에이전트 호출 불가·발견/추천만): apply/submit_referral/submit_transaction.
- **agent-실행**: 대다수가 `call/unlock_discoverable`(발견체인·operand=agent_tool_name·operator resolution) = **reach/discovery** 문제이지 execute-vs-advise 아님.

## 4. 오프라인 36% 상한의 정체 (아티팩트)
`bank_actionreq_estimate.py`/target-40 도출 = "gold action_tool 있음 ∧ 에이전트가 그 도구 미호출 ∧ 마지막 asst 턴 회피(텍스트/transfer)". **apply류가 user-실행이라 "에이전트 미호출"은 항상 참** → user-실행 gold를 가진 태스크가 무조건 target으로 잡힘 + user 실행 후의 마무리 텍스트를 회피로 오분류. ⇒ 36%는 레버 사정거리가 아니라 **분류 오류**.

## 5. 자기교정 (핸드오프 전제 철회)
- 핸드오프 `2026-07-13 LATE §3` "REACH 오진단 철회 → action-required 36%"는 **재-철회**: 초기 REACH 분석은 KB-death에 오염됐고(task_001 KB-dead=웹사이트 조언), act-vs-advise 재프레임은 user-실행 구조를 놓쳤다.
- action-required 코드/리마인더 채널 자체는 정상(offline 14/14·live 발화 확인) — **문제는 banking A2 action_tools가 user-실행/agent-실행을 뭉뚱그린 것**. 레버가 아니라 스펙.

## 6. 함의 / 다음 후보
- **action-required의 유효 사정거리**(있다면) = agent-실행 도구(`unlock/call_discoverable`)를 조언으로 회피하는 경우뿐. 단 그건 **reach/discovery**(GET→FIND 발견체인)와 구분 필요 — 별개 레버(controller)일 개연.
- banking binding = **reach/coverage/horizon + ⋈**(C52 재확인). scaffold 후보 = discovery controller·coverage 게이트, action-required 아님.
- 결정: 40-태스크 유료 probe **NO-GO**(스퓨리어스 확인·예산낭비). retail 스택 확정 우선(E-XFER-bank 재시퀀싱 유지).

## 6b. ★재채점 + gold↔우리 경로 분석 (사용자 지시 2026-07-13·`bank_rescore_pathdiff.py`·`bank_pathdiff_percase.py`)
> buggy audit(assistant-only 스캔) 폐기 → 하네스 `action_checks[].action_match`(정확 채점) + 궤적 대조.
> **frontier 궤적은 소실([[47]])** — aggregate만: opus4.5 pass1 24.7·gpt5.5 37.4 vs 우리 floor **6.1%**(목표격차).

**재채점 (bankxfer_floor_bank_t4·198 valid·infra 13)**: PASS(reward=1.0) **12/198 = 6.1%**. reward_basis: DB 172·ACTION 22.

**실패 gold-action 세분 (action_match=False·전 action)**:
| 분류 | 건수 | 정체 |
|---|---|---|
| **(A) 필요 도구 미호출** | **580** | REACH/discovery 미완 or 조기 give-up(transfer) |
| **(B) operator/operand ⋈ 오선택**(핵심키 틀림) | **509** | 도구는 불렀으나 wrong agent_tool_name/card_type |
| (C) 정확 도달·타인자/기준 미스 | 152 | 올바른 operator/operand·하위인자 or 기준 오류 |
| (D) 기타 인자 | 97 | — |
- sim 지배: ⋈ 오선택 115 > REACH 64 > NO-START 4. 태스크 95개 중 89 전패·6 부분pass.

**per-case 정독 (3건·[[08]])**:
- **task_003 = operand ⋈**: 에이전트가 카드 발견·제안·user 실행되나 gold `Silver Rewards` vs user가 `Business Platinum` 신청 → wrong operand(카드).
- **task_023 = 검증(F1)+조기 escalation(F5)**: gold=`log_verification`→apply. 우리는 `get_user_information_by_*`로 검증 시도했으나 **log_verification 미완**→`transfer`로 포기(gold apply 미도달).
- **task_035 = operator GET 실패**: gold=`emergency_credit_bureau_incident_transfer_1114`을 KB발견→unlock→call. 우리 KB질의("credit score discrepancy")가 그 도구명 못 찾음→lookup만 하다 transfer.

**결론(사용자 직관 확증)**: banking 실패 = **{operand+operator} 해소**(경계 아님·act-vs-advise 아님). **retail과 동일 GET→FIND→INFER→ASK 루프**·ABox만 다름(banking `operator_resolution=discoverable`=KB-GET 단계 추가·retail=direct). 도메인-일반 레버:
1. **operator/operand FIND**(⋈ 509+152): 의도/요구→도구·카드 formalize→결정론 select — **`formalize_intent_tool` 재사용**(retail fexec 동형·U2/[[00]]).
2. **discovery controller**(REACH 580 A): 의도가 discoverable 도구 필요 시 KB-GET+unlock/call을 give-up(transfer) 전 강제 — retail E-PLAN discovery-precondition 동형.
3. **verify/persistence 게이트**(F1/F5·task_023): log_verification 완결 강제 + 조기 transfer 차단 — 기존 auth 게이트+persistence.
⇒ **action-required 아님.** banking↔retail 통일 = t2_resolve 디스패처(operator/operand kind)·formalize_intent_tool·eplan·gate 재사용, ABox(banking A2)만 교체.

## 6c. ★통일 스택 구현 (사용자 지시 "banking 해결·retail과 일반화 합치자"·세 레버 순서대로)
> 전부 도메인-일반 scaffold + banking A2 데이터([[05]]클린)·오프라인 유닛 검증(21/21 suites)·엔진 리터럴 0.

- **Lever 0 (requestor 마킹)**: A2 `action_tool_executor`(user/assistant). action-required는 agent-실행만 대상 + **고정밀화**(formalize가 구체 target 낼 때만 발화·target=none action-ask 미발화=Δspurious). `test_action_reminder` 16/16.
- **Lever 1 (operator FIND·⋈ 509)**: `resolve_operator`에 FIND 추가(A2 `find_intent`). 발견 후보 ≥2 중 `formalize_intent_tool`(재사용)로 의도-매칭 도구 검증·선택≠formalize면 deny(operator-find). FAB(미발견)는 기존. `test_operator_find` 9/9.
- **Lever 2 (discovery controller·REACH 580)**: reach 실패 55/55가 조언종료→action-required 100% 발동 실측(별도 컨트롤러=게이트증식 회피). target=discoverable dispatcher면 **discovery-required** 피드백(getter→unlock→call 발견체인 안내). `test_action_operator` 9/9.
- **Lever 3 (verify/persistence·F1/F5·26 sims)**: A2 verify 게이트 `verify_gather_prefix`. 신원수집(get_user_information_by_*)+검증(log_verification)미완+포기 시 완결 리마인더. action-required 미발동(apply=user-실행) 케이스 보완. `test_verify_persistence` 8/8·live 4/4.
- **통일**: 전 레버가 T2_RESOLVE=1 단일 경로·retail은 `action_tool_executor`/`find_intent`/`verify_gather_prefix` 미기재→자동 무발동(하위호환). `formalize_intent_tool`·`resolve_operator`·auth게이트 재사용 = banking↔retail 통일.
- **드라이버**: `bank_actionreq_probe.sh`(양 arm prov 공통·GR만 T2_RESOLVE·3레버 audit).
- **★라이브 스모크(bankar_uni5·GR arm·5 태스크·nt1·[[30]])**: **infra 0·크래시 0·pass 1/5(task_023)**(floor 0/5). 발화 L1 operator-find 3·L2 discovery-required 4·L3 verify-persistence 6. **task_023 causal 확인**(궤적 정독): floor=log_verification 미완+포기로 fail → GR=신원수집→**log_verification 호출**→user 올바른 카드 apply·gold 2/2 match=**Lever 3 의도와 정확 일치**([M]·n=1·user-sim 변동 배제는 다표본 필요). **task_035**: L2 발화했으나 emergency 도구 KB-discovery 실패(reach 잔여=semantic·L2 리마인더로 미폐쇄). ⚠**궤적 길어짐**(태스크당 450s+·리마인더가 추가 검색/검증 유발·비용↑·단 tme 폭증 없음·term 전부 user_stop). **다음=소액 probe(G vs GR·표적 nt1·[[09]] 승인)로 순 pass 이동+Δspurious+비용 측정.**

## 7. 산출물
- 코드: `t2_gate_patch.py`(action-required 리마인더 채널·offline 14/14 유효) · `bank_actionreq_probe.sh`(드라이버·KB키+audit 수정).
- sim: `sim_results/bankar_smoke5b.results.json.gz`(KB-dead·무효) · `bankar_smoke5c.results.json.gz`(KB정상·본 분석).
- 등급: **[M]·소표본(5 sim)이나 구조증거(requestor split·gold 결정론)는 [S]-급**.
