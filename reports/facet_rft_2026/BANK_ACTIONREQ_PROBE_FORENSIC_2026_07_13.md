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

## 7. 산출물
- 코드: `t2_gate_patch.py`(action-required 리마인더 채널·offline 14/14 유효) · `bank_actionreq_probe.sh`(드라이버·KB키+audit 수정).
- sim: `sim_results/bankar_smoke5b.results.json.gz`(KB-dead·무효) · `bankar_smoke5c.results.json.gz`(KB정상·본 분석).
- 등급: **[M]·소표본(5 sim)이나 구조증거(requestor split·gold 결정론)는 [S]-급**.
