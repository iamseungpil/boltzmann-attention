# thinking 보강 — completion/persistence scaffold 설계 (2026-07-08·리뷰용)

> **위치**: `QWQ_AGENTIC_FAILURE_FORENSIC_2026_07_08`(thinking 약점=조기종료/give-up) + `LOAD_REDUCTION_ARCH_DESIGN §4`(E2
> 완결-게이트) + DR(`RELWORK_AGENTIC_HORIZON`: 조기종료·external-verifier)의 수렴. thinking을 배포-lever로 살리는 보강 설계.
> **불변**: [[05]] scaffold 도메인-일반·A2만·게이트 증식금지 · [[13]] scaffold 최소·smoke먼저 · [[08]] per-case·offline먼저
> · [[09]] 무료先·gpt-4.1 최종1회 · [[03]]#9 over-block=대칭크레딧·실측前 banking금지 · [[10]] verifier=결정론·LLM=formalize.

---

## 0. 한 줄
thinking(추론모델)이 friction서 **스스로를 "정책상 못 하니 escalate/deny"로 설득**해 조기종료→ 완결 실패한다. **외부 결정론
완결/persistence 게이트**(E2 확장)가 종료-시도를 가로채 미완이면 deny하여 **모델의 decision 이득은 유지·persistence는 scaffold가
보장**. 목표 = QwQ+게이트 > base(현 parity 0.526 vs 0.557)를 over-block=0로 실증 → thinking+scaffold=net-positive lever.

## 1. 문제 (forensic-도출·근거=QWQ forensic §7/§7b)
thinking-model(QwQ-32B·reasoning-parser)은 base와 **parity**(0.526≈0.557)다. 원인=**12승(per-decision 정확도)=12패(완결/
persistence 붕괴)**. 패배 12의 스텝별 분류 → **3 슬라이스**:
| # | 슬라이스 | 예(task) | 내부추론 서명 | scaffold? |
|---|---|---|---|---|
| ① | **premature-transfer**(거의 안 해보고 포기) | t13·t30·t106 | *"if I can't handle it, I should transfer"*·탐색 1~3회 | ✅ persistence 게이트 |
| ② | **universal-coverage 미완**(all/both 다 안 함) | t42·t55·t87 | 일부 write만·미커버 엔티티 잔존 | ✅ coverage 게이트(E2) |
| ③ | **mis-formalize**(task 오독→포기) | t49·t87 | item-match 실패를 "없음"으로·목표주소를 검색으로 오해 | ❌ **learn 몫(범위 밖)** |
- **핵심**: ①② = 순수 완결/persistence 부하(모델은 *할 수* 있는데 안 함) → 결정론 scaffold 대상. ③ = 요청 *오해*(formalize
  실패) → 게이트 불가·학습-wing(G2). **본 설계 = ①② 한정**(③ 명시 제외).

## 2. 방법 확정 (DR + LOAD_REDUCTION)
- **DR(`RELWORK_AGENTIC_HORIZON`)**: long-horizon서 coverage/조기종료 1%→25%(capability≠reliability)·**self-reflection(외부
  피드백無)은 개선 안 함[2310.01798]→ external gate/verifier가 답**. ⇒ 내부(prompt/self-critique) 아니라 **외부 결정론 게이트**.
- **LOAD_REDUCTION_ARCH §4(E2)**: 완결 게이트=universal 양화사→결정론 enumerate working-set→미커버 시 종료-deny. 본 설계는
  E2를 **재사용**(②)하고, thinking-특이 ①(premature-escalation)을 **대칭 primitive**로 추가.
- **thesis 정합([[10]]/[[00]])**: 모델=decision(boundary translator)·게이트=persistence/완결(결정론 verifier). = 우리 논지 그대로.

## 3. 공통 primitive — entity-coverage (도메인-일반)
①② 모두 같은 원리: **사용자의 관련 엔티티 집합을 결정론 enumerate → 관측(inspection/action) 커버리지 추적 → 미커버 상태서
종료-시도 시 deny**. 차이는 커버 대상:
- ① = **inspection 커버리지**(give-up 전 후보 엔티티 다 *봤나*)
- ② = **action 커버리지**(완료 전 대상 엔티티 다 *처리했나*)
- **working-set 도출(A2-구동·[[05]])**: `coverage_spec{entity, predicate_source}` — 예 {entity=user의 orders, predicate=대화서
  LLM-formalize한 필터(pending·특정품목 포함 등)}. **enumerate=코드(DB)·predicate formalize=LLM·gold/eval 미접근**. retail 토큰 0.

## 4. Gate P — persistence (anti-premature-escalation·①)
- **트리거**: agent가 `transfer_to_human_agents` 시도 OR actionable 요청에 무행동 종료.
- **DENY 조건(보수적)**: 사용자 candidate 엔티티 중 **미검사(inspection 안 한) 것이 존재** AND 종료사유가 "not found/can't"
  계열. = "다 안 보고 포기"만 차단. (t30=최신 1주문만 보고 포기·t13/106=탐색 1~3회.)
- **행동**: deny + regen 넌지("You have not inspected all candidate orders/items; continue before escalating.")·`gen_gated` deny경로 재사용.
- **anti-cheat**: transfer가 **gold인 task 절대 미차단**(t10/12/25/46/50=out-of-policy 요청→transfer 정답). ⇒ **모든 후보를 이미
  검사한 뒤의 transfer는 통과**(그건 정당 escalation). 애매하면 gate off(over-block 회피 최우선).

## 5. Gate C — coverage/completion (E2 재사용·②)
- **트리거**: user turn에 **명시 universal 양화사**(all/both/every/each … orders/items) — 좁게.
- **working-set**: predicate formalize(LLM)→ 결정론 enumerate(DB).
- **DENY 조건**: 종료/transfer 시도 시 working-set에 **미-action 엔티티 잔존**.
- **행동**: deny + regen("You still have {remaining} orders to process.").
- **anti-cheat(§LOAD_REDUCTION §4.3)**: 명시 양화사 + 올바른 엔티티레벨만·암묵 scope("고쳐줘"·단수)=미발화·predicate 애매→gate off.

## 6. 명시 제외 (범위 밖)
- **③ mis-formalize**(t49·t87): 요청 오독 = formalize 실패 → 게이트 불가·**learn-wing(four-bench→τ² swap)** 귀속.
- ⋈/variant operand 경계(QWQ forensic §5): 게이트 아님(결정 자체)·별도.
- prompt-only persistence 넌지 단독 = **금지**([[42]] prompt-eng 약함·DR self-reflection 무효). 넌지는 *deny와 결합*해서만.

## 7. anti-cheating / 도메인-일반 감사 ([[05]])
1. working-set enumerate = agent가 가진 정보(DB fetch·present)만·**gold/eval/DB-정답 미접근**.
2. 게이트 = **종료 차단만**(답·operand 미주입)·모델이 여전히 무엇을·어떻게 결정.
3. 트리거·enumerate·넌지 = 도메인-일반("candidate entities"·"universal quantifier"·retail 토큰0). `grep "if domain"=0`.
4. 도메인 추가 = `coverage_spec` A2-swap만(신규 코드 0).

## 8. 측정 프로토콜 (offline→smoke→확인·[[09]])
- **Phase A (offline·무료·즉시)**: 기존 QwQ `qwq_rparser_floor_nt1` 궤적에 Gate P/C **replay** — (a)give-up LOSS(①② 슬라이스)
  **정확 발화율**(recall) (b)**passing control over-block=0**(대칭크레딧). ★옛 naive 게이트=over-block 19(실측)→ **정밀 조건(§4/5)
  으로 재측정 필수**. GO 문턱=recall 유의 & over-block 0(or 극소).
- **Phase B (live-smoke·무료 user-sim)**: Gate P/C 배선 → QwQ ①② fail-set(N~10) + control(passing 10)·nt=1·per-case →
  **pass 복구 + over-block=0**. (배선=`t2_gate_patch`+`gate_interpreter` coverage kind.)
- **Phase C (유료 gpt-4.1·GO 후 1회·승인)**: QwQ+게이트 nt=1 or nt=4 full → base 대비 net(>0.557?).
- 지표 = per-case·pass^1(robust)·gold-write diff·**over-block(control 회귀)=0 강제**.

## 9. GO / NO-GO
| 결과 | 해석 | 다음 |
|---|---|---|
| Phase A recall高 + over-block 0 | ①② = 순수 완결부하·게이트로 복구 | Phase B 배선 |
| Phase B pass복구 + over-block 0 | 완결-scaffold 유효·QwQ>base 경로 | Phase C 확인 |
| over-block>0 (control 회귀) | 게이트가 정상 궤적 훼손 | 발화조건 축소·재측정 (or 폐기) |
| recall低/복구無 | 잔여≠순수 완결부하(=③ 오독·경계 지배) | learn-wing G2·게이트 보류 |

## 10. base와의 관계 (일반성·thesis 강화)
E2(②)는 **원래 base의 coverage 잔여**(CLEAN_NT4 M1)용으로 설계됨. thinking은 **같은 클래스 약점을 증폭**(조기종료). ⇒ **동일
도메인-일반 게이트가 base·thinking 양쪽 완결부하를 커버** = QwQ-특이 패치 아님·일반 scaffold. thinking+게이트가 base+게이트를
넘으면(decision 이득 유지) thinking=진짜 net-positive lever 확증.

## 11. ★리뷰 대기 (open questions)
1. **Gate P "exhaustion" 신호**: (a)inspection-커버리지(모든 후보 엔티티 검사) vs (b)turn/read-count 문턱 vs (c)둘 다. 권장=(a)
   (도메인-일반·enumerate 재사용)·단 "후보 엔티티" 정의(user의 전 orders? 요청 품목 포함가능 orders?)가 관건.
2. **P/C 통일 vs 분리**: 둘 다 entity-coverage니 **단일 게이트(kind=coverage·mode=inspect|act)**로 통합? 권장=통합(게이트 증식
   방지·[[05]]).
3. **override 강도**: hard-deny+regen(침습적·측정 깔끔) vs soft 넌지-후-재생성. 권장=deny+regen(E2 일관).
4. **over-block 허용치**: 0 strict vs 극소 용인(control 회귀 1~2). 권장=0 목표·>0면 조건축소.
5. **persistence가 flail로?**: 종료 막으면 QwQ가 헛도는(turn 낭비·여전히 fail) 위험 → Phase B가 판정(복구 vs flail). offline은
   recall만·복구는 못 봄(대칭크레딧: shaped≠closed).
6. **scope**: Gate C(②·E2 재사용·저위험) 먼저 vs P+C 동시. 권장=**offline Phase A 둘 다 먼저**(무료)→ recall/over-block 보고 배선순서.
7. **테스트베드**: QwQ(thinking) 우선 vs base도(일반성). 권장=QwQ 먼저(약점 큼)·GO 시 base 회귀로 일반성 확인.

## 12. 시퀀싱
1. **Phase A offline(무료·즉시)**: Gate P/C 정밀조건 replay → recall + over-block. (도구=scratchpad·기존 궤적.)
2. 리뷰(recall·over-block) → 통합게이트 배선 여부·조건.
3. **Phase B live-smoke(무료)**: 통합 coverage 게이트 배선 → QwQ fail-set+control·per-case.
4. GO → Phase C(gpt-4.1 1회·승인). NO-GO → learn-wing G2.
