# 결과 정리 — 043 close-chain 실패 조사 (2026-07-24~25)

> 종합 문서. 정본=`RESEARCH_MASTER.md §3 원장 C135~C160`. 이 문서는 그 아크를 결과 중심으로 재구성.
> 등급: [S]=궤적/전수 포렌식 검증 · [M]=측정(caveat 명시) · 수치는 전부 원장 provenance.
> 성격: **하나의 실패(task_043 "카드 닫기")를 정보-맞춘 격리로 끝까지 파고든 조사**. 표면 귀속을 6회+ 반증.

---

## ⚠️ 정정 고지 (2026-07-25 후속·C159/C160) — **이 문서의 §0·§2·§4 핵심 주장은 반증됐다**

DB-diff 전수 재검(`dbdiff_nt4.py`·원장 C159)에서 아래가 확정됐다. **아래 절들을 읽을 때 반드시 이 고지를 함께 읽어라.**

1. **"nt4에도 close-before-retention이 남아있다"는 판정이 매칭 아티팩트였다.** `close_credit_card_account`를 **인자 문자열**로 매칭해 close *실행*으로 셌으나 실제는 **`unlock_discoverable_agent_tool`**(잠금해제)였다. dispatcher 실행만 추리면 **nt4 4/4 trial 전부 close 미실행**이고, **DB-diff가 독립 확증**한다(`credit_card_accounts.status` 차이 0건). §2.1의 "close@44 < apply_flag@56"은 sim0의 **unlock@44 · apply@60**이다.
   ★**단 현상 자체는 초기 런서 실재했다(C161)**: rall24a CLOSE@95·rall25a CLOSE@84·reg043_base/treat·fix_base 모두 **실제 close 실행**. 즉 **기전은 실재**했고(C142/Track-C 주장은 자기 데이터서 타당), **틀린 것은 "nt4에도 남아있다"는 판정**이다.
2. **게이트는 soft가 아니라 성공이었다(중요·긍정).** 로그에 pre-close deny **16회 발화**·close 실행 0. 교차-런 matched pair(C161): 게이트 **無**=close 실행(fix_base@52) vs 게이트 **ON**=close **0**(fix_treat·dd·pr·nt4 전부) ⇒ **게이트가 표적 현상을 인과적으로 소거**. §4 표의 "pre-close 게이트=soft(nt4서 여전히 close)" 행은 반증되고, **[S]급 성공 사례로 재분류**된다. (C152 *discovery*-게이트 soft는 별건이라 유지.)
3. **진짜 잔여 blocker는 부기(bookkeeping)였다.** sim0 ndiff=2·sim2 ndiff=1이고 **차이는 전부 `agent_discoverable_tools` CALLED 레코드 누락**(직접호출 미등록·C149). 상태(잔액·flag·status)는 gold와 일치 ⇒ **행동은 정답인데 등록 부기로 탈락**.
4. **범위 한정(C160)**: 97태스크 DB-diff 재집계(공식 db_match와 mismatch 0 검증)에서 "부기만 누락"은 **4%**뿐이고 87%는 실제 상태차를 동반 ⇒ **위 3은 043 국소 현상**이며 벤치 일반의 지배 실패는 **여전히 행동**(C150/C151 유지).
5. **§3.1/§7의 L4-계열 수치도 과대계상**: C158 재측정서 A2-index 조건의 "올바른 waiver 회상"은 **0/16**(정규식이 "annual fee+retain"에 걸린 과대계상). 또 오프라인 프로브의 **"CLOSE" 지표는 태스크 실패를 예측하지 않는다**(라이브서 같은 unlock 행동이 게이트 개입으로 정답 궤적이 됨).

**살아남는 것**: §1 방법론(정보-맞춘 격리)·§5.3 자기교정 사슬·§5.4 db_match brittleness·C142의 clean-vs-polluted 해리(별개 런)·§6 직접호출 규명(오히려 승격).

---

## 0. 한 줄 결론 ⚠️**(정정됨 — 위 고지 참조)**

> **구 결론(반증)**: "실패의 진짜 원인은 reactive-execution(retention 前 성급 close)이고, 결정론 게이트는 soft다."
> **현 결론(C159/C160)**: nt4에서 **close는 일어나지 않았고 게이트가 막았다**. 남은 gap은 정책-회상 실패가 아니라 **discovery 호출 경로 부기**(직접호출→미등록)였다 — 단 이는 **043 국소**이며, 97태스크 규모에선 여전히 **행동 실패(잘못된/누락 write)가 지배**(87%)한다.

task_043(Platinum 카드 해지 요청·수수료 waiver 수락→카드 유지가 gold)에 대한 아래 §2~§4의 기전 서술은 위 고지대로 읽어야 한다.

---

## 1. 조사 방법론 — 정보-맞춘 격리 귀속

한 실패의 원인을 "표면 귀속"이 아니라 **정보-맞춘 격리 프로브**(A_minimal 깨끗맥락 vs B_full 오염맥락)로 판정([[18]]/[[08]]). 이 세션은 그 방법론의 자기-실증이 됐다 — **표면 귀속이 매 단계 틀렸고, 격리를 끝까지 밀어서야 진짜 원인에 도달**. 저자 자신의 성급한 결론도 3회 교정(C135·C139·C140).

| 원장 | "원인"으로 보임 | 격리/포렌식 후 |
|---|---|---|
| C135 | plan 능력 결손 | ⚠**철회** — 프로브 맥락에 절차가 애초 부재였음(과다귀속) |
| C137 | scale-불변 능력 결손 | ⚠과다귀속 |
| C138 | 정보-부재 | 절차 주면 조립됨(정보였음) |
| C139 | salience/position | ⚠부분교정 — BM25 정상·문서 retrieved됨 |
| C140/141 | lost-in-the-middle | ⚠반증 — position 무관·**presence가 지배** |
| **C142** | — | **execution 확정**: 깨끗맥락서 모델 apply_flag 추론·계획·실행 다 함(E0 4/4)·라이브만 건너뜀 |
| **C155** | — | **[[18]] 격리 판정=LOAD**(능력 경계 아님·재부각 회복) |

---

## 2. 핵심 발견 1 — 실패 기전 ⚠️**부분 정정 (C159/C161)**

> **기전은 실재했다. 틀린 것은 "nt4에도 남아있다"는 판정이다.**
> · **실재 근거(C161)**: rall24a **CLOSE@95**·rall25a **CLOSE@84**(둘 다 apply 無)·reg043_base/treat·fix_base = **실제 close 실행**.
> · **nt4 실측(DB-diff 4 trial 전수)**: close 실행 **0/4**·`credit_card_accounts.status` 차이 **0건**·apply_flag **3/4 실제 실행**(sim0@60·sim1@84·sim2@60). "close@44"로 읽은 것은 **unlock@44**.
> · **차이를 만든 것 = pre-close 게이트**(C161 matched pair: 게이트無 close 실행 / 게이트ON close 0).

### 2.1 무엇이 일어나는가 — 정정된 서술
- **043 nt=4 pass = 0/4**(전부 db_match False). ← 유효
- ~~4 trial 중 3+가 retention 완료 前 카드 close~~ → **nt4에선 0/4**. 실제 궤적은 **unlock→게이트 deny→선행 read 수행→apply_flag 실행**.
- **초기 런(게이트 이전)에선 그 패턴이 실재**했고, 그것이 이 조사를 시작시킨 진짜 현상이다.

### 2.2 능력이 아니라 LOAD (C155·[[18]] 3조건 격리)
trial0(close_idx44·정책 1276자 추출) 정보-맞춘 격리:
| 조건 | RETAIN(올바름)/6 |
|---|---|
| A_minimal(깨끗+정책) | **6/6** |
| B_raw(긴문맥·정책 묻힘·중립) | **3/6**(+other 3 = 라이브 재현) |
| B_resurface(긴문맥+정책 재부각) | **4/6**(CLOSE 0) |

⇒ **정책이 salient하면 올바로 retain·긴 문맥서 묻히면 close·재부각/focus가 회복** = 능력 경계가 아니라 부하. **개선 가능**.

### 2.3 왜 "정보"가 아닌가 (C138·C142)
모델은 close 절차(doc_003 Step4 수수료 waiver)를 **실제로 retrieve했고**(BM25 정상·C139), 깨끗맥락서 주면 **완전한 plan을 냄**(E0 4/4·C142). 문서가 27K 오염 맥락에 **이미 있는데도** 미실행(C146 R_none 0/6) = **salience 문제**이지 정보/retrieval/능력 아님.

---

## 3. 핵심 발견 2 — 처방: 무엇이 회복시키나

### 3.1 재부각(source 정책문서)이 회복 (C146 make-or-break)
rall25a 043 close-직전 컷·multi-step 프롬프트(single-next 아티팩트 제거):
| 조건 | apply_flag | dispute |
|---|---|---|
| R_none(재부각 없음) | 0/6 | 0/6 |
| R_chain(compact 이름) | 3/6 | 6/6 |
| R_reground(실제 정책문서) | **5/6** | 6/6 |

- **단순 read(dispute_history)**: compact 이름만으로 6/6 회복.
- **정책-조건 write(apply_flag)**: 재부각으로 0→5/6.
- ★**집계가 가린 질적 차이(원시 completion 정독)**: R_chain의 apply_flag 언급은 **틀린 의미 앵무새**("계정 닫힘 표시 flag"로 오해=authority/인자 소실)·R_reground은 **올바른 정책추론 동반**("annual fee+2년 tenure→수수료 waiver"). 즉 문자열 카운트(3 vs 5)는 doc 가치를 **과소평가** — 진짜 기여는 언급률이 아니라 **의미 정확성**.

### 3.2 reasoning(비-scaffold)이 부분 회복·격리보다 안전 (C156)
"정책이 문맥에 묻혔을 뿐"이라, **행동 前 정책 추론 지시**만으로 자기회상:
| 조건 | RETAIN/6 |
|---|---|
| B_raw_neutral(중립) | 0 |
| B_raw_REASON(추론 지시·재부각 無) | **3** |
| B_resurface(재부각) | 4 |
| A_minimal(깨끗격리) | 6 |

- **reasoning = test-time compute**([[13]])·**전체 문맥 유지→C129 안전**(격리는 문맥손실로 해로운 오결정 위험·REF_ISO C129 교훈).
- ⚠caveat: 3/6=부분회복·추론 프롬프트가 다소 leading·단일 trial.

### 3.3 A_minimal 격리는 왜 최선이 아닌가 (C156)
① 자동구성 딜레마(도메인 큐레이션=scaffold or 모델 요약=오염 재발) ② **C129/REF_ISO 위험**(격리 시 문맥손실→해로운 오결정). ⇒ **컨트롤러는 재부각만·메인이 결정**(격리 sub-agent가 결정하면 위험).

---

## 4. 무엇이 hard이고 무엇이 soft인가 (guarantee의 성격)

이 세션의 관통 교훈: **deny+regen 게이트는 "hard 제약"인 척하나 실제로는 soft**(에이전트 준수 의존·[[07]]/[[42]]).

| 접근 | 결과 | 성격 |
|---|---|---|
| ~~pre-close 게이트(deny+재부각)~~ | ~~발화하나 nt4서 여전히 close·에이전트 무시~~ ⚠️**반증(C159)**: deny **16회 발화**·close 실행 **0/4** ⇒ 실제로 차단함 | ~~soft~~ → **작동(hard에 준함)** |
| DISCOVERY-DISPATCH 게이트(deny+regen) | 직접호출 1→0이나 **에이전트 포기**(재발행 안 함)→목표 미달성·pay 교란 | **soft**(C152) |
| 프롬프트(명시 규칙) | 32B가 여전히 직접호출·**KB "Use X" prior-override** | **soft**([[42]] 재실증·C153) |
| **reasoning(행동 전 추론)** | close-before-retention 0→3/6 회복 | 부분·비-scaffold(C156) |
| **vLLM guided decoding**(스키마 grammar 제약) | 스키마 밖 이름 hallucinate를 **결정론 차단** | **진짜 hard**(C154·미구현) |

⇒ [[45]] "guarantee는 hard(결정론)가 필요"·[[13]] "형식/reasoning은 scaffold보다 learn/decode 우선" 실증 지지.

---

## 5. 엔지니어링 결과 — BRANCH-REGROUND 구현·라이브·자기교정

### 5.1 구현 (C146·도메인일반·[[05]] 통과)
- `resurface_doc`: 이미 retrieved된 tool 출력서 pending write 명시 정책문서 블록만 헤더인식 추출(doc-id 리터럴 0).
- `branch_reground_reminder`: 남은 read/simple-write=compact 이름 · 남은 정책조건 write=추출 정책문서 재부각.
- 배선: user_stop 경로(`_check_termination`) + **pre-close 생성-레벨 트리거**(finalize write 시도 시 선행 미완이면 deny+재부각). A2=`finalize_writes`. selftest 8/8·회귀0.

### 5.2 라이브 nt1 = 기전은 표적대로 발화 (C147·[S])
matched pair(base vs treat·treat만 T2_BRANCH_REGROUND=1·gpt-5.2):
- pre-close deny 발화(close 시도→선행 미완 deny+재부각).
- **apply_flag: base=미호출 → treat=gold-정확 호출**(`flag_type=annual_fee_waived reason=loyalty_benefit expiration=11/14/2026`=재부각 doc 인자). log_reason base False→treat True.
- ⇒ **재부각 정책문서가 apply_flag을 표적대로 정확 실행**(C146 오프라인의 라이브 확증).
- 단 **db_match=False 둘 다**(task 실패는 별도 원인 → §5.3).

### 5.3 자기교정 사슬 — task 실패 원인 (C147→C148→C149)
- **C147 오귀속**: "pay(75.00)가 막음" → **틀림**. 포렌식 결과 pay는 gold와 완전동일·성공(reward_basis=DB·action_match=False는 무관 아티팩트). **[[08]] 위반(집계만 보고 직행) 자기교정.**
- **C148 진짜 원인**: **close over-action** — retention-accept task(gold에 close 없음)인데 양 arm 모두 카드 닫음. 내 scaffold **2 근본원인**: ①pre-close 훅 버그(직접호출 close가 fam-strip 前 게이트 우회) ②chain 스펙이 close를 required_writes로 강제→정책위반 close를 능동 유도. **둘 다 수정**(훅 항상 fam-strip·close를 required_writes서 제거하되 finalize 게이트엔 유지).
- **C149 검증**: 수정 재런=treat 카드 **안 닫음(x0)** vs base **여전히 close(x1)** ⇒ **게이트가 close 방지의 인과**. 단 db_match=엄격 full-DB 해시(`agent_discoverable_tools` CALLED 포함=read-coverage+정확 suffix+over-action0 다 요구)라 여전히 미달.

### 5.4 db_match 메트릭 재고 (C149·C150)
- db_match는 **매우 엄격/brittle**: 전체 gold action-set(discovery reads 포함)을 **dispatcher 경로로** 호출+writes 정확+over-action 0 다 필요.
- 게이트 스택으로도 **97태스크 db_match = 9/97(9%)**(C150). per-task 신호로 brittle·**per-step action-match이 더 dense**할 수 있음.

---

## 6. 부차 발견 — 직접호출(direct-call) 형식 문제

### 6.1 systemic 아님 = 5% quirk (C150)
- 97태스크 전수: 직접호출 있는 태스크=**5/97(5%)**·db_match 지배 실패는 다른 곳(missing-write 70%·missing-read 57%·C151).
- 무료 확인이 저수율 게이트 구축을 차단([[09]] 규율 실증).

### 6.2 정체 = 스키마 밖 이름 hallucination (C154·결정적)
- 라이브 스키마(`env.tools.get_tools()`)는 **discoverable 직접이름 제외**(dispatcher만). 모델의 직접호출=**KB "Use X_3847" 문구서 스키마 밖 이름 hallucinate**.
- **격리 실증**: 깨끗 스키마+KB 프롬프트→모델이 **auto 모드서도 dispatcher 올바로 사용·직접호출 0** ⇒ 능력 아니라 **stochastic slip**(런마다 변동 1·1·0).
- **사용자 vLLM 아이디어 검증**: 스키마 깨끗하니 **guided/structured decoding이 hallucinate를 결정론 차단**=진짜 hard(deny/prompt=soft와 근본 다름). 단 우선순위 낮음(5%·stochastic). **gate(dd_fb) 폐기 권고**.

---

## 7. 노벨티 판정 (C145·딥리서치 106 에이전트·3-표 검증)

**"최초 발견" 아님.** 진단은 선점, 델타는 좁고 명확:
- **★결정적 근접선행 = "Plans Don't Persist"(arXiv 2606.22953·2026-06)**: ①plan 실패=표현적(문맥서 밀려나면 action당 ~4.1× 감쇠)=우리 전제 선점 ②**stale plan 재부각 처방도 실패**=우리 C144(compact plan 재부각 실패) 선점.
- **우리 델타(OPEN)**: 그들 **부정 결과(plan 재부각 실패)를 긍정으로 확장** — **source 정책문서 재부각은 회복**(C144 R_rawdoc)·**LLM-추출 plan의 결정론 추적**.
- 동기(clean-vs-polluted 해리)=선점(Laban 2505.06120·2509.09677)→인용/양보.
- **논문 필수인용**: 2606.22953·2505.06120·2509.09677. **구현 경고**: plan만 재부각하면 그들처럼 실패·정책문서 재부각 필수.

---

## 8. 정직한 caveat·미해결

- **단일 태스크(043)·단일 도메인·주로 32B·nt 작음**. 일반화는 미확립.
- **라이브 task pass = 여전히 0**(nt4 0/4). 단 **원인은 정정됨**(C159): close over-action이 아니라 **discovery 등록 부기**(sim0 ndiff=2·sim2 ndiff=1·상태는 gold 일치).
- ~~**scaffold 게이트는 soft**~~ ⚠️**반증(C159)**: pre-close 게이트는 발화·차단 **모두 성공**(close 0/4). soft로 남는 것은 **discovery-dispatch 게이트**(C152·에이전트가 재발행 포기)에 한한다.
- ~~**비-scaffold(reasoning) 부분회복(3/6)**~~ ⚠️**반증(C158)**: 순수(도메인-일반) reasoning은 억제 실패(6~7/12 CLOSE≈대조)·C156의 3/6은 **leading 프롬프트 산물**. 또 그 "CLOSE" 지표 자체가 태스크 실패를 예측하지 않음(C159).
- db_match=brittle 메트릭(9%)·per-step action이 더 나은 신호일 수 있음(벤치 신호 적합성 재고 필요). **[C160이 강화]**: 부기 레코드가 해시에 포함돼 **행동 정답도 탈락**시킴(4%).
- **★방법론 교훈(이 문서 자신)**: 도구 *호출*을 **인자 문자열 매칭**으로 세면 unlock/실행이 섞인다. 이벤트 귀속은 **실행 경로(dispatcher)로 분리**하고 **DB-diff로 독립 확증**해야 한다([[08]]).

## 9. 다음 (우선순위) ⚠️**(C159/C160 반영해 재작성)**

1. ~~close-before-retention 완전화~~ **폐기** — 그 현상이 실측상 존재하지 않음(close 0/4·게이트가 차단).
2. **행동 실패(본류)**: 97태스크의 87%가 실제 상태차 동반(C160) ⇒ 잘못된/누락 write가 여전한 주표적. 기존 행동 레버 작업으로 복귀.
3. **부기(형식) 레버**: vLLM guided decoding — 상한이 **REG_ONLY 4% + near-miss 7태스크**로 정량화됨(C160). C154의 "우선순위 낮음" 판단 유지가 타당.
4. **메트릭**: db_match brittleness(부기 포함)·per-step action-match 적합성 재고.
5. **소급 재감사**: C155~C158 파생 주장 정정(본 고지)·E-F3-ISO(별도 큐·무료).
