# 검색·형식 이론 메모 — x395 격리 프로브의 F1~F8 을 무엇이 설명하는가

작성 2026-08-19 · 대상 = `x395_compliance_iso.py` (t7326 표적 12~14자리) · 모델 Qwen2.5-32B-Instruct-GPTQ-Int8 · 로컬 vLLM
등급 [S] 이 문서에서 직접 계수 · [M] 표본 소수·간접·집계표 경유 · [D] 문헌·코드 주석 진술만
관련 메모리: [[62]] 결손 먼저 · [[18]] 격리 의무 · [[08]] 포렌식 선행 · [[55]] 우리 배관 먼저 · [[57]] 부정통제 · [[25]] 우리 도구 100% · [[46]] 노벨티 · [[23]] gold 금지 · [[59]] 패턴매칭 금지
자매 문서(중복 금지·먼저 읽어라): `FC_FORMAT_HYPOTHESIS_2026_08_19.md` (§1-3 유의성 게이트 · §7 [?]1~6 이 아래 X1·X6·X7·X11 을 이미 사전등록)

> ⛔ **이 문서의 결론은 "이론이 없다"가 아니라 "설명할 실측이 아직 확립되지 않았다"이다.**
> §2 의 기각은 확정이고, §3 의 생존은 **전부 §5 의 게이트 통과를 조건**으로 한다.
> §6-C 의 D0(원자료 재분류)를 통과하기 전에는 F2·F3·F4·F5·F6 중 **어느 것도 논문 주장으로도 이론 선별 근거로도 쓸 수 없다**.

---

## 1. 설명해야 할 것 (보고된 실측 · 수치 그대로)

질문은 전 팔 동일 — "지금 다음에 호출할 도구 하나를 JSON 으로 내라". 도구 후보 목록도 전 팔 동일 [S](코드 `prompts` 확인).

| 팔 | 전달 | exact | 말만(said_only) | 호출률=1−말만 | a = exact/호출률 |
|---|---|---|---|---|---|
| A_min | 절차문장+원장요약+손님요청+도구목록 | 0.58 | 0.00 | 1.00 | 0.583 |
| B_tail4 | 대화 마지막 4메시지 | 0.08 | 0.67 | 0.33 | 0.250 |
| B_tail8 | 마지막 8메시지 | 0.25 | 0.58 | 0.42 | 0.600 |
| B_tail16 | 마지막 16메시지 | 0.17 | 0.58 | 0.42 | 0.400 |
| B_tail32 | 마지막 32메시지 | 0.31 | 0.42 | 0.58 | 0.524 |
| B_full | (실제로는 **앞** 60메시지) | 0.39 | 0.17 | 0.83 | 0.467 |
| C_neg | 절차문장을 무내용 문구로 교체 | 0.19 | 0.00 | 1.00 | 0.194 |

n=36/팔 [M]. 파생 관측:

| | 진술 | 유의성 [M] |
|---|---|---|
| **F1** | 디코더 온도 무효. A_min 0.58(T=0) ↔ 0.58(T=0.7) · B_full 0.42 ↔ 0.38 | z=0.34, p=.73 · **검정력 없음**(95% CI [−0.19,+0.27]) |
| **F2** | 대화 팔 안에서 문맥이 길수록 좋다 (0.08→0.39) | z=3.33, p=.0009 · 호출률로 재면 z=4.99 |
| **F3** | 같은 정보를 구조화 요약으로 주면 0.58, 원문 대화로 주면 0.39 | **z=1.64, p≈0.10 — 미확립** |
| **F4** | "말만"은 대화 팔에서만(0.17~0.67), 구조화 팔은 정확히 0.00 | Fisher p=.0006 |
| **F5** | 부정통제(무내용 문구) 0.19 (A_min 0.58 대비) | z=3.71, p=.0002 |
| **F6** | 같은 모델이 라이브 멀티턴에서 이 자리들을 전부 놓쳤다 | **선정 기준 = 정보량 0**(§2-F) |
| **F7** | 별도 프로브: 정책 상수 인용 0.90~1.00 정확, 적용 성분 오류로 최종 금액 0/24 | 별도 런·미매칭 |
| **F8** | 과거 실측: 호출 강제 시 over-action 2→8, pass 효과 null | 별도 런·baseline 호출률 미보고 |

⚠ **부기 불일치 1건**: 본표 B_full exact = 0.39 인데 F1 의 B_full T=0 셀은 0.42 다. 같은 팔·같은 T=0 인데 값이 다르다 [M]. 다른 런이면 F1 은 런 간 비교로 오염된 것이다.
⚠ **산술 불일치 1건**: 7팔 × 36 = 252 ≠ 보고된 216. 216 = 6팔 × 36 과 정확히 맞는다 [S]. 코드 기본값은 `--mode next,plan` 이고 plan 모드는 `plan[0]` 을 pred 로 쓴다 [S] — 표에 plan 모드가 섞였을 위험이 남는다.

---

## 2. 기각된 설명

### 2-A. 모던 홉필드 간섭 서사 — **기각**
Ramsauer 2020 계열(어텐션 = 연상기억 업데이트, 유사 패턴의 metastable 평균)의 우리 자리 예측은 "문맥 질량↑ → 유사 패턴 간섭↑ → 검색↓" 였다 [D].
**기각 축자 수치**: 대화 팔 안에서 창이 4→8→16→32→60 메시지로 늘 때 exact 는 **0.08→0.25→0.17→0.31→0.39 로 오른다** [M]. 부호가 반대다.
lost-in-middle(2307.03172)도 이 자리에 안 붙는다 — 절차문장은 전 팔에서 **맨 뒤 고정**이다 [S](코드: `... + "\n\n# 정책 절차(축자)\n" + proc + q`).

### 2-B. 제약 디코딩 계열 전부 — **적용 불가(기각)**
GAD(2405.21047) · label bias(1603.06042) · Constraint Tax(2606.25605 / 2605.26128) · Let Me Speak Freely(2408.02442) 는 grammar mask·`tool_choice`·`guided_json` 을 전제한다 [D].
**기각 축자 근거** [S]: 프로브의 요청 body 는 전부다 —

```python
body = json.dumps({"model": MODEL, "messages": msgs, "temperature": temp, "max_tokens": maxtok})
```

`response_format` · `tools` · `tool_choice` · `guided_*` 가 **하나도 없다**. 그리고 네이티브 `<tool_call>` 채널 자체를 안 쓴다 — 모델은 평문 content 에 JSON 을 쓰고 우리는 정규식으로 판다 [S].
⇒ **선행 보고의 "F4 계기 감사(grammar mask)가 최우선" 경보는 여기서 종결**한다. 동시에 첫 토큰 로짓·마스크 도달성·two-pass 처방 전부 x395 에 무의미하다.

### 2-C. Altmann & Trafton 2002 (goal-activation) — **전체 계정으로는 기각**
축자 [D]: *"Priming is the only way to overcome the retroactive interference that affects an old goal."* (p.48) / Note 6: *"If no element is above threshold when the system samples, that attempt fails and the system can either try again or move on to another activity."*
- 기각 ①: 모형에 **출력 채널 변수가 없다**. 단조·유의한 축은 exact 가 아니라 호출률(0.33→1.00)이다 [M].
- 기각 ②: 검색문턱 τ 가 F5 에서 역행한다. C_neg 는 표적 활성이 최저(a=0.194)이므로 τ 미달 사건이 최다여야 하는데 실측 말만은 **0/36 = 최소** [M]. **τ 로 F8 을 사면 F5 에서 판다.**
- 기각 ③: A5 의 `−½ln t` 항이 요구하는 logit 기울기는 s<0.5 에서 >1.0 인데 실측 0.64(s≈0.78 = 모형 수렴조건 밖) [M].
- 회수: P5 의 되먹임 비용 명제(*"strengthening beyond a certain point actually degrades memory"*)는 §5-P4 의 부정통제 논거로만 남긴다.

### 2-D. McDaniel & Einstein 다중과정 / 단서 초점성 — **기각**
축자 [D]: *"successful nonfocal PM performance depends upon the engagement of attention-demanding processes (e.g., monitoring) directed toward identifying the PM cue."*
기각: 진행 과제 부하(대화 길이)가 커질수록 성적이 **오른다**(0.08→0.39) [M]. 이 이론의 유일한 강한 셀인 F3 는 p≈0.10 로 미확립이다 [M]. 문헌 효과는 .73↔.18(4.1배), 우리 것은 1.5배이면서 ns [D/M].

### 2-E. Xie 2111.02080 (ICL = 암묵 베이즈 추론) — **전제 미충족 기각**
Eq.6 의 `exp(n·r_n(θ))` 에서 n 은 **예시 개수**이고 정리는 동일 concept 에서 뽑힌 예시 나열을 전제한다 [D]. 우리 창에는 (x,y) 예시 쌍이 **0개**이고 단일 비-i.i.d. 에피소드다 [S].
게다가 ε_delim(*"The prompt formatting (e.g., choice of delimiter) can also be a source of mismatch."*)은 사전학습 분포에 자연스러운 대화 전사를 **덜** 벌하므로, 순진하게 읽으면 B_full 이 이겨야 한다 [D].

### 2-F. F6("라이브 전멸")을 설명한다는 모든 학점 — **무효**
케이스 선정 조건 축자 [S]: `if not nm or cn.get(nm): continue` — **그 도구를 한 번도 안 부른 것만** 남긴다.
P(라이브 실패 | 표적) = 1 은 관측이 아니라 **정의**다. F6 을 "강하게 설명한다"던 Altmann·McDaniel·Duncan 의 학점은 선정 기준을 설명한 것이다.

### 2-G. 어트랙터 주기(2502.15208) · 알고리즘 상(2412.01003) — **기각**
전자: 어트랙터는 `T_{n+1}=P(T_n)` 반복 사상의 궤도인데 우리 프로브는 1회 순방향 통과다 [S/D].
후자: 초록의 `merely varying context size` 에도 불구하고 본문 실험은 `leval=400` 고정이고 변주 축은 training steps 와 data diversity 뿐이다 [D] — **문맥축 실증이 0 건**이다. 이 논문을 "문맥 길이 상전이 실증"으로 인용하면 안 된다.

### 2-H. 합성 수식 `exact = σ(η(n,φ))·a(κ)` — **수식 폐기, 요인 분해만 유지**
- `n·Δℓ`("문맥 길이 = 역온도")가 성립하려면 logit(호출률)이 n 에 선형이어야 하는데, n-선형과 ln n-선형이 잔차로 구분되지 않는다 [M].
- `a` 의 팔 간 동질(χ² p≈0.37)은 **동질의 증거가 아니라 검정력 부족**이다(실측 범위 .250~.600, 셀 n=12~36) [M].
- 분모(호출률)가 §6-X1 로 오염돼 있다.
- 남기는 것은 어휘 하나뿐: **"팔은 무엇을 고르는가(a)가 아니라 발행하는가(호출률)를 바꿨다."**

---

## 3. 살아남은 이론

### 3-①. [언어학·대화분석] 담화 의무 + 조건적 적절성 = **화용적 채널-선택**

**출처** David R. Traum & James F. Allen, "Discourse Obligations in Dialogue Processing", *Proc. ACL 1994*, pp.1–8 (ACL Anthology P94-1001) · Emanuel A. Schegloff, "Sequencing in Conversational Openings", *American Anthropologist* 70(6):1075–1095, 1968 (p.1083).

**축자 인용** [D]
> "when an agent is asked a question, this creates an obligation to respond. The agent does not have to adopt the goal of answering the question as one of her personal goals in order to explain the behavior." (Traum & Allen 1994)
> "In cases where the agent does not know the answer, the obligation to respond may be discharged by some explicit statement of her inability to give the answer." (같은 곳)
> "Given the first, the second is expectable; upon its occurrence it can be seen to be a second item to the first; upon its nonoccurrence it can be seen to be officially absent — all this provided by the occurrence of the first item." (Schegloff 1968:1083)

**형식 진술**: 창(context window)이 담화 상태 σ 를 결정하고, σ 가 **다음에 due 한 행위의 종류**를 결정한다. σ = "미결 요청 있음" ⇒ due = 조치(도구 호출). σ = "종결·인계 완료" ⇒ due = 없음, 그리고 그때 발행되는 것은 산문 또는 (시스템 프롬프트가 명시적으로 허가한) 기권 JSON 이다.

**설명 범위** — 이 이론이 살아남은 진짜 이유는 t7326 원자료 직접 계수다 [S]:

| 창 | 창 안에서 "마지막 도구 호출 이후" 메시지 비율(평균, n=23) | 보고된 말만 |
|---|---|---|
| tail4 | **0.891** | 0.67 |
| tail8 | 0.652 | 0.58 |
| tail16 | 0.413 | 0.58 |
| tail32 | 0.243 | 0.42 |
| head60 (= B_full) | **0.141** | 0.17 |

- Spearman(종결부 비율, 말만) ≈ **+0.97**, Spearman(종결부 비율, exact) ≈ **−0.90** [S].
- **선정 대상 23/23 의 마지막 user 메시지가 `###STOP###` / `###TRANSFER###` / `###OUT-OF-SCOPE###` 종결 센티널이다** [S].
- ⇒ F2 사다리·F4 사다리·구조화 팔의 정확히 0.00 이 **하나의 담화 변수로** 나온다. 구조화 팔에는 대화가 0줄이므로 종결 신호가 **재료로 존재하지 않는다**. 즉 0.00 은 "지지집합 절단" 같은 형식 사건이 아니다.

**한계** ⛔ **결정적**: 창 길이와 종결부 비율의 Spearman = **−1.000** [S]. 이 설계에서 "문맥이 길수록 좋다"와 "종결 오염이 적을수록 좋다"는 **관측상 동일**하다. 이 이론은 현재 데이터로 **지지도 반증도 되지 않는다**. §5-P1/P2 가 오직 이 공선만 판다.
부수 한계: 인간 상호작용의 기술(記述)이라 정량 예측이 없다. 계산 이식본(Traum & Allen)도 forward-chaining 규칙 수준이다 [D].

---

### 3-②. [인지심리·메타기억] 보고 기준 (monitoring–control)

**출처** Asher Koriat & Morris Goldsmith, "Monitoring and Control Processes in the Strategic Regulation of Memory Accuracy", *Psychological Review* 103(3):490–517, 1996 (PMID 8759045).

**축자 인용** [D]
> "The control mechanism then compares that assessed probability with a preset response criterion probability, Prc: The answer is volunteered when Pa > Prc, but withheld otherwise." (p.494)
> "people can boost the accuracy of their memory reports only by screening out answers that they feel are likely to be incorrect, not by enhancing the overall correctness of their answers ... only about 1% of the answers differed between the two phases." (p.507)
> "Indeed, because forced-report situations require that Prc = 0, both of these contrasts may be viewed in terms of criterion level." (p.495)

**형식 진술**: 산출 = (Pa, Prc) 문턱 비교. 강제 보고 ⇔ Prc = 0 ⇒ 수량(input-bound)↑ · 정확도(output-bound)↓ · **새 정답은 생기지 않는다**. 세 인자 = monitoring effectiveness / control sensitivity / criterion setting.

**설명 범위**
- F8 에 대한 문헌 전체 중 가장 정밀한 적합 — 강제 시 호출 수량↑ · over-action 2→8 · pass null 3항이 부호·구조 모두 일치 [M/D].
- ⭐ **우리 계기가 이 이론을 문자 그대로 구현하고 있었다** [S]. 시스템 프롬프트 축자:
  > `"If you believe no tool call is needed, reply {\"tool\": null, \"reason\": \"…\"}."`

  `parse_tool` 은 그 JSON 을 **정상 파싱해 `nm=None` 을 돌려주고**, 채점은 `said_only = (not nm) and not plan` 이다 [S]. ⇒ **지시대로 낸 기권이 "말만"으로 채점된다.** F4 는 "산문으로 도망쳤다"가 아니라 **"기권했다"**일 수 있다. 그러면 K&G 가 F5(C_neg 기권 0.00)에서 부딪혔다던 모순도 사라진다 — C_neg 에는 종결 신호가 없어 기권할 근거가 없기 때문이다.

**한계**: payoff 문구가 전 팔 동일하므로 Prc 를 움직인 자유변수가 이론 안에 없다 [S]. Pa 를 한 번도 안 읽었다. 팔 간 비교는 K&G 의 정당한 검정이 아니다 — 정당한 검정은 **동일 항목 forced/free 2단 절차**이고 우리는 그것을 한 적이 없다.

---

### 3-③. [인지심리·안전공학] 규칙 적용 결손 (goal neglect / rule-based mistake / inert knowledge)

**출처** John Duncan, Hazel Emslie, Phyllis Williams, Roger Johnson & Charles Freer, "Intelligence and the frontal lobe: The organization of goal-directed behavior", *Cognitive Psychology* 30(3):257–303, 1996 · Duncan et al., *JEP: General* 137(1):131–148, 2008 · James Reason, *Human Error*, CUP 1990 (GEMS, 3장) · Renkl, Mandl & Gruber, "Inert knowledge: Analyses and remedies", *Educational Psychologist* 31(2):115–121, 1996.

**축자 인용** [D] ⚠ 2차 출처 경유 · **원문 대조 미완**
> goal neglect = "disregard of a task requirement even though it has been understood and remembered"; "the neglected requirement slips the subject's mind." (Duncan et al. 1996)
> "the mistake arises from the application of a 'bad' rule or the misapplication of a 'good' rule [a rule of proven worth]." (Reason 1990, GEMS)
> knowledge, although seemingly available, is often not used for problem solving — that is, it remains "inert". (Renkl et al. 1996)

**형식 진술**: 인출(상수·규칙의 축자 재현)과 적용(이 사례에 어느 규칙이 걸리는가)은 **분리 가능한 단계**이고 후자만 독립적으로 붕괴할 수 있다. GEMS 는 이것을 실행 단계의 슬립과 구별한다 — 슬립은 표집 잡음에 흔들리지만 **규칙기반 실수는 하나의 틀린 답에 확신 있게 수렴**한다.

**설명 범위**: **F7 을 담당하는 유일한 후보이고, F7 은 §6 교락 목록에서 살아남은 유일한 관측이다** — 인용 0.90~1.00 ↔ 적용 0/24 는 x395 의 창 설계·`said_only` 채점·선정 tautology 어느 것과도 무관한 별도 프로브다 [M]. 그리고 GEMS 는 F1(온도 무효)의 유일한 *기전적* 독법을 준다: 결손이 규칙 선택에 있으면 실행 잡음(디코더 온도)은 원리상 아무것도 못 고친다.

**한계**: 분류 체계이지 생성 모형이 아니다 — 수치를 못 낸다 [D]. 우리 F1 은 애초에 검정력이 없어 이 독법을 지지하지 못한다(§1) [M]. 정독 미완([?]-9).

---

## 4. 최선의 통합 설명 — **억지로 하나로 못 묶는다. 두 기전의 중첩이다.**

### 기전 I — 담화 상태가 **발행 여부**를 정한다 (관할: F2 · F4 · F6 사다리 전부)
창이 종결된 대화를 담을수록 "지금 due 한 조치"가 담화적으로 없어지고, 모델은 지시된 기권 JSON 또는 산문을 낸다. 이 기전 하나가 호출률 0.33→1.00 전 구간과 말만 0.67→0.00 전 구간을 만든다 [S 상관 · M 인과].
**관할 밖**: 어느 도구를 고르는가(a). a 는 팔 간에 단조가 아니다(0.250 / 0.600 / 0.400 / 0.524 / 0.467) [M].

### 기전 II — 재료가 **무엇을 고르는가**를 정한다 (관할: F3 · F5 의 크기)
A_min·C_neg 와 B_* 는 형식만 다른 게 아니다 [S]:
- `base = tools + 손님 요청 + 원장` 은 **A_min·C_neg 에만** 붙고 B_full·B_tail* 에는 안 붙는다(코드 축자 확인).
- 선정 대상 23건 중 **첫 user 메시지(손님 요청)가 tail4/8/16 창 안에 들어간 것은 0건**, tail32 에서도 3/23 뿐이다.
- 절차 줄은 `if tool in s or re.match(r"(?i)\s*(before|first)\b", s)` 로 뽑으므로 **표적 도구명이 프롬프트에 축자로 들어 있다** — A_min·B_full 에는 있고 C_neg 에는 없다. ⇒ **F5(0.58↔0.19)는 "재료 유무"가 아니라 "정답 이름 유무"다.**

### 두 기전은 이 설계에서 **직교하지 않는다**
형식 축(구조화 ↔ 대화)과 재료 축(손님요청·원장 유무)이 **완전 공선**이고 [S], 창 길이 축과 종결부 비율도 **완전 공선(ρ = −1.000)**이다 [S].
따라서 위 서사는 **검정된 분해가 아니라 관할 배정 제안**이다. §5 의 P1~P4 가 정확히 이 두 공선을 깨러 간다.

### 세 번째 기전 — 관할 밖에 남는 것
F7(적용 결손)은 두 기전 어느 쪽도 안 낸다. 격리에서 a ≈ 0.50 인데 적용 프로브에서 0/24 다 [M].
⇒ **진짜 남은 질문은 채널이 아니라 "a 가 어디서 0 으로 무너지는가"일 수 있고, 채널 축(F2·F3·F4)은 그 위에 얹힌 큰 부수 효과일 수 있다.**

---

## 5. 새 예측 5개 (아직 안 잰 것 · 전부 로컬 vLLM·기존 궤적 재조립 · 유료 0)

> ⛔ 게이트 D0(§6-C)을 통과하기 전에는 P1~P5 의 종점 해석이 전부 오염된다. **D0 먼저.**

### P1 — 결정점 절단 `B_prefix` ★ 기전 I 의 존폐
표적이 호출됐어야 할 시점 p 까지의 접두사 `msgs[:p]` 만 보여준다(종결 대화 제거·길이는 그 시점이 정하는 대로).
**사전 고정 종점**: 말만 **≤ 0.15** 이고 exact **≥ 0.50** → **F2·F3·F4 는 전부 프로브 아티팩트**이고 [[62]] ② 의 "격리에서는 된다"가 확정된다. 말만 **≥ 0.40** 이거나 exact **≤ 0.30** 이면 → 종결 오염이 아니라 실재 결손이고 기전 I 은 부수 효과로 강등.
**무료 실행**: `convo(sim, tail=None)` 대신 gold `action_check` 의 순서 위치로 슬라이스. 기존 `.results.json.gz` 재조립 · 신규 런 0.

### P2 — 위치·길이 분리 `B_mid32` / `B_pre4` ★ 공선 파괴
`B_mid32 = msgs[n-64 : n-32]`(32메시지·종결 이전) · `B_pre4 = 마지막 도구 호출 직전 4메시지`.
**사전 고정 종점**: 말만(B_mid32) − 말만(B_full) **≤ 0.10** → F2 는 길이가 아니라 **종결 오염**(⇒ §2-A 의 홉필드 기각은 유지되나, 그 자리를 메우던 Xie·Altmann 독법도 함께 폐기된다). 말만(B_mid32) ≈ 0.42(= B_tail32 수준)면 → 길이 축이 실재하고 F2 가 진짜 현상.
**부수 종점**: `B_pre4` 의 exact 가 `B_tail4`(0.08)보다 **≥ 0.25** 높으면 "짧은 창"이 문제가 아니었음이 한 셀로 확정된다.

### P3 — 재료 균등화 `B_full_plus` / `A_min_noledger` ★ F3 의 운명
`B_full_plus` = B_full + `# 손님 요청` + `# 원장`(A_min 과 **바이트 동일** 블록 이식). `A_min_noledger` = A_min − 원장 − 손님요청.
**사전 고정 종점**: exact(B_full_plus) − exact(B_full) **≥ 0.12** → F3 의 상당분은 형식이 아니라 **재료**. exact(A_min_noledger) ≈ exact(B_full) → A_min 우위는 전부 원장이 만든 것 ⇒ **F3 폐기**.
⚠ **P1 통과 후 접두사 위에서** 돌려야 의미가 있다 — 종결된 대화 위에서 원장을 만들면 원장이 "인계 완료"를 충실히 인코딩해 같이 무너진다.

### P4 — 정답명 마스킹 `A_min_masked` ★ [[62]] ② 전제의 검증
`proc` 블록에서 표적 도구명만 `<TOOL>` 로 치환(문장 구조·길이 유지). 부정통제로 **무관 도구명을 마스킹한 팔** 병행([[57]]).
**사전 고정 종점**: exact(A_min_masked) **≤ 0.25**(= C_neg 수준) → **A_min 의 0.58 은 준수가 아니라 복사**이고, "격리에서는 된다"가 무너져 레버 논의 전체가 [[62]] ① 로 되돌아간다. **≥ 0.45** 유지 → 격리 능력 실재.

### P5 — 기권 채널 조작 ★ F4 의 정체 (K&G vs 화용)
시스템 프롬프트 **한 문장만** 3조건: (a) 현행(`{"tool": null, "reason": …}` 허가) (b) 그 문장 **삭제** (c) 같은 길이 무내용 문장으로 치환(부정통제).
**사전 고정 종점**: 말만(b) ≤ 말만(a) − **0.30** 이면서 exact(b) − exact(a) **≤ 0.05** → **"말만"은 지시된 기권**이고 F4 는 현상이 아니라 채점 범주다(K&G 축이 우리 데이터에 직접 붙는다). 말만이 안 움직이면 → 기권은 지시가 아니라 담화가 연 것이고 기전 I 이 강화된다. exact 가 **≥ 0.15** 오르면 → 강제와 같은 방향의 F8 재현이므로 상쇄(over-action)를 반드시 함께 보고할 것([[57]] · 모트).

---

## 6. 교락 배제 목록

### 6-A. ✅ 배제됨 (더 안 봐도 된다)

| 후보 | 근거 |
|---|---|
| grammar mask · `guided_json` · `tool_choice` · JSON 강제 | 요청 body 에 **없다** [S]. 네이티브 `<tool_call>` 채널도 안 쓴다 |
| 도구 목록 위치·거리 | 전 팔 맨 앞 동일 프리픽스 [S]. 거리 최단인 tail4 가 최악 — 부호 반대 |
| 시스템 프롬프트 팔 간 차이 | `SYS` 는 mode 별로만 갈리고 arm 간 동일 [S] |
| 후보 목록 차이 | `tools` 문자열 전 팔 동일 [S] |
| 절차문장 위치(lost-in-middle) | 전 팔 맨 뒤 고정 [S] |
| 문맥 초과(입력 절단) | 최장 팔이 ~4~5k 토큰, 32k 한계 안 [M]. 게다가 최장 팔이 말만 최저 — 부호 반대 |

### 6-B. ⛔ 남아 있는 교락 (해소 전 인용 금지)

| # | 교락 | 위험 |
|---|---|---|
| **X1** | `said_only` 가 ⑴지시된 기권 `{"tool":null}` ⑵`"ERROR " + str(e)` ⑶`max_tokens=400` 절단 을 **전부 같은 칸에** 넣는다 [S] | ★★★ F4 무효화 가능 |
| **X4** | 선정 대상 **23/23 이 종결 센티널로 끝난다**; 창의 종결부 비율이 말만과 ρ≈+0.97, 창 길이와 ρ=−1.000 [S] | ★★★ F2·F4 무효화 가능 |
| **X5** | `B_full` = `convo(sim)` = `msgs[:60]` = **앞** 60메시지. 선정 23건 중 **13건이 nmsg>60** 이라 끝을 아예 못 본다 [S]. 사다리가 중첩이 아니다 | ★★★ "길수록"이라는 서술 자체가 틀림 |
| **X6** | `base`(손님요청+원장)가 A_min·C_neg 에만 붙는다; 첫 user 메시지가 tail4/8/16 창에 **0/23** [S] | ★★★ F3 은 형식·재료 분리 불가 |
| **X7** | 절차 줄 선택 규칙이 **표적 도구명을 포함하는 줄**이라 정답 이름이 프롬프트 안에 있다 [S] | ★★ F5 해석 · A_min 절대수준 |
| **X9** | 케이스 선정이 "호출 0회"를 요구 ⇒ F6 은 정의 [S] | ★★★ F6 정보량 0 |
| **X10** | 표적 23건이 **13개 대화**에서 나온다(단일 대화가 최대 3표적) [S]. 모든 z 가 독립 n 가정 | ★★ 전 유의성 재계산 |
| **X11** | `temp = 0.0 if k==0 else a.temp` ⇒ T=0 셀의 n 이 나머지의 1/(n−1) [S] | ★★ F1 검정력 0 |
| **X12** | 216 ≠ 7팔×36; `--mode next,plan` 기본이고 plan 은 `plan[0]` 을 pred 로 쓴다; 코드는 `B_tail64` 도 만든다 [S] | ★★ 표 provenance 미확정 |
| **X13** | `PORTS=[8140,8141]`, `PORTS[idx % 2]` — 두 서버 기동 플래그 미대조; 작업 큐 순서상 팔↔포트 균형 무보장 [S] | ★ |
| **X14** | `tool_universe` 가 정규식 스크레이프 + 하드코딩 이름 목록. 표적 전원 포함 여부 미검증 [S]. **도메인 문서가 원격 경로라 로컬 감사 불가** [S] | ★★ [[25]] |
| **X15** | C_neg 는 섹션 라벨도 바뀐다(`# 정책 절차(축자)` → `# 안내`) [S] | ★ |

### 6-C. 게이트 D0 (호출 0 · 최우선 · [[55]] 0단계)
원격 `reports/facet_rft_2026/x395_compliance_iso.json` 은 **로컬에 없다** [S]. 회수해서 `raw`(400자) 전량을 결정론 재분류: `EMIT` / `TOOL_NULL` / `PROSE` / `TRUNCATED` / `ERROR`.
**사전 고정 종점**: B_tail4 의 말만 0.67 중 `TOOL_NULL + ERROR + TRUNCATED` 합이 **≥ 0.30 이면 F4 를 논문에서 제거**. `PROSE` 가 **≥ 0.55** 로 남을 때만 F4 를 현상으로 쓸 수 있다.
동반 감사: 팔별 항목 수·모드 인쇄(X12) · 포트별 기동 커맨드 축자 대조(X13) · `tool_universe` 인쇄 + 표적 포함 검증(X14) · gold 호출 문자열 주입 왕복으로 파서 위음성 0 확인(X1) · F1 부기 불일치 확정 · {exact, 오호출, 말만} 배타·전수 검산(§1 의 a 분해가 여기 걸려 있다).

---

## 7. 설계 함의 — 결정점 커널의 입력 형식

이 이론(기전 I + II)이 맞다면 결정점 커널의 입력 규격은 다음이어야 한다.
⚠ **P1·P2 통과 전에는 구현하지 마라** — 지금 구현하면 아티팩트에 맞춰 짓는 것이고 [[62]] ① 위반이다.

1. **접두사만 준다.** 창은 **결정 시점까지의 접두사**로 자른다. 종결·인계·작별 이후의 턴은 결정점 입력에 **절대 포함하지 않는다**. (기전 I · P1)
2. **재료는 형식과 무관하게 항상 동반한다.** 손님의 원 요청 + 원장(호출 이력·관측 엔티티)은 어느 형식으로 렌더하든 **빠지지 않는다**. 지금처럼 형식에 따라 재료가 딸려 다니면 형식 축이 원리상 식별 불가다(기전 II · X6). 이것은 [[65]](메인엔 답만)와 충돌하지 않는다 — 원장은 과정이 아니라 **답들의 집합**이다.
3. **기권을 1급 출력으로 두되 채점에서 분리한다.** `{"tool": null, "reason": …}` 은 유지하되 로그·채점에서 `TOOL_NULL` / `PROSE` / `ERROR` / `TRUNCATED` 를 **절대 합치지 않는다**(X1). 기권 자체는 [[64]](거부는 무엇을 하면 풀리는지 함께 담아라)의 대상이지 실패가 아니다.
4. **담화 채널을 열지 마라 — 단 조건부.** 결정점 입력을 `user` 턴으로 렌더하면 응답 의무가 생긴다(Traum & Allen). 상태 블록 렌더가 기본값이되, **P5 가 "말만 = 지시된 기권"으로 나오면 이 조항은 근거를 잃는다.**
5. **절차문장이 정답 도구명을 담을 수 있음을 명기한다.** 담는 경우 그 과제의 실체는 준수 판단이 아니라 **주의분산 하의 선택·복사**다(X7). 논문·설계서에서 이 둘을 같은 이름으로 부르지 마라([[03b]]).
6. **출력 예산을 결정점에 맞춘다.** 현행 `max_tokens=400` 은 절단을 기권으로 만든다(X1). 그리고 에러 문자열을 출력 범주에 섞지 마라 — [[25]] 위반이다.
7. **정보-맞춤 감사를 상설한다.** 팔을 추가할 때마다 "이 팔에 없는 재료가 무엇인가"를 표로 남긴다([[18]]). x395 는 이 표가 없어서 F3 를 못 쓰게 됐다.

---

## 8. 선행 대비 우리 위치 ([[46]] 규율)

**이미 선점된 것 — 인용·양보** [D]
- false-success 현상·수치: `2606.09863`(τ² airline 45 / retail 48 / telecom 3 · judge AUROC ≤ 0.65).
- 구조로 막되 통과율 0% 를 지불: `2606.11688`.
- **제약이 도구-선택은 사고 기권 판단은 판다**: `2608.13959`(Qwen3 EN 기권 Δ_total −29.5pp [−37.5, −21.5] · 선택 +5.9~+62.7pp). ⇒ **우리 F8 은 현상으로서 새롭지 않다.**
- 멀티턴 열화 + CONCAT > SHARDED: `2505.06120` — "an average drop of 39% across six generation tasks"; "when LLMs take a wrong turn in a conversation, they get lost and do not recover." ⇒ **우리 F3 의 조작과 동형이다.** F3 를 "구조화가 낫다"로 쓰면 그대로 선점된다.
- 의미보존 형식 변형만으로 큰 진동: `2310.11324`(중앙값 7.5pt · 최대 76pt; "Sensitivity remains even when increasing model size ... or performing instruction tuning"). ⇒ **우리 F3(+19pp, p≈0.10)는 이 잡음 분포와 미분리.**
- register 가 최빈 질량을 옮긴다: `2607.18476` — "the register moves the mode, not the temperature"(plain→JSON 최빈 41%→64%).
- 열거·coverage 불완전: `2404.09593`([[49]]) · 강제·게이트의 대가: `2607.07405`([[61]]) · Skitka 1999(omission/commission) · Strom 2010(hard-stop RCT 조기 중단).

**우리 델타 후보 — 아직 확립 전** [M]
- ⭐ **"결정점 창의 담화적 종결 상태가 도구 발행을 끈다"** — 문헌은 멀티턴 열화를 *길이·누적*으로 설명해 왔다(2505.06120). 우리 [S] 계수는 그 자리에 **종결 오염**이라는 경쟁 변수를 놓고, 두 축이 τ² 궤적에서 **완전 공선(ρ = −1.000)**임을 보인다. 이 공선 자체를 지적한 선행을 아직 못 찾았다. P2 가 이것을 가른다.
- ⭐ **기권 채널의 계측 오염** — 지시된 기권을 "말만"으로 채점하면 knowing–doing gap 이 **부풀려진다**. 선행이 이 분리를 했는지 미확인([?]-6).
- ⛔ **"배치가 knowing-doing 을 닫는다" 헤드라인은 여전히 금지**([[46]]). 우리 F4·F6 은 지금 상태로 그 주장의 근거가 못 된다.

**폐기해야 할 우리 쪽 주장**
- "F3 = 계산적 비등가(Blackwell 위반)" — Blackwell 위반은 원장 요약이 대화의 garbling 일 때만 성립하는데, X6 로 **A_min 이 garbling 이 아님(정보 추가)**이 확정됐다 [S]. 위반 주장 자체가 성립하지 않는다.
- "지지집합 절단이라 exact-zero" — 0.00 은 지지집합이 아니라 그 팔에 **종결 재료가 0줄**이라서다 [S].
- "문맥 길이 = 잠재상 사후분포의 역온도" — n-선형과 ln n-선형이 구분되지 않는다 [M]. 수사로 쓰지 마라.

---

## 9. [?] 목록

1. **[?]-1** 원격 `x395_compliance_iso.json` 회수. 없으면 §1 표의 어떤 셀도 재검증 불가 [S: 로컬 부재 확인].
2. **[?]-2** 216 vs 252 산술 확정 — 표에 `mode=plan` 이 섞였는가. 팔이 6인가 7인가(코드는 `B_tail64` 도 만든다).
3. **[?]-3** F1 의 B_full 이 0.42 인가 0.39 인가. 다른 런이면 F1 폐기.
4. **[?]-4** 도메인 문서(`/home/woori/scratch/.../banking_knowledge/documents`)가 로컬에 없어 `proc_lines`·`tool_universe` 를 감사하지 못했다 [S]. 표적 14/14 가 후보 목록 안에 있는가.
5. **[?]-5** 실제 선정 14건의 task id 목록 — 내 재현은 doc 필터 없이 **23건 / 13 대화**다 [S]. 최종 14건의 클러스터 수를 확정해야 유의성 재계산이 된다.
6. **[?]-6** `2606.09863` / `2606.11688` 이 "명시적 기권"과 "산문 대체"를 분리해 채점했는가. 안 했으면 우리 X1 지적이 델타가 된다 — 정독 필요.
7. **[?]-7** F7 프로브(24문항)의 재료·창 설계가 x395 와 같은 교락을 갖는가. F7 은 현재 유일한 무오염 관측이라 이 확인이 가장 값지다.
8. **[?]-8** F8 실험의 baseline 호출률. 없으면 "오호출 배수 = 1/호출률" 검정을 못 한다.
9. **[?]-9** Duncan 1996 / Reason 1990 원문 축자 확보(현재 2차 출처 경유) — §3-③ 을 논문에 쓰기 전 필수.
10. **[?]-10** 8140/8141 두 서버의 기동 커맨드 축자 대조(X13).
11. **[?]-11** 라이브 12자리의 실패 표현형 분류(산문 / 오호출 / 침묵). 전부 산문이면 F6 은 F4 로 접히고, 오호출이면 기전 I 밖이다 — 기존 궤적 재분류로 무료.
