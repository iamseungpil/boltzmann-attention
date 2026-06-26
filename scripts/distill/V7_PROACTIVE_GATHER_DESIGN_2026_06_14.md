# 설계서 v7: proactive 2-hop gather (R2 gather + R4 select) — 2026-06-14 (리뷰용 DRAFT)

> 상위 = `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md` · 진단 = `AUTONOMOUS_PROGRESS_2026_06_14.md`(v4 autopsy·v6 중간eval) · 불변 = memory `feedback-thesis-tbox-transfer-direction`(★★★ TBox는 SOPBench/TaskBench서만 학습·ABox-swap 전이·타깃벤치 특화 금지) · `feedback-selector-verifier-deterministic`(★★★).
> **상태: DRAFT — 사용자 리뷰 후 구현.**

## 0. 동기 (autopsy → v6 → 남은 한 칸)
- **v4 τ² 전수 autopsy**: pass 2/20(둘다 read-only)·write필요 태스크 전멸. 지배 실패 = **tool-fetchable 값(order_id `#W0000000`·주소) 날조** + 재시도 루프 붕괴.
- **v6(fetch-teaching: TaskBench `<node-N>` threading + fetchable-randomize)**: ✅**identity 값 날조 잡힘**(auth grounded 18/19, v4/v5는 fab ~半). ❌**order_id 여전히 날조**(task5 dump: user_id 획득 후에도 `get_user_details`로 주문목록 fetch 안 하고 `#W0000000` 날조→error→포기).
- **남은 한 칸 = "2-hop proactive gather"**: *없는* arg 값을, *이미 가진* 다른 값(user_id)을 입력으로 받는 **lookup 도구(get_user_details)를 능동 선택·호출해 그 출력서 획득/select.* 

## 1. 스킬 분해 (무엇을 가르치나)
τ² order_id 사례의 일반형:
```
가진 것: user_id (find_user로 grounded 획득)
필요한 것: order_id (대화에 없음·user가 withhold)
가용 도구: get_user_details(user_id) -> 주문목록(order_id들 포함)
정답 행동: get_user_details 호출(R2 gather) → 출력서 올바른 order 선택(R4 select) → 사용
실패 행동(현 7B): order_id placeholder 날조
```
= **R2(gather-before-act) + R4(select-from-output)**, 단 **inferred**(user/instruction이 "먼저 조회하라"고 안 말함). 벤치-일반 규율.

## 2. 왜 현 학습데이터에 없나 (autopsy의 더 깊은 층)
| 벤치 | gather 추론? | 출력값→arg? | 결론 |
|---|---|---|---|
| SOPBench (ast_*fc) | ✅ 정책서 getter 추론 | ❌ getter=*decision* 입력(균형·점수)·arg 아님 | inferred-ask는 있으나 fetch-**arg** 희소(1.9%) |
| TaskBench (threaded) | ❌ 그래프/지시문 given | ✅ 출력→입력(41%) | 구조는 있으나 *선택*은 복제(inferred 아님) |
| **교집합(inferred + 출력값→arg)** | — | — | **두 벤치 어디에도 풍부하게 없음 = v7이 채울 칸** |

## 3. ★lever = SOPBench LLM user-sim + withholding 페르소나 (벤치 task 불변)
**핵심: 벤치 task를 *수정/합성*하지 않는다. SOPBench 기존 user-sim 인프라를 쓰되 user가 *공개하는 정보*만 바꾼다.**
- **확인된 사실**(`run_simulation.py`): user-sim 3모드 = `usr_adv`(dump)·**`usr_gpt`(LLM user-sim = τ²와 동일 패러다임)**·`usr_human`. user가 공개하는 정보 = **`task["user_known"]`**(per-task 필드·L159/L308). 비-LLM은 "Here is all the information I can provide: {user_known}"; LLM user-sim은 user_known을 갖고 user 연기.
- **withholding 구성**: `user_known`을 **{goal + leaf identity(username/name/zip 등)}만** 남기고 **getter-생산 가능 값(getter_map 출력 슬롯)·lookup-획득 값은 제거** → user가 그 값을 안 줌 → 모델이 **lookup 도구를 능동 호출해 획득**해야 함.
- **thesis 순수성**: withholding 정책 = *user 측 = ABox/A2*. 벤치 task(도구·정책·DB)는 불변. SOPBench서 학습 → τ²(역시 user-sim·withhold)로 전이. = `feedback-thesis-tbox-transfer-direction` 부합(타깃벤치 특화 0).

## 4. ★정직한 뉘앙스 (리뷰 포인트) — withholding이 만드는 두 종류
1. **ask-gather** (user-only 값·getter 없음): user가 withhold → 모델이 **묻는다**. = D5/L3 ask 분기의 자연 버전. SOPBench 대부분 write-arg(amount 등)가 여기 해당.
2. **fetch-gather** (getter-생산 값): user가 withhold → 모델이 **getter 호출**(= τ² order_id 핵심). SOPBench서 **희소**(getter 출력=arg인 1.9% 케이스 = pay_loan(amount=owed_balance) 류).
- ⇒ **withholding 단독으론 fetch-gather가 여전히 희소.** 처방:
  - (a) **타깃 오버샘플**: getter-출력이 write-arg인 태스크(getter_map로 결정론 식별)를 우선 withhold·오버샘플.
  - (b) **TaskBench threading(v6) 유지**: 구조적 fetch-then-use(41%) 보강 — inferred는 아니나 "출력값을 arg로"의 양을 받침.
  - (c) **2-hop 합성 식별**: getter의 *입력*이 다른 getter의 *출력*인 체인(예: user_id→get_user_details→order_id→get_order_details)이 SOPBench에 있나 census 후, 있으면 그 체인 withhold로 정조준. (없으면 한계 정직 보고.)

## 5. 파이프라인 (기존 자산 재사용)
1. **withholding 태스크 변형 생성**: 각 SOPBench 태스크의 `user_known`을 identity-only로 재구성(getter_map으로 getter-생산 슬롯 제거). 결정론 스크립트(휴리스틱 아님).
2. **teacher rollout 생성**: `run_simulation.py --user_model <LLM>`(user-sim) + assistant=**강한 teacher**(gpt-4.1 FC). 학생7B는 아직 2-hop 불가 → teacher가 *성공 gather* 궤적 생성(SOPBench usr_gpt가 원래 teacher 생성). mode_fc·tool_full.
3. **success 필터**: `action_should_succeed` 매칭 성공만(결정론 평가기) → 날조/실패 궤적 배제.
4. **변환+randomize**: 기존 `fc_convert_sopbench`(+`fc_randomize_fetchable`) → fetch-gather 값도 randomize(copy 강제). = v6 파이프라인에 합류.
5. **v7 SFT**: v6 데이터(threading) + withholding-gather rollout(신규) → 7B LoRA.
6. **eval**: 3-way+ (`tau2_eval_adapter.sh`) + `tau2_autopsy.py` — **핵심 지표 = order_id가 이제 get_user_details로 fetch되나**(autopsy: get_user_details 호출율↑·#W0000000 날조율↓) + compliant-pass.

## 6. 결정론·검증기 불변 (준수)
- 생성기(teacher)=LLM은 허용(생성기에만 LLM). **success 필터=결정론 평가기**(action_should_succeed)·**withholding 구성=결정론**(getter_map). 선별/검증에 LLM-judge 도입 0 = `feedback-selector-verifier-deterministic` 부합.

## 7. 리스크 / 열린 질문 (리뷰)
- **R1**: SOPBench에 "getter-입력이 다른 getter-출력"인 진짜 2-hop 체인이 있나? (§4c census 선행 — 없으면 v7은 1-hop fetch-gather만 강화, order_id류 2-hop은 부분 전이만 기대.)
- **R2**: withholding 시 teacher가 실제로 gather하나, 아니면 teacher도 묻기만/날조? (teacher 품질 = 데이터 품질 상한 — 소량 파일럿으로 선검증.)
- **R3**: user_known에서 무엇이 "getter-생산"인지 결정론 매핑 — getter_map은 *condition→getter*. user_known 키↔getter 출력 슬롯 매핑 정합성 확인 필요.
- **R4**: τ² 전이 = SOPBench getter-슬롯 의미(balance·score)와 τ² order_id 의미가 달라도 *규율*(없으면 lookup 호출)이 전이되나 = R1 도구이름 전이 실증의 값-버전. 미지수(이게 thesis 핵심 베팅).
- **R5**: 비용 — teacher rollout 생성 = API 비용. 규모 산정 필요(파일럿 N=50/도메인 후 확대).

## 8. 성공 기준 (사전등록)
- **1차(기제)**: v7 autopsy서 **get_user_details(또는 lookup) 호출율 ↑ & order_id 날조(#W0000000) ↓** vs v6. (날조→fetch 전환 실증.)
- **2차(점수)**: τ² compliant-pass v6 대비 ↑·base 0.17 돌파 지향. 0.3+면 강한 전이.
- 음성(2-hop 미전이)도 1급 진단(어느 층서 막히나: gather 추론 vs select vs 의미매핑).

## 8b. ★재좌표 (사용자 리뷰 + M1 census + in-dist eval, 2026-06-15)
- **M1 census 확정**: 학습 7도메인서 tool-출력→arg(2-hop) = **1.9% 희소**(online_market b=0). customer_service(session_token 2-hop)는 **참조 온톨로지·executor 없음=학습불가**. 변환기는 보존함(P0 아님)·소스가 구조적 부재.
- **★in-dist eval(v6 online_market): mean-pass 0.60·success 0.33·dirgraph 0.33 ≫ base 0~21%** → in-dist 안 떨어짐·**τ² 0.05는 순수 전이 문제(R4)**. 모델 정상. dirgraph가 최약=전이 타깃.
- **★헤드라인 재서술(사용자)**: "빈 칸 채우기" → **"R(특히 R4/dirgraph 시퀀싱)을 SOPBench/TaskBench서 학습·ABox-swap으로 각 벤치 base→frontier로 올리면 τ²/SOP-Bench로 *커플 전이*"**. 학습신호 = **in-dist dirgraph↑가 τ²↑와 커플되는 부분**(커플링 실험으로 실증 중).
- **★2-gap**: τ² = (i)시퀀싱(R4·SOPBench 있음·전이가능) + (ii)2-hop id-lookup gather(부재). (i)이 (ii) binding까지 끌어올림 → (ii) 소스 = 3번째 벤치.
- **★3번째 벤치 = Seal-Tools**(딥리서치 확정): 현실 서비스-API 엔티티·`API_call_N` 출력→arg·586 nested·Apache-2.0·gold JSON(결과 합성)·변환 LOW-MED. **반드시 value-randomization 결합**(심볼참조→복사강제). 2순위 BFCL V3 multi-turn(grounded). 보조 NESTful(수학만·gap 불충족). 기각: ToolBench(CC-BY-NC=주권충돌)·AppWorld(REPL+Amazon중첩)·API-Bank(API-검색≠값-fetch).
- ~~⇒ v7 학습 = SOPBench + TaskBench + Seal-Tools~~ → **★★정정(2026-06-15): Seal-Tools 강등·ComplexFuncBench 승격 (§8c).**

## 8c. ★★3번째 벤치 재순위 (2026-06-15, 추가 딥리서치 3클러스터 + ComplexFuncBench 1차 데이터 검증 — §8b Seal-Tools 결정 정정)
> 동기 = "AppWorld/ToolBench 외 2-hop 벤치 더 탐색"(사용자). 3 병렬 에이전트(전부 arXiv+repo+라이선스 1차 검증). **핵심 = Seal-Tools 1순위는 잘못된 축 가중.**

- **★결정적 구분 = 단발-심볼형 vs 멀티턴 observe-then-use**:
  - **단발 DAG 생성**(Seal-Tools·TaskBench·NESTful): 지시문→전체 plan 한 번에, 심볼참조(`API_call_N`/`<node-N>`) threading. **실제 출력 관찰 없음.**
  - **멀티턴 observe-then-use**(ComplexFuncBench·BFCL·RestBench·τ²): 호출→**실제 응답 관찰**→출력서 값 추출→사용.
  - **우리 gap(order_id fetch해 쓰기)은 후자.** Seal-Tools=전자=**TaskBench 동류**(사용자 "동형" 관찰 확인). **v6은 이미 TaskBench threading(41%)로도 order_id 날조** → Seal-Tools 추가 = v6이 불충분 입증한 종류를 더 넣는 것. 관찰할 실제 출력이 없어 grounded fetch 학습 구조적 불가.
- **★새 1순위 = ComplexFuncBench** (arXiv `2501.10132` v1, HF `zai-org/ComplexFuncBench`, apache-2.0): Booking.com 실 API(호텔/항공/렌터카/택시/관광), **추론형+grounded**. verbatim: *"LLMs are expected to infer the correct parameter values based on user constraints **and API responses**."* 1,000샘플·평균 5.07 call·3단계 인간검수. Seal-Tools 약점 두 축(추론형·grounded) 둘 다 충족 + 멀티턴 observe-then-use = 우리 native-FC 타깃과 *동형*(변환 더 쉬움).
  - **✅BLOCKING #1 해소(1차 데이터 검증)**: gold에 **녹화 API 응답 포함**(`role:"observation"`에 Booking JSON verbatim; 위경도 `32.873055/-117.215935`가 다음 호출 `pick_up_latitude/longitude`로 threading 실증) → **라이브 RapidAPI 없이 SFT 소스 가능.**
  - ⚠️**남은 BLOCKING(채택 전)**: ①**라이선스 공백** — zai-org 선언 apache-2.0이나 GitHub LICENSE 부재 + 데이터=Booking/RapidAPI 파생→**하류 ToU가 응답값 재배포에 별도 적용 가능**(카드 미언급). 주권 프레이밍상 실사 필수. ②**flights↔airline 근접도메인**=τ² 전이가 근접이라 SOP-Bench 원거리보다 약한 증거(헤드라인=원거리·근접=보조). ③**ComplexEval=LLM-judge 혼합**→불변 위배→**gold 궤적만 소스, eval은 결정론 재구현**(타입→인자-타입→사전조건→replay).
- **보강**: RestBench(`2306.06624`, **MIT**)=추론형 전형(TMDB/Spotify `user_id`/`playlist_id` threading·명시단서 없음) 단 소규모(2도메인~157). BFCL V3 Miss-Params(apache-2.0·grounded·결정론)=**D5 ask/fetch 게이트 직결**·ask-eval 최적.
- **강등/제외**: Seal-Tools=단발-심볼형(TaskBench 동류·신규정보≈0)→선택 augmentation·저우선. ToolHop(apache/cc-by)=합성코드→시퀀싱 보조만. ToolSandbox(Apple 비상업)·ToolBench(데이터 research-only)·τ²(contamination)=제외.
- **⇒ v7 2-hop 소스 = ComplexFuncBench**(Seal-Tools 대체). 레시피 정정: `sft_v7 = sop_rand2 + d5_ask2 + tb_all_v4 + complexfuncbench`. **tb_all_v4가 이미 단발-threading 운반 → seal_tools는 이중 잉여.**

### 8c-BLOCKING#2 해소 — ToU 실사 결론 + 논문/특허 소스 분리 (2026-06-15)
- **1차 소스 확인**: HF `zai-org/ComplexFuncBench` 메타=apache-2.0이나 **GitHub `THUDM/ComplexFuncBench` LICENSE 파일 404(부재)** → repo 적용 라이선스 모호. 데이터=**Booking.com API via RapidAPI**(43 API·5도메인·repo가 response-eval에 RapidAPI 구독 요구). 응답=curated 저장(gold 포함). **HF·repo·논문 어디에도 Booking/RapidAPI 파생 응답의 사용/재배포 ToU 진술 없음 = 공백.**
- **판정**: apache-2.0(설령 적용돼도)은 저자 코드·주석만 덮음 — **Booking의 응답 *콘텐츠*는 재라이선스 불가**(Booking/RapidAPI ToU는 통상 응답 caching·재배포 제한). 공백=미해결(안전 아님).
- **리스크 보정**: 연구/논문 **LOW**(널리 인용되는 벤치 표준사용) · **상업/주권/특허 MED-HIGH**(재배포권 불명확 데이터로 배포모델 학습=책임 + clean-provenance 셀링 훼손).
- **★결정(사용자 2026-06-15): ComplexFuncBench로 *논문 먼저*** — 방법-검증(observe-then-infer-gather 전이) 연구단계는 LOW 리스크로 진행. ToU caveat 박제.
- **★★특허/프로덕션 = clean 소스 필수(명시)**: **ComplexFuncBench 응답으로 학습한 모델은 특허·배포에 사용 금지.** 방법 검증 후 **clean-provenance 소스로 동등 gather-데이터를 *재생성*** 해 특허·프로덕션 모델 학습. clean 후보: **우리 SOPBench user-sim withholding(완전 소유=최우선)** / RestBench(MIT) / BFCL(apache·단 데이터 출처 재확인). = 주권/특허 셀링 보존·ToU 노출은 연구단계 국한.
- **eval 불변(BLOCKING #3 재확인)**: ComplexEval LLM-judge 미사용 — gold 궤적만 SFT 소스, eval=결정론 재구현.

## 9. 마일스톤
M1 §4c census(2-hop 체인 유무) + §7-R3 매핑 확인 → M2 withholding 구성 스크립트(결정론) → M3 teacher 파일럿(N=50·R2 검증) → M4 전량 생성+변환+SFT(v7) → M5 eval(3-way+autopsy).
