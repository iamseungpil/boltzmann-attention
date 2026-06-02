# Rung1 구현 핸드오프 — login 특별취급 제거 + required_set 균일화 + R3 종료

> 작성 2026-06-02. 대상 = 구현 세션(이 대화 맥락 없음 가정·단독 실행 가능).
> 마스터 = `EXPERIMENT_DESIGN.md`(§0~§4). 결과 권위본 = `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`(Exp-4-rung1-trained).
> 이 문서 = 그 §3 Rung1 ①(SFT)의 **구현 세부**. 목표·순서·지표 변경은 마스터에서만.

---

## 0. 한 줄 목표 / 이 작업이 고치는 것

**teacher가 over-login을 학습 데이터에 직접 생성하던 결함을 제거한다.** login 같은 도메인-특수 기능을 *특별취급(별도 establish-phase)* 하지 말고, **required_set의 평범한 멤버**로 균일하게 다룬다. TBox가 배우는 규칙은 단 하나 — **"required_set에 정의된 도구만, 의존순서로 호출 → 다 되면 act, 아니면 stop"**(도메인-일반 실행 불변식). login이라는 개념은 TBox에 존재하지 않는다.

이로써 §3의 실행 불변식 R1–R4가 전부 "required_set 충실 실행" 하나로 붕괴하고, R1(parsimony=불필요 호출 금지)은 **자동 충족**된다(set 밖이면 안 부름).

---

## 1. 근거 (코드·실측, 추측 아님)

### 1.1 over-login의 코드상 근원 = teacher의 establish-phase 특별취급
`scripts/distill/sopbench/build_tbox_planner_sft.py`:
- **블록 C (`ests`), L169-179**: goal의 login/auth establishable을 **`ont["operators"][goal]["precondition"]`(제네릭 operator precond)** 에서 뽑아(L171), creds 가용 시(L178) 시퀀스에 주입. → login이 *이 태스크에 불요여도* operator precond에 있고 creds가 있으면 teacher가 호출 시연.
- **L159**: 반대로 task constraint의 establishable은 `continue`로 **건너뜀**.
- 즉 login은 "task가 실제로 요구하는 required_set(=`task["constraints"]`)"에서 빠지고, **제네릭 precond + creds 휴리스틱**이라는 별도 경로로 들어온다. → teacher 데이터가 "always login"을 가르침 → 모델 over-login.

### 1.2 evaluator-정합성도 이 수정 편
- GT 의존성 = `task_dep[goal] = task["constraints"]` (**L131-132**). evaluator가 보는 required = **task["constraints"]**이지 제네릭 operator precond이 아니다. teacher가 evaluator와 **다른 소스**를 쓰고 있었던 것이 버그의 본질.

### 1.3 실측(Exp-4-rung1-trained, 권위본)
- should_T 45 실패 = **(A) REFUSE_after_login_False 19/45**(login 불요 태스크서 login 호출→False→거부) + **(B) ACTED_but_dirgraph0 16/45**(goal 성공했으나 4-8회 반복·미종료) + 조기ACT 4 + 잔여 6.
- gather-grounding은 **달성**(정책조건 getter 호출 0→43%). 즉 게더는 학습됨. 병목은 (A)(B).
- (A) = 위 1.1 establish-phase가 근원. (B) = teacher가 ACT에서 즉시 `break`(L281)라 **post-success 종료 예제 0개**.

### 1.4 ⚠️ "over-login = prior-override"는 **미확정**(설계서에 단정 금지)
- s1(source=1) 진단이 prior-override를 시사했으나 **렌더 confound 미해소**: goal `needs[]`/STATUS는 `goal_constraint`가 명시될 때만 task-pruned, 아니면 풀 precond(login 포함)로 렌더됨(`two_stage_client.py` **L158-160**). eval은 `_lighten`일 때만 gconstr 전달(**L477-478**), teacher는 아예 안 넘김(L238). → s1 프롬프트가 login을 "BLOCKED→먼저 호출"로 *명시 렌더*했을 공산이 큼 = 모델이 따른 것이지 무시한 것 아님.
- **본 수정(establish-phase 삭제 + required_set 균일)이 teacher 측 over-login을 제거**하므로, prior-override 여부는 *수정 후 재측정*으로 판별한다(사전 단정 금지).

### 1.5 ★login 외 도메인-특화 학습 전수 감사 (2026-06-02, 코드 `build_tbox_planner_sft.py`+`build_v2_prompt` 전수 검색)
**결론: 학습 *타깃*에 영향하는 도메인-특화 학습은 login 특별취급(+프롬프트 쌍둥이)이 유일하다. 그 외 teacher에 도메인-특화 *가정* 2건(GETTER bank 손-map · accounts/username slot 확장)이 있어 cleanup 필요(모델 누수는 없으나 균일성·train/eval 정합). 나머지는 ABox-입력이거나 alias로 보호됨.**
- **(1) ★프롬프트-측 establish 쌍둥이 (T3로 반드시 *함께* 제거 — T1만으론 불충분)**: `build_v2_prompt`가 (a) **"HOW TO ESTABLISH preconditions: to establish 'X', call Y"** 블록(L253/259), (b) establishable 미충족 시 **"BLOCKED — first call: login_user"** STATUS(L172)를 렌더 = teacher establish-phase의 프롬프트 판. teacher만 고치고 프롬프트를 두면: (i) login이 프롬프트에서 여전히 특별 개념으로 노출, (ii) **source=1(s1)에서 needs/STATUS가 full operator precond(login 포함)로 렌더되는데 T1 후 teacher 타깃은 login-free → train/eval 불일치**.
  - ★**source=3(헤드라인 alias_s3)은 이미 깨끗**: source=3 템플릿(L210-225)은 needs/STATUS·HOW-TO-ESTABLISH를 **렌더 안 함**(도구 설명만) → 프롬프트-측 login 노출 0 → **헤드라인 regime은 T1(teacher)만으로 충분**(모델이 required_set을 NL서 추론). = §1.4 confound는 *s1 전용* 현상.
  - source=1(s1, 비교군)만 needs를 **task-pruned로 렌더**해야 깨끗(안 그러면 s1 = alias 효과 + login-렌더 효과가 혼동). → T3에 포함.
- **(2) GETTER_BY_DOMAIN 하드코딩 bank dict (L76-88)**: bank만 손-map, 타 6도메인은 auto-derive(`getter_map.json`) → **bank 특별취급(불일치)**. LODO(holdout=bank)엔 bank teacher가 학습에 안 쓰여 *모델 누수는 없음*. 단 균일성 위해 **삭제하고 auto-derive에 일원화 권장**(cleanup).
- **(3) accounts/username slot 확장 (L137-141) — 도메인-특화 가정 + train/eval 비대칭**: teacher가 `task["initial_database"]["accounts"][username]` 필드를 slots에 평탄화(**"accounts"·"username" 리터럴 하드코딩**). (a) bank-중심 identity 스키마(university=student_id·hotel=guest 등엔 부적합), (b) `slots`→`set(slots.keys())`가 프롬프트 "ALREADY KNOWN/ESTABLISHED"로 렌더되는데 **eval-side `_update_slots`(`two_stage_client.py` L563-579)는 도구결과 dict + "Here is all the information" 덤프를 키-무관 일반 수확**(account 특수처리 0) = **train/eval 분포 불일치**. 학습 *타깃*엔 직접 영향 없으나(slot은 입력측) 프롬프트 known-state를 도메인별로 비대칭화. → **T3d cleanup**.
- **(4) 누수 아님 — 확인된 것**: getter-선택(condition→getter)은 **alias로 보호**(alias_s3서 도구명 마스킹→설명 의미매칭 강제, 암기 불가); establishable/condition **kind 구분은 ABox 구조**(T1 후 라우팅 균일하면 정당); 인자 채움은 **resolver 몫**(planner SFT 타깃 아님); `should_succeed` 터미널은 **GT 출력 라벨**(일반 예측 스킬). → **planner-타깃에 login 외 도메인-특화 학습 없음**.

---

## 2. ★구현 전 단일 게이트 (코딩 시작 전 반드시)

**required_set 소스 검증 1건** — 리모트(rr.ps1) 실행. 로컬 python=Store 스텁이라 측정 전부 리모트.

- **확인할 것**: 몇 개 goal에 대해 *evaluator가 실제로 요구하는 의존성*(innate `dep_innate` + `task["constraints"]` 합성 결과) == `task["constraints"]` leaf 집합인가?
- **분기**:
  - **같으면** → required_set = `task["constraints"]` leaf 그대로 사용(아래 §3). login은 제약에 있을 때만 required = 정답.
  - **다르면**(일부 goal이 login을 **innate**로 요구하는데 constraints엔 없음 = 메모리 PartB mismatch) → required_set 소스를 **"evaluator-authoritative 합성 의존성"**으로 정의(여전히 균일·login 비특수). **원칙 불변, 소스만 확정.**
- **왜 게이트인가**: task["constraints"]가 권위 소스가 아니면 establishable을 빼는 순간 **under-login**(goal precond 미충족→실패)으로 갈 수 있음. 이 한 가지만 확인하면 안전.
- 검증 스니펫 위치 제안: `scripts/distill/sopbench/precheck_required_source.py`(신규, ~30줄: 도메인별 N개 task에서 `dep_innate[goal]` vs `task["constraints"]` leaf diff 출력).

---

## 3. 구현 작업 (순서대로)

파일: `scripts/distill/sopbench/build_tbox_planner_sft.py`

### T1. required_set 균일화 (login 특별취급 제거) — (A) 19건 표적
- **L159의 `if kind == "establishable": continue` 삭제.** establishable leaf도 required_set에 포함하되 **균일 매핑**:
  - check leaf(callable, tool_names에 있음) → 자기 자신 (현 L161-162 유지)
  - condition leaf → getter (현 GMAP/GETTER 경로 유지, L163-168)
  - **establishable leaf → 그 `by` 액션**(login 등). predicate info의 `by`로 도구 결정, tool_names에 있으면 required에 추가.
- **블록 C(`ests`, L169-179) 전체 삭제.** creds-availability 게이팅(L178)도 삭제.
- **next_decision step 3(establish phase, L219-223) 삭제.** 이제 step 1(gather required)이 login도 포함하므로 별도 단계 불요.
- 결과: `required_set` = `task["constraints"]` leaf 전부(균일, establishable 포함). next_decision = "required 중 미실행 도구 호출 → 다 되면 ACT(should_succeed) else STOP." **도구별/도메인별 분기 0. TBox에 login 개념 없음.**
- 의존순서: establishable(login)이 그것을 전제하는 check보다 먼저 호출되도록 정렬 필요할 수 있음(login→auth-필요 check). 1차는 leaf 순서로 두되, 재학습 후 순서위반 보이면 위상정렬 추가.

### T2. R3 종료(post-success exit) — (B) 16건 표적
- **근원**: L281 `if target in ("STOP","ACT",goal) or target in executed: break` 가 goal/ACT emit 즉시 루프 종료 → post-success 예제 0개.
- **수정**: goal/ACT가 should_succeed일 때 즉시 break하지 말고, **goal 실행(상태 전진) → 다음 스텝에서 종료 예제 1개 추가**.
  - next_decision에 새 분기: "goal이 executed에 있고 관측 성공(observed/history) → 종료 토큰 반환."
  - **종료 토큰 포맷(scratchpad)**: refuse-STOP과 구분되도록 **`ready=true; done=true; STOP`** (done 플래그). eval 파서는 STOP/exit_conversation을 동일 종료로 접으므로(`two_stage_client.py` L507-518) 루프는 정상 종료되고, evaluator는 "goal이 성공 호출됐나"로 should_T/should_F를 구분(토큰 충돌 없음). **단 `permitted=true; STOP` 같은 모순형은 금지** → done 플래그로 명시.
  - ⚠️ **`target in executed: break` 무한루프 가드는 유지.** goal-break만 완화(should_succeed & goal 1회 성공 후 종료 1스텝 추가). should_F는 현행대로(goal 미호출, STOP).
- **빌드-타임 assertion 추가**: "history에 goal-success가 있는데 그 뒤 tool-call(비종료) 타깃이 있는 예제 = 0." 위반 시 빌드 실패.

### T3. 프롬프트(`build_v2_prompt`) — 종료 토큰 + ★login 쌍둥이 제거 (§1.5)
- **T3a 종료 분기**: scratchpad 룰 블록(L216-226)에 1줄 추가: "goal이 이미 성공 호출됨 → `ready=true; done=true; STOP`(종료)." eval 파서(L507-530)가 done-STOP을 정상 종료로 처리하는지 확인(현재 STOP→exit_conversation이므로 동작; done 플래그는 파싱 영향 없음).
- **T3b ★프롬프트-측 login 특별취급 제거 (§1.5-(1), T1과 짝)**: establishable을 condition/check와 **균일 렌더**.
  - source=1 STATUS의 establishable-전용 "BLOCKED — first call: login_user"(L172)를 일반 "VERIFY/CALL: <도구>"로 통합(establishable·condition 동일 처리).
  - **별도 "HOW TO ESTABLISH" 블록(est_str, L253/259) 제거** — establishable이 required_set 멤버로 needs에 균일 포함되므로 중복.
  - ★**source=1 needs/STATUS를 task-pruned required_set으로 렌더**(현재 gconstr=None시 full operator precond=login 포함 → T1 후 teacher 타깃과 불일치). teacher가 쓰는 동일 required_set 소스로 통일.
  - ⚠️ **source=3(헤드라인)은 needs/establish 미렌더라 변경 불요**(이미 깨끗) — T3b는 s1 비교군 정합용. 단 *하지 말 것*: source=3에 required_set을 needs로 노출(=answer-key, §5).
- **T3c (cleanup) GETTER_BY_DOMAIN 하드코딩 bank dict 삭제** → auto-derive(`getter_map.json`) 일원화(§1.5-(2)). 모델 누수는 없으나 균일성.
- **T3d (cleanup, `build_tbox_planner_sft.py` L137-141) accounts/username slot 확장 삭제** → `task["user_known"]` 기반 **일반 slot 도출**로 eval `_update_slots`와 정합(§1.5-(3)). ⚠️적용 후 `truth()`/`resolve_args` 라벨이 불변인지 census 확인(slot 축소가 GT 평가 인자를 깨면 안 됨).

### T4. 데이터 재생성 + 검증
- LODO holdout=bank, non-bank 6도메인 학습셋. alias regime은 헤드라인이므로 **`--alias --source 3 --scratchpad`**(alias_s3_scratch)와 비교군 `--source 1 --scratchpad`(s1_scratch) 둘 생성.
- 빌드 후 검증(rr.ps1):
  - `ready=false` 뒤 ACT가 한 번도 없는가(0).
  - establishable(login) 타깃이 등장하는 태스크 == task["constraints"]에 establishable leaf 있는 태스크(특별취급 0 확증).
  - post-success 종료 예제 수 > 0, T2 assertion 통과.
  - 터미널 정확도(should_succeed 정합) census.

---

## 4. 측정 (재학습 후, 잠긴 지표)

`EXPERIMENT_DESIGN §3 결정②` 지표 위계로 보고(net 금지·BOTH 단독판정 금지):

- **1차 = ACT-recall | 충분게더** (게더 완료·정답=ACT인 태스크에서 실제 ACT 비율). 현 baseline = 0/9(s1)·2/18(alias_s3). 목표 = 상승.
- **분리 보고 = STOP-recall**(should_F 정답). 현 baseline = 36%/30%. **비회귀 가드**(T1이 거부를 줄이는 방향이라 should_F 악화 위험 — 반드시 분리 모니터).
- **3차 가드레일 = ordering-violation**(조기 ACT). 붕괴 모델은 ACT를 안 해 0=만점이 되므로 1차로 쓰지 말 것.
- **헤드라인 = BOTH**(dirgraph∩goal) = 위 분해의 산출물. 현 0-2 → 목표 다수(G-SFT: ≥15/48).
- **레버별 기대**:
  - T1(login 균일) → (A) 19건 표적 → ACT-recall↑ AND STOP-recall 비회귀.
  - T2(종료) → (B) 16건 표적(disjoint) → 반복-goal 소멸·dirgraph 회복. should_F 무영향(종료 예제는 STOP-bias 아님).

---

## 5. 하지 말 것 (anti-patterns)

- ❌ login/auth를 도메인별로 손-처리하거나 creds 휴리스틱으로 게이팅. **모든 도구는 required_set 멤버로 균일.**
- ❌ teacher establish-phase만 고치고 **프롬프트(`build_v2_prompt`)의 establish 쌍둥이(HOW-TO-ESTABLISH 블록·"BLOCKED→first call login" STATUS)를 방치**(§1.5-(1)). T1·T3b는 **반드시 짝으로**. (안 그러면 s1 train/eval 불일치 + 프롬프트가 여전히 login을 특별 개념으로 노출.)
- ❌ source=3/alias에서 required_set(정답)을 `needs[]`로 렌더. = answer-key = anti-cheat 붕괴 + NL→dirgraph 컴파일이 RoG식 그래프-실행으로 강등(novelty 상실). **규칙(도구·condition·getter)은 ABox에 명시 OK / "이 요청에 무엇이 발화하는가(required_set)"의 task-조건화는 *렌더 아니라 TBox 학습*.**
- ❌ "over-login = prior-override"를 설계서에 단정. 본 수정 후 재측정으로 판별(§1.4).
- ❌ goal-break 완화 시 `target in executed` 무한루프 가드 제거.
- ❌ 로컬 python으로 측정/인용. 전부 rr.ps1 실측·RC·scanned 확인 후만 "실측". fabrication 금지.

---

## 6. 다음 사다리 (이번 작업 후)

1. T1+T2 재학습 → §4 측정. BOTH ≥ 다수면 G-SFT 통과.
2. **잔여 over-login이 source=3에서 지속하면** = teacher-유발분 제거 후의 *진짜* NL→required_set 추론 실패 → ABox/TBox 시너지 사다리:
   - **L1 ABox-ablation**(eval only·헤드라인 겸용): 빈/틀린/셔플 ABox → 행동 Δ. 불변이면 prior 지배 = 전이 thesis 적신호.
   - **L3 반사실 ABox-swap 학습**(데이터·thesis-안전): 같은 NL·required_set 다른 ABox 쌍 → 고정 prior로 못 풀게 강제(= §3.0b 합성에 paired-required-set).
   - **L4 xattn ABox-memory**(Rung3): L3도 prior가 이기면, 이제 측정 기반으로 정당화.
3. SFT 음성-부재 잔여(터미널-전 중복 게더 등) → ② DPO(`build_dpo_pairs`/`dpo_train`, init=Rung1 어댑터).

---

## 7. 인프라 (반복 사고 방지)
- 측정 전부 리모트: `rr.ps1`(메시지당 1호출·병렬 형제취소 주의). `git pull` 후 작업(체크아웃 behind 잦음). 배포=git push(SFTP 금지, branch=facet-rft-2026, 자동 commit+push 허용).
- solo 학습(2잡/48GB OOM). vLLM kill 잔여 `/dev/shm/vllm*` 정리 + GPU별 PID kill.
- 리모트 SOPBench clone: `/home/woori/scratch/SOPBench`. env=`seka_env`(py3.12, peft 0.19). 어댑터명 `tbox_v2`=FC통과. 상세=`reference_infra_setup.md`/`REMOTE_ENV.md`.

## 8. 코드 레퍼런스 요약
| 위치 | 내용 | 작업 |
|---|---|---|
| `build_tbox_planner_sft.py` L159 | establishable `continue` | T1 삭제 |
| 〃 L169-179 (블록 C `ests`) | login 특별 establish | T1 삭제 |
| 〃 L219-223 (next_decision step3) | establish phase | T1 삭제 |
| 〃 L161-168 | required 매핑(check/getter) | T1 establishable→`by` 추가 |
| 〃 L281-282 (break) | goal/ACT 즉시 종료 | T2 완화(가드 유지) |
| 〃 L131-132 | `task_dep[goal]=constraints` | §2 소스 검증 근거 |
| 〃 L137-141 | accounts/username slot 확장(도메인 가정+train/eval 비대칭) | T3d 일반화(user_known 기반) |
| `build_tbox_planner_sft.py` L76-88 | GETTER_BY_DOMAIN 하드코딩 bank dict | T3c 삭제(auto-derive 일원화) |
| `two_stage_client.py` L216-226 | scratchpad 룰 | T3a 종료분기 추가 |
| 〃 L170-175 | establishable STATUS "BLOCKED→first call login" | T3b 일반화(login 특별취급 제거) |
| 〃 L253, L259 | "HOW TO ESTABLISH" est_str 블록 | T3b 제거(needs로 균일 흡수) |
| 〃 L507-530 | 터미널 파서 | T3a done-STOP 확인 |
| 〃 L158-160, L477-478 | goal_constraint 렌더 게이트(full precond vs task-pruned) | §1.4 confound 근거 / T3b source=1 task-prune |
