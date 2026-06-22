# HANDOFF 2026-06-22 PM — 32B→frontier 갭 = flow 규율 + Opus-as-agent 검증 + ★다음=11 both-fail 전수풀이

> **진입 = 메모리 `06-NOW`(단일 진실원) + 이 handoff.** 직전 = `HANDOFF_2026_06_22`(AM·키스톤/통제프레임).
> ★**다음 세션 (사용자 지시) = both-fail 11개 task 전부 Opus-4.8-agent로 풀어 분석**(§다음 세션).

## 0. 오늘 서사 (한 단락)
통제프레임(C0..C4) 설계 → fetch-first 7B 실패 원인 확정(**schema-example-copy prior·M4**·scale-의존 0.47→0.045→0.006·지시저항·시연민감·maxprompt 역효과·프롬프트한계 딥리서치 [[42]]) → **learn 전이 실패 확정**(SFT 52·DPO-fab 35·abstract→real 안 됨) → **fetch/select 분담 정식화**(fetch=결정론·select=LLM·`FETCH_SELECT_DIVISION`)·learn을 A→**B1(selection)** 재타깃 → banking prior 확인(금융도 user_id 날조·인프라벽5 해소) → **★사용자 전략전환: 32B면 fetch-first 통과→상용=32B·진짜=32B→frontier 갭** → **32B vs frontier 전수census**(갭=flow완수/복구 53% > operand 40%·32B pass-any3 0.77≈gpt-4.1 0.82) → leaderboard 확인(top retail=Claude Opus4.6 0.92·gpt-4.1=중상위) → **★Opus-4.8 직접 T105 풀이=gold정확·진단: gpt-4.1도 operand맞히나 flow서 무너짐**.

## 1. ★최상위 결론 (이 세션 핵심)
- **32B→top-frontier 갭의 최대 레버 = flow 규율/완수/복구(53%) > operand(40%)**. 실증: T105서 gpt-4.1이 operand(변종 부분-스펙 매칭) 정답 후 *redundant 잘못된 재시도*로 실패. top-frontier(Opus)=정답 한 번 내고 STOP.
- **상용=32B 전제**(fetch-first 통과·pass-any3 0.77≈frontier). 연구가치=32B→frontier cheap-replication: **flow-discipline controller(결정론 여지)** 우선 + operand B1.
- learn(SFT/DPO)은 abstract→real *전이 실패*(A=날조). autofetch(결정론)만 grounding 작동. fetch=결정론/select=LLM 분담 확정([[10]] 정합·`FETCH_SELECT_DIVISION`).

## 2. ★★다음 세션 = both-fail 11개 전수 풀이 (사용자 지시)
**대상 = 32B(3시행 전부)·gpt-4.1 *둘 다* 실패 = top-frontier-only 후보 11개**: `2, 3, 4, 20, 21, 99, 100, 103, 105(완료·gold일치), 109, 110`.
**목표**: 각 task를 Opus-4.8-agent로 풀고(시나리오+실데이터 추론), gold 대조 검증, 32B/gpt-4.1 실패원인 분류 → **top-frontier 능력 종합지도**(operand vs flow-discipline vs completion vs recovery vs *unsolvable*).
- ★정직: 0.92 천장 = Opus도 ~8% 실패 → 11개 중 일부는 *genuinely unsolvable/ambiguous/mislabeled* 가능. Opus도 못 풀면 그렇게 분류(학습가능 능력 아님).
- **방법(T105서 확립)**:
  1. `frontier_solve_kit.py --task <id> --brief` = 시나리오 + 실데이터(user record·orders·관련 product 변종 카탈로그 from db.json) **gold 비공개** 출력.
  2. Opus 추론 → 행동 시퀀스(부분-스펙 규칙: 명시 안 한 속성=현재값 유지·변종 avail 확인·flow: 정답 후 STOP).
  3. `frontier_solve_kit.py --task <id> --check '<json actions>'` = gold 대조(action-match) + gold 공개.
  4. 32B/gpt-4.1 궤적 읽어 실패원인 분류(operand/flow/completion/recovery/unsolvable).
  - 헬퍼 = `scripts/distill/tau2/frontier_solve_kit.py`(이 세션 작성·commit). 무결성=brief는 gold 숨김·check서만 공개.
- **산출**: 11-task 분석표(각: 액션수·Opus solve✓/✗·32B실패유형·gpt-4.1실패유형·능력분류) → `THIRTYB_VS_FRONTIER_GAP` §확장 + cheap-replication 타깃 확정(flow-controller 설계).

## 3. 자산·실측 (commit)
- **문서**: `THIRTYB_VS_FRONTIER_GAP_2026_06_22`(갭census+T105검증·6007621)·`FETCH_SELECT_DIVISION_2026_06_22`·`FETCHFIRST_PROMPT_FAILURE_MECHANISM_2026_06_22`(M4+딥리서치)·`RULE_LEVER_COST_EFFICIENCY_PROGRAM`·`LLM_CONTROL_EXPERIMENT_REDESIGN`·`C4_LEARN_FETCHFIRST_CROSSOVER`.
- **데이터**: 32B=`data/simulations/on_n32int8_floor_retail`(3trial)·frontier=`retail_gpt41_nogate`(gpt-4.1 1trial)·변종카탈로그=`data/tau2/domains/retail/db.json`·gold=`tasks.json evaluation_criteria.actions`.
- **leaderboard(6/22)**: retail Claude Opus4.6 0.92·Sonnet4.6 0.917·GPT-5.2 0.82 / airline LongCat-Thinking 0.765·GPT-5.1 0.67. Qwen32B 미등재(Simia-Tau FT retail 0.617≈우리 32B-int8 0.60).
- **실측표 (retail schema_copy·앞 세션)**: base40·prompt44·skill49·fewshot-dg47·fewshot-retail0·scaffold-perform32·SFT-learn52·DPO35. = 도메인일반 learn 전부 prior 못닫음·autofetch만.

## 4. 진행중/보류
- 🔄 **B1-select DPO**(GPU0 b1 beta0.1·GPU1 b5 beta0.5·`overnight_bselect.sh`): bsel_af eval 마무리중. 회수=`grep -h BSEL_ROW /home/woori/scratch/overnight_bselect_*.log`·마커 `BSEL_DONE_*`. 판정=bsel_af의 B_wrong_write < base_af?(operand 40% 레버). 폴러 `bmrxhaipl`.
- ⏸️ **banking autofetch 전이**: prior 확인됐으나 producer가 name-lookup(get_user_information_by_name)이라 autofetch @auth_user 패턴과 달라 *엔진 손질 필요*. `t2_run_gated` banking 분기(no_knowledge+샌드박스stub)는 됨. 대화형으로.
- ⏸️ flow-discipline 레버 설계(53% 레버·결정론 controller "의도충족 write 후 추가write 차단/확인" vs learn).

## 5. 불변·함정
- [[05]] 결정질문 행동전 명시·tau2 학습금지·single-facet full-agent평가 금지([[03]])·CDP GitHub금지([[32]])·진행률 가시([[30]]).
- ssh_run: `cd /c/workspace` 먼저·`/path`로 시작하는 --cmd는 MSYS 망가짐(비-경로 토큰 선행)·PipeTimeout=백그라운드 채널잡힘(launch는 성공·setsid)·heredoc f-string 이스케이프 따옴표 금지·remote autoresearch 커밋有(push전 fetch+rebase)·gpt-4.1 user-sim COST GUARD·banking=OpenAI키`/home/woori/.openai_key`(no_knowledge면 불요).
