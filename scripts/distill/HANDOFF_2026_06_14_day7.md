# HANDOFF 2026-06-14 day-7 (★plan X 착수·native-FC 변환기 구현·τ² 전이 실측·R1b 신설·값-randomization이 날조 제거)

> 📌 마스터 = `EXPERIMENT_DESIGN.md` §0 배너(plan X). 현 진입점 = 이 문서 + `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md`. 진행로그 = `AUTONOMOUS_PROGRESS_2026_06_14.md`. 불변 = memory `feedback-thesis-tbox-transfer-direction`·`feedback-selector-verifier-deterministic`(★★★). 직전 = day-6.

## 0. ★★★ 현 방향 (불변)
**plan X**: 작은 오픈웨이트 모델이 **R1-R8 벤치-불변 규율(TBox)**을 학습 → **안 본 벤치(τ²·SOP-Bench)로 ABox-swap 재학습0 전이.** 학습 = SOPBench(FC 성공 rollout) + TaskBench → **native OpenAI function-calling 궤적**. 헤드라인 = 주권 + 벤치-횡단 전이 + 결정론 compliance(raw accuracy 아님). **τ²는 전이 *타깃*(학습 아님).** 검증기/선택기 = 영구 결정론.

## 1. ★첫 행동 (순서)
1. **v4 학습 상태 확인**(`sft_runs/qwen7b_fc_tbox_v4`·GPU0서 ep0 진행중·29체크포인트). 수렴 후 최신 체크포인트 τ² 재테스트(아래 §3 패턴).
2. **L2 provenance 게이트 recall 수정**(§3 RED): substring 검사가 합법값 false-block → v4 with-L2 **0.0**. **타입별 정규화-매치**(D2)로 교체 + **추출기 recall 선실측**(리뷰어 #3 BLOCKING). 안 고치면 L2는 net-negative.
3. **다음 학습단계 = RL/DPO with L1/L2 활성**(사용자 발의 "L1,L2 전제하 학습"=RL의 literal형). SFT-v4 위에 rollout서 L1(날조차단)+L2(reject)+보상(compliant-pass)으로 복구정책 강화.

## 2. day-7 확정 결과 (전부 커밋·권위본)
- **★generator-gap 부검**: 7B(0.17) vs gpt-4.1(0.81) compliant-pass 갭 = **생성기 능력차** 확정. oracle@4(7B)=0.40<frontier pass^1 → 선택기로 못 닫음. 7B 실패본체 = **잘못된 DB-state(43%)+에이전트 붕괴(16%)+인자날조(78% tool-error)** = R-룰 위반(§부검).
- **★native-FC 변환기 파이프라인 구현·검증**(`scripts/distill/taskbench/fc_*`): P1a `fc_convert_taskbench`(DAG 레벨병렬 R6·verbatim 인자 R1)·P1b `fc_convert_sopbench`(FC 성공 rollout 1:1·sender role정규화·`<dom>_assistant.py` actions→A1·cleaning)·P2 `fc_build_sft`(per-traj 랜덤 전역alias=R1강제·QC·합성결과census·`--max_per_bench` 균형). **소스=t1c 아니라 FC 성공 rollout**(실인자/결과). py3.12 seka_env.
- **★τ² 전이 빠른-확인 + R1b 발견**: fctbox(plan-X 7B) τ² compliant-pass **0.10/0.05 < base 0.17**. 부검: ✅**R1 도구-이름 grounding 전이 작동**(τ² 실도구명 복사=별칭학습 일반화·thesis 코어) ❌**인자-값 날조**(`johndoe@example.com`·placeholder). provenance 실측 = base 날조 7% → fctbox **40%→60% 단조**(학습할수록 악화) = **파국적 망각**(base ask-user 능력 LoRA가 덮음).
- **★R1b 신설**(TB §10.5·`R1B_PROVENANCE_DESIGN_2026_06_14.md` v2): "인자값 provenance·무날조 — user/tool 출처만·부재시 ask-user/read-tool 획득·자가생성 금지". 3-레이어: **L1**(XGrammar decode-mask·원천차단) **L2**(provenance 검증기·검출) **L3**(학습된 복구순서 fetch→ask). ★프레이밍: L3=숫자·L1/L2=결정론 보장. provenance=compliance *필요조건*(충분 아님·인증게이트 직교).
- **★★값-randomization이 날조 제거 (핵심 성공·`fc_value_randomize.py`)**: user-제공 값을 포맷-보존 랜덤토큰으로 일관치환(user발화+call+출력) → memorize 불가·복사 강제(도구명 alias의 값버전). **v4(값랜덤+ask-user, sft_v4) τ² 날조 0-90%→0-5%**(grounded 19/20)·pass **0.10-0.15**(base 0.17 근접). = R1b 학습-측 작동 확정. **남은 gap(→base/frontier)은 날조 아닌 task-해결 능력.**
- **◐L2 게이트 = false-block (RED·미해결)**: `t2_gate_patch.py` env `T2_PROVENANCE=1`(인자값∈컨텍스트? deny·G1-G4 직교). offline 85호출 차단 확인. **그러나 v4 with-L2 live = 0.0** = substring 검사가 *합법값 false-block*(정규화/recall 문제·리뷰어 #3 확정). → **타입별 정규화-매치 + recall 실측 필요**(§1.2).

## 3. 인프라·실행 패턴 (재현용)
- **학습**: `lora_train_chat_toolcall.py`(messages+tools·assistant-only 마스킹·`--save-every N`=optimizer-step·체크포인트=`resume_adapter/`). **flash-attn 필수**(seka_env에 `flash_attn-2.8.3+cu12torch2.7` 휠 설치됨·sdpa는 ~0.3step/s 느림·FA ~1.5×). grad-accum 4=빠른 업데이트.
- **τ² serve+test**: `driver_v4*.sh` 패턴 = adapter snapshot→`vllm serve Qwen2.5-7B --enable-lora --tool-call-parser hermes`→`t2_run_gated --gate 1 --agent_model <name> --user_llm openrouter/openai/gpt-4.1`. L2 켜기=`export T2_PROVENANCE=1`.
- **provenance 날조율 측정**: 첫 인증-call(find_user_id/login) 인자값 ∈ 이전 user 발화? (greeting `?` 오염 주의 — ask-text율 아닌 인자 provenance).
- ⚠️**GPU**: woori 2×49GB·느림(7B). coworker = A100/H200 노드(`node_run_planx.sh`·preemption-safe·캐노니컬). GPU0/1 충돌금지.
- ⚠️**git**: 로컬(C:\workspace\ba-frft) + 원격워크스페이스 이중클론 = 충돌원. **진행로그/문서는 로컬 클론서만 편집·push**(원격은 pull). `--cmd` 내 **백틱 금지**(명령치환). 원격 dirty offload_*.sh = coworker 것·건드리지 마라. branch=facet-rft-2026.
- **coworker**: plan X에 병렬(요청서 v7=`COWORKER_REQUEST_TB_SCALE.md`). node_run_planx.sh에 "R1b 전 학습 보류" 경보. R1b 발견 동기화됨.

## 4. 다음 (우선순위)
1. **L2 recall 수정**(타입별 매치)→재테스트(net-positive 확인). 안 되면 값-randomization(SFT) 단독으로 진행·L2는 보장-only.
2. **v4 수렴 후 최신 체크포인트 τ² 재테스트** — pass가 base(0.17)→frontier(0.81) 쪽으로? (날조 아닌 task-능력 gap).
3. **RL/DPO with L1/L2**(= "전제하 학습" literal·다음 학습단계).
4. **L1 decode-mask 구현**(동적 guided_json·인자값 컨텍스트-후보 enum) = 결정론 보장. ⚠️vLLM OpenAI 스택서 위험·scoped 시도.
5. **전이 테스트**: SOP-Bench(직접 ABox-swap)·τ² 정식(벤치-횡단 R1b 전이). coworker 노드=캐노니컬 학습.

## 5. 논문 좌표
헤드라인 = plan X 벤치-횡단 전이 + R1b(no-fabrication 결정론 보장 + fetch/ask 복구학습) — AgentSpec/Prose2Policy가 안 한 칸. NLP 1급/COLM 노릴 공백이나 **전이 결과가 약하면 mid-tier.** 딥 리서치(트렌드 서베이) 백그라운드 진행(완료 시 `/workflows`). 부검=메모리 `project-cross-bench-transfer-plan`·생성기갭.
