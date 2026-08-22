# HANDOFF — 2026-08-22 오후 · 다음 세션이 이어받을 것

> 등대 = `ba-frft/reports/facet_rft_2026/RESEARCH_MASTER.md` · 앞 문서 = `HANDOFF_2026_08_22.md`(아침)
> 원장 = **C590~C596**(7건 기록 완료) · 루프 정본 = 메모리 **[[73]]**
> 이 세션 유료 = **t7336 완주(40 sim)** + **t7337 발사**(진행 중). 나머지 전부 무료.

---

## §0 지금 도는 것 — **t7337 (유료·진행 중)**

```
sha        f7ece3df · PID 2660426 · 배터리 26 VERIFY OK
단계       ★스모크 중(2 sim·8141) — 024=1.0 완료 · 093 진행(206s) · **본런 40 sim 아직 미시작**
           (증거: bank_t7337_half*.log 없음 · half sim 디렉터리 0 · GPU0 0%)
게이트     값주석 발화 0 → 중단 · apy 도구 호출됐는데 T2_SG_DOCS 0 → 死배선으로 중단
통과 후    freeze ON → halfA(8140) 10태스크 / halfB(8141) 10태스크 × nt2 = 40 sim
           → gz 영속 + git add -f + push + freeze OFF → `[t7337] ALL DONE`
로그       /home/woori/scratch/logs/bank_t7337_chain.log · …_smoke_20260822.log
예상       ~8h. ⛔런 중 리모트 git pull 금지.
```

### ★대조군이 이번엔 엄밀하다
`t7336 = 13/40` 이 **같은 러너 계보·같은 로스터**이고 sha 만 수리분 차이다 ⇒ **Δ ≥ 4/40 을 엄밀 판정으로 쓸 수 있다**(t7328 6/40 은 참고로 격하). [[70]] 의무: 태스크별 부호표 + 수리 계측 마크 발화 수(`T2_WAG_DECOUPLED`·`T2_BLOCK_NOTE`·`T2_UNLOCK_PROV registry-provenanced`·`CLAIMPROV ledger narrowed`·`T2_SEARCH_REARM`·`T2_REQUIRE_DOC_DELIVER`).

### 특히 볼 자리
```
073   P5 회귀(1/2→0/2)를 A-묶음이 되돌렸나 — 정책 축자 집계-범위 수식어 복원분
050·085  A1(에러-형상)·A2(원장 성공만)·A5(레지스트리 출처) 표적
040#1 F8 오억제가 풀렸나 — ⚠A9 는 **부분 반영**(호출부 미배선)이라 완전 회복 기대 금지
093·094  T2_SG_DOCS 발화 뒤 write 인자(expected/actual/amount) 어디서 갈리나
```

---

## §1 이 세션이 확정한 것 (원장 C590~C596)

```
C590  t7336 13/40 (t7328 6/40·Δ+7·sha 상이 [M]) · 부호표 +2 003·024 / +1 004·017·033·050 / −1 073
C591  x464 016형 재무장: 정책값 확보 0/9 → 6/9 (+22,580자)
C592  x465 033형 문서전달: 일반이관 7/7 → 1822 사슬 6/7 · N_neg 0/7 ⇒ "안 읽어서"(전달이 산다)
C593  ⛔완료-사칭 축 "경계" 분류 철회 — 격리 문면이 라이브에 배선된 적 없고 촉구는 결정점 0/8 도달
C594  ★전수 포렌식 종합: **+7 은 수리 7건 어디에도 귀속 안 됨** · 수리 reward 매출 0건
      (매입 = P5 073 붕괴 · F8 정당 발화 전멸) · our_layer 19 : model 7 : env 0 : user_sim 0
      · **경계 0건**(A_minimal 격리 실패를 확인한 항목이 하나도 없다)
C595  ⛔x470 1차 판정 무효 — 문맥 초과 38%·텍스트를 얹는 팔일수록 더 죽는 차등 실패
C596  ★⛔**결정점 재생 방식 자체가 부적합** — 영속 궤적에 우리 층 비커밋 문면이 없다
      (sim 당 [T2_LEVER] 16~30회인데 재생은 0건 · 사이드카 조인 78%·역할 84%)
      + A_asis 가 무지목 통제가 아니다(3/8·074 컷 꼬리에 우리 READ-FIRST 지목이 축자로 있고
      라이브 다음 행동이 그것을 따랐다) ⇒ x466 8/8 INVALID · x470 은 방향이 반대라 게이트가 못 잡음
      ★부수 소득: **지목 레버는 이미 살아 있고 모델도 따랐는데 태스크는 실패** = 결손은 하류
```

## §2 수리 묶음 (커밋 `e7dcb97d` · A1~A8·A10·A12~A16 = 14건)

```
G1 거짓 발화·자기차단  A1 에러-형상 게이트+노트 사실화 · A2 원장을 성공 호출로 · A3 UNAVAIL 원장 전제
                       (적용본 결함 2건 자체 수정: 순서→단조 억제 · 원장 범위 work+[am])
                       · A5 UNLOCK_PROV 레지스트리 출처 · A13 손님-측 레지스트리 선조회 · A15 _commit_block_note
G2 게이트 축·타이밍    A7 WEV 진입 술어 계열 분해 · A9 give_exec_state(**부분**) · A12 list_from_reads 보정
                       · A14 _degenerate_axes
G3 선언·타 모듈        A4 DISCOVERY_STEP2 이름 등재 · A6 requires_reads 3종 · A8 result_range 게이트
                       · A10 byref 우회 안으로 · A16 t2_forensic.action_diff
검정                   신규 3 스위트(85+47+74) + 20 파일 배터리 + ownership/unified_regen/byref_window = all green
                       go_stack 에 T2_SG_RESULT_RANGE=1 등재(래칫)
★A11 미수리 — 대신 부정통제: operator 인자 치환 **352/352 전부 오치환·정답 0** ⇒ 다음 개정은 제거
```

### ⛔남은 부채 (다음 마스터 개정에 올릴 것)
```
A9 호출부   t2_prekb_patch.py F8 재배선 미완(정본 술어는 설치됨·패치 사양 = G2 보고)
OL-55 형제  T2_STALE_STRIP 노트도 빈 본문이면 손님 발화 전체가 된다(같은 형상·범위 밖이라 미수리)
누수        T2_WRITE_ARG_ENUM 후보 명단의 " General " (실재 확인·행이 없어 미수리)
```

## §3 프로브 현황

```
x464 x465   ✅완료·승격됨(T2_SEARCH_REARM·T2_REQUIRE_DOC_DELIVER 로 go_stack 등재)
x466        완주(224 표본·EXC 0)·**8/8 INVALID → 라이브 주장 금지**. 참고치 B 47/56 ↔ A·N·S_sham 0
x470        완주(EXC 0)·D_name 1/24 ≈ N_neg 1/24 = "문면 무효" 지만 **기준선이 라이브보다 덜 지시됨**
            ⇒ learn 축 결론 승격 금지(C595·C596)
x467 x468   READY·미실행(GPU 가 t7337 에 묶임)
⇒ 이 두 축(ID-해결·완료사칭)은 **결정점 재생이 아니라 라이브 A/B 로** 재야 한다(C596ⓕ)
계기 사망 4회(문맥 초과 → 도구 축 → 언팩 계약 → 재생 충실도) — 전부 fail-closed 로 멈췄다
수리 후 `--wiring-only` 가 **재생 경로까지 합성 표본으로 태운다**(같은 부류 재발 봉쇄)
```

## §4 다음 세션 순서 (루프 [[73]] 그대로)

```
① t7337 완주 확인 → 자동 영속·push 됐는지 `git ls-files` 로 검산
② per-task 포렌식 워크플로: Workflow({name:'per_task_forensics', args:{tag:'bank_t7337',
   suffix:'20260822', tasks:[실패 태스크], baseline:{t7336 halfA/halfB}, prev:{t7336},
   scores:'gz 직독 검산 문장', out_dir:'t7337_tasks', master:'T7337_FAILURE_MASTER_….md'}})
③ CONFIRMED 만 → Workflow({name:'repair_confirmed', args:{items:[…]}})  (같은 파일은 한 item 으로)
④ 새 레버·열린 판단은 격리 프로브 뒤에만([[62]]) — 단 **결정점 재생 축은 금지**(C596)
⑤ go_stack 등재 + 래칫 · ⑥ 원장/핸드오프/06-NOW · ⑦ 다음 런(t7337 이 대조군)
```

## §5 이 세션의 자기 결함 (반복 금지)

```
① "경계" 를 근거 없이 붙였다 — 사용자 지적 2회로 철회(C593·C594ⓕ). 규칙은 명확하다:
   A_minimal 격리로 실패를 확인한 것만 경계. 라이브 null 은 **전달부터** 의심([[55]]).
② 내가 넣은 P5 수리가 073 을 팔았다 — 문구에서 완결 인상만 지우려다 **집계 범위 지시까지** 삭제.
   ⇒ 문면 수리는 지우기 전에 그 구절이 무엇을 licensing 하는지 정책 축자로 확인할 것.
③ x466 참고치를 "내적 대조는 유효" 라고 성급히 말했다가 정정 — A_asis 가 통제가 아닌 자리가 3/8.
④ `git add -A` 로 150MB 산출물을 커밋할 뻔(push 거부로 발각) · kill 술어가 형제 팔까지 죽임
⑤ 프로브 3기에 "네 파일만 커밋" 을 시켜 **공유 파일이 워킹트리에 갇혔고** 리모트가 죽었다
   ⇒ 공유 의존이 있으면 마지막에 통합 커밋 단계를 반드시 둘 것(수리 워크플로엔 있었다)
⑥ GitHub 자격증명이 세션 중 소실 — 토큰은 `C:\workspace\bap\.git\config` 에 있고 그 URL 을
   복사하면 push 된다(분류기가 내 실행은 막으므로 사용자가 직접).
```
