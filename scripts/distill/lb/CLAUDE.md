# 디렉토리 메모리 — `scripts/distill/lb/` (브랜치 `lb`)

> 세션 시작 = 이 파일 하나. 정본은 `docs/RESEARCH_BASE.md`(기전 F1~F6 · LB1~LB7 · 판정 규칙 · 도는 실험 · 확정 결함).
> 구 코드베이스 `scripts/distill/tau2/` 는 **읽기만** 한다. 고치지 않는다. 도는 레인(rep1·rep2·base)은 리모트의 구 트리다.

## 이 코드베이스가 무엇인가
- LB1~LB7 = 파일 7개 + 조정기 1개(`lb_coordinator.py`) + 배관 4개(`lb_runtime` `lb_a2` `lb_run` `lb_report`). 파일 수를 늘리지 않는다 — 판정 유형이 늘면 그 LB 파일의 `kind` 가 는다.
- 선언은 `a2/<domain>.lb.json` 하나. 태스크별·케이스별 예외는 **전부 데이터**로 간다. 코드에 도메인 이름·수치·문장·정규식을 넣지 않는다(`tests/test_lb.py` 가 강제).
- 플래그는 `T2_LB1~7` 뿐(기본 켬). `LB_SIDECAR`·`LB_DOCS_DIR` 는 경로(하네스).
- 출구는 `lb_coordinator.say()` 하나. 같은 표적엔 명령 하나, 사실은 합집합, 고정 산문(E5)은 계산 결과(E1·E2)에 지지 못한다. 충돌은 `[LB_CONFLICT]` 로 남기고 `lb_report.py` 로 접는다.

## 일하는 규칙 (사용자 지시 · 2026-09-08)
1. **찾기 전에 짓지 마라.** 첫 수는 `ls`·`grep`. 구 코드의 계약은 `docs/UTTERANCE_INVENTORY_2026_09_08.md`·`docs/flags_inventory.tsv` 에 있다.
2. **도는 실험에 쓰기 전 승인.** 리모트 트리·큐·엔진·워커를 건드리는 것은 전부 승인 뒤. 읽기는 자유.
3. **원인 진술은 4칸**(주장+양화 / 축자인용+파일:줄 / 반증조건 / 선행확인 경로). 못 채우면 "모른다".
4. **집계에서 결론 직행 금지.** per-step 궤적을 읽은 뒤에만. 도구 출력이 전부 같으면 도구가 죽은 것.
5. **성적 주장은 reward 짝 A/B 뿐.** 이 코드베이스는 리모트 e2e 0회다 — 효과를 말하지 않는다.
6. 새 판정을 넣을 때 먼저 묻는다: 어느 LB 의 규칙 하나의 인스턴스인가? 데이터면 `lb.json`, 새 규칙이면 그 LB 의 `kind` 하나 + 자기검정.
7. 응답·보고는 한글(축자 인용·기술 용어 원어·commit 영어). 커밋은 이 브랜치에서, push 는 지시가 있을 때만.

## 지금 상태
- 커밋 `0903ef12`(엔진 7 + 조정기) · `09205d20`(서브콜 레버 이전) · 배터리 23/23 · 2,755줄 · py 12개.
- 확정 결함 처리: 048 = 절차는 손님 발화만으로 열린다 · 049 = `intent_chains` 를 `procedures` 로 접어 코드 하나·선언 하나.
- 부채: `account_opening`·`prescription:*` 절차의 `_quote_order` 가 AUTHORED(정책 축자 없음). 미이전 = `resolve_write/action_operator/recommendation`·`catalog_arg_docs`·`ledger_metrics` 다단 프롬프트.
- 다음 수순(승인 필요): 리모트에 `repo_lb` 트리 → base 4/4 태스크(048·049 포함) 짝 A/B.

## 문서 지도
| 물음 | 파일 |
|---|---|
| 기전·LB 표·판정 규칙·도는 실험·확정 결함·선행연구 | `docs/RESEARCH_BASE.md` |
| 구 레버 185개의 판정(VALID 0 · HARMFUL 22 · DARK 38) | `docs/LEVER_ROSTER_2026_08_19.md` |
| 구 코드가 읽던 플래그 358개 → 어느 함수·어느 L군 | `docs/flags_inventory.tsv` |
| 구 코드의 발화 지점 96곳(중재기 경유 0) | `docs/UTTERANCE_INVENTORY_2026_09_08.md` |
| 새 구조·실행·이전 표 | `README.md` |
