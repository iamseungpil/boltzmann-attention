> ⚠️ **SUPERSEDED by plan X (2026-06-14 · `CROSS_BENCH_TRANSFER_PLAN_2026_06_14.md`)** — SOPBench metric grounding(scaffold-era). 역사·근거 보존용(삭제 아님).

# internal_get_database 근거 조사 (2026-06-05) — 사전 논문/리더보드 사용 여부

> 질문: internal_get_database를 쓰는 이전 논문/리더보드 결과가 있나? 정당한가?
> 결론: **offered 도구 아님(코드+프롬프트). 단 released 리더보드의 react-mode 런 3종이 호출해 full DB(비밀 포함)를 받음 = unsanctioned leakage. 우리 fc-mode 런은 0회 호출 → 우리 결과와 무관.**

## 1. 코드: internal_get_database는 에이전트에 제공되지 않음
- `env/task.py:235-236`: `if not provide_database_getter: actions = [a for a in actions if a["name"] != "internal_get_database"]` — 호출가능 tool spec에서 제거.
- `env/task.py:145` (gather_dependency_instructions): 지시문에서도 제외.
- `provide_database_getter`는 **기본 False, 코드 어디서도 True로 설정 안 됨**(grep: 정의부만, 호출처 없음).
- ⇒ released SOPBench의 정식 도구셋에 internal_get_database **없음**.

## 2. 프롬프트(released llama3.1-70b): offered 안 됨
- 해당 런 프롬프트에 `internal_get_database` 문자열 **없음**, `internal_check_username_exist`는 **있음**. ⇒ offered 목록에 없음 = 호출 시 환각.

## 3. released 리더보드: react-mode만 호출, full DB 반환
- `output/bank/ast_*.json` 전수 스캔: internal_get_database 호출 **54회 = 전부 react 모드** (deepseek-r1 7, gemini-2.0-flash-thinking 4, llama3.1-70b 26+17). **fc/oracle/act-only = 0회.**
- 모든 호출이 `(True, {accounts:{...identification..., admin_password..., balance...}})` = **full DB(login 비밀 포함) 반환**.
- react 모드는 executor가 offered 목록으로 제한하지 않고 domain_system의 메서드를 실행 → 모델이 (환각으로) 도달 가능. fc 모드는 strict tool spec이라 불가.
- ⇒ internal_get_database = **react-mode executor 누수 경로**(벤치가 cred-absent 태스크에 감춘 바로 그 비밀을 반환). 의도된 도구 아님(bug report의 "에이전트 도구 아님" intent 수준 확증).

## 4. 우리 런(fc mode): 0회 호출, 무관
- `eval_t1c_loginfirst`/`eval_t1c_dggate` internal_get_database 호출 = **0**. fc 모드라 tool_names에 없음 → gate의 `if "internal_get_database" in tool_names` 구동 조건 False → 한 번도 안 불림.
- ⇒ **우리 cred-absent 통과는 internal_get_database가 아니라 no-login OR 경로**(internal_check_username_exist + goal; transfer 실측 trace 확인). 제 앞선 "047d가 internal_get_database로 통과·honest 32" **철회**.

## 5. 함의
- internal_get_database는 우리 결과(BOTH 33)에 **기여분 0** → 이 축의 honest-ceiling 우려는 우리 런엔 해당 없음.
- 남는 정당성 질문은 별개: cred-absent 태스크가 **no-login OR 경로**로 통과하는 것(bug report Part B: 의도는 login인데 그래프에 no-login 분기 존재). 이는 벤치 설계 비일관이며, 에이전트가 *offered된* internal_check_username_exist로 그래프를 충족하는 건 정당 플레이로 볼 수 있음.
- (참고) released react-mode 모델이 internal_get_database로 cred-absent/login-required 태스크를 통과했다면 그건 그들 리더보드의 누수 이슈 — 우리와 무관.

스크립트: `diag_idb_usage.py`(모드별 호출수), `diag_idb_offered.py`(프롬프트 offered 여부), `diag_idb_legitimacy.py`(우리 transfer 통과기전), 응답 DB-DATA 확인 인라인.
