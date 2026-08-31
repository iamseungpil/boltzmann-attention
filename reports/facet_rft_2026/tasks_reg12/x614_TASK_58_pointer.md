# x614 — TASK_58 포인터 (`t7391_reg12` · retail)

**정본 보고서**: `reports/facet_rft_2026/tasks__20260829/TASK_58.md`
(형제 9편과 같은 디렉터리 · `tasks_reg12/` 는 §74-b 훅이 `TASK_58.md` 신설을 막아
`^x\d+[_.]` 프로브 명명만 허용하므로 여기는 포인터만 둔다.)

- **채점축**: `reward_basis=["DB","NL_ASSERTION"]` · `breakdown={"DB":0.0,"NL_ASSERTION":1.0}`
  ⇒ **DB 한 축만** 죽었다. `action_checks` 는 진단용([[69]]).
- **변이표**(`F.mutating_tools("retail")`): gold 1 · done 1 · **missing 1 · wrongarg 1** ·
  blocked/extra/dup 0 · `sidecar='absent'`.
- **틀린 필드 2개(한 사슬)**: `new_item_ids[1]` `6017636844`→`2913673670`
  ⇒ 차액 $42.55→$452.07 ⇒ `payment_method_id` `gift_card_9368765`→`credit_card_7455506`.
- **결정 지점**: msg[15](기준 도착 **전** 후보 17→2 절단) · msg[27](기준 도착 **후** 재검토 0).
- **대조군**: `hist_gpt52_reg12_PASS` 가 같은 모델·같은 seed 626729 에서 **gold 인자 그대로** 통과
  ⇒ **회귀**다.
- **우리 층 지목(둘 다 선행 반복)**: ⑴`T2_PRESENT_NESTED`·`T2_CALC` 미수출
  (`run_t7391_retail.sh:48-60` · `t2_gate_patch.py:1100,1103`) — 형제 7편이 이미 지목.
  ⑵`t2_resolve.py:1276,1280` 의 리스트 접힘 + `{"status":"resolved"}` 미소비 —
  `tasks__20260829/TASK_9.md:288` 이 이미 지목(본 건이 **2번째 실사례**).
- **선행 원인 교체**: `A1_V3_PROBE_FORENSIC_2026_07_13.md` 의 t58 행은
  `[T2_L4] substituted 3815173328→3714494375`(에스프레소 슬롯·우리 층 자해)였다.
  t7391 에서 그 결함은 **재발하지 않았고**(에스프레소는 정답), 대신 같은 문서 §3 이
  *"t58 잠재"* 로 적어 둔 **(F2) 복합기준 무시** 축이 랩탑 슬롯에서 실현됐다.
