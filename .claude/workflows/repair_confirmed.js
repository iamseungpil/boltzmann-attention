export const meta = {
  name: 'repair_confirmed',
  description: 'Repair only the CONFIRMED our-layer defects from a forensic master: one agent per item (closed predicates, unit test, regression), then one battery verifier, then a single commit',
  whenToUse: 'per_task_forensics 마스터가 나온 직후. args = {items:[{id, title, brief, report, files}], battery:[test files]}',
  phases: [
    { title: 'Repair', detail: '항목당 에이전트 · 닫힌 술어 · 단위검정' },
    { title: 'Verify', detail: '배터리 전체 회귀 1회' },
  ],
}

// args 계약:
//   items   : [{id:'R2', title:'P5 합산 지시 복원', brief:'…무엇이 왜 틀렸나·어디를 고치나…',
//               report:'T7336_FAILURE_MASTER_2026_08_22.md', files:['a2/banking_knowledge.specific.json', …]}]
//   battery : ['test_a2_three_layer.py', 'test_flag_registry.py', …]   (없으면 아래 기본 배터리)
//   ⚠항목들은 같은 파일을 동시에 고치면 갈린다 — 같은 파일을 만지는 항목은 하나의 item 으로 묶어 넘겨라.
const A = args || {}
const ITEMS = A.items || []
const BASE = String.raw`C:\workspace\ba-frft`
const T2 = BASE + String.raw`\scripts\distill\tau2`
const BATTERY = A.battery || [
  'test_a2_three_layer.py', 'test_flag_registry.py', 'test_sg_docs_delivery.py', 'test_sg_fetch_iso.py',
  'test_sg_isofb.py', 'test_sg_src0_axis.py', 'test_claim_backed_write.py', 'test_comparator_read_first.py',
  'test_fab_fix_note.py', 'test_ground_warning_echo.py', 'test_search_rearm.py', 'test_require_doc_deliver.py',
]

const REPAIR_SCHEMA = {
  type: 'object',
  required: ['id', 'status', 'files_changed', 'tests_added', 'summary'],
  properties: {
    id: { type: 'string' },
    status: { type: 'string', enum: ['REPAIRED', 'NO_FIX_OPEN_JUDGMENT', 'BLOCKED'] },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests_added: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
    reason_if_not: { type: 'string' },
  },
}

function repairPrompt(it) {
  return `tau2 banking scaffold **수리 항목 ${it.id} — ${it.title}**. 응답·주석 한국어(기술용어 원어). 작업은 **전부 로컬** \`${T2}\` (\`py -3\`·\`PYTHONIOENCODING=utf-8\`·stdin 스크립트 첫 줄 \`# -*- coding: utf-8 -*-\`). SSH·리모트 금지. **git 커밋·push 금지**(마지막에 한 번만 한다 — 다른 항목 에이전트가 동시에 돌고 있다). 이 항목이 만지는 파일: ${(it.files || []).map(f => '`' + f + '`').join('·') || '(지정 없음 — 보고서에서 확정)'} — **그 밖의 파일은 건드리지 마라**.

## 근거
${it.brief}
정본 보고서: \`${BASE}\\reports\\facet_rft_2026\\${it.report || ''}\` 의 해당 절을 먼저 정독.

## 규율
- 닫힌 술어만(집합·원장·문자열 비교). 열린 판단(의도·유사도·"정답은 X")을 엔진에 넣지 말 것([[59]]·[[66]]) — 판단이 필요한 항목이면 **수리하지 말고** \`NO_FIX_OPEN_JUDGMENT\` 로 사유를 보고.
- 엔진에 도메인 리터럴 0(내용은 A2/A3 선언으로·두 층 specific/gate 바이트 동일·[[24]]).
- 거부/피드백 문면은 **무엇이 틀렸나 + 무엇을 하면 풀리나** 둘 다([[64]]).
- gold/tasks 파일 열람 금지([[23]]). 새 레버 신설 금지([[62]] — 기존 기구의 커버리지 구멍만 닫는다).
- 재현 단위검정(수리 전 결함 재현 양성대조 + 수리 후 + 부정통제) 추가. 기존 거동 보존(플래그 OFF = 바이트 동일 원칙이 있으면 따른다).

## 산출물
코드/선언 수정 + 검정 파일. StructuredOutput 으로 반환(파일 목록·검정 파일·요약).`
}

phase('Repair')
const done = await parallel(ITEMS.map(it => () =>
  agent(repairPrompt(it), { label: `repair:${it.id}`, phase: 'Repair', schema: REPAIR_SCHEMA })))
const results = done.filter(Boolean)
log(`수리 ${results.filter(r => r.status === 'REPAIRED').length}/${ITEMS.length} · 무수리(열린 판단) ${results.filter(r => r.status === 'NO_FIX_OPEN_JUDGMENT').length}`)

phase('Verify')
const verdict = await agent(
  `tau2 banking scaffold — 수리 묶음 **배터리 회귀 1회 + 단일 커밋**. 로컬 \`${T2}\` 에서 아래 검정을 전부 돌려라(\`PYTHONIOENCODING=utf-8 py -3 <file>\`). 실패가 있으면 **원인 파일을 지목**하고 고치지 말고 보고하라(수리 에이전트의 의도를 모른다). 전부 PASS 면 \`cd ${BASE} && git add -A scripts/distill/tau2 && git status --porcelain\` 로 변경 목록을 확인하고(⛔reports/·sim_results/·150MB 류는 add 금지 — scripts/distill/tau2 만), 한 커밋으로 \`git pull --rebase origin facet-rft-2026 && git push origin facet-rft-2026\`. 커밋 메시지는 영어로 수리 항목 id 나열.

수리 결과(입력):
\`\`\`json
${JSON.stringify(results, null, 1)}
\`\`\`

배터리: ${BATTERY.join(' · ')} + 수리 에이전트들이 추가한 검정 파일 전부.
반환: PASS/FAIL 표 + 커밋 sha(또는 실패 파일).`,
  { label: 'verify:battery', phase: 'Verify' })

return { repaired: results, verdict }
