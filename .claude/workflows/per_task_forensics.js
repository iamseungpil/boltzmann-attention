export const meta = {
  name: 'per_task_forensics',
  description: 'Per-step trajectory forensics for every failed task of a tau2 run: one agent per task, adversarial refutation of our-layer attributions, master synthesis',
  whenToUse: '런이 끝나 sim_results 에 gz 가 내려온 직후. args = {tag, suffix, tasks:[{id, half, fails}], prior_reports:[...], reference:"t7328 6/40 ..."}',
  phases: [
    { title: 'Trace', detail: '실패 태스크마다 에이전트 하나 · 궤적 per-step 추적' },
    { title: 'Refute', detail: '우리-층 귀속만 적대적 반증' },
    { title: 'Synthesize', detail: '마스터 표 + 처방 큐' },
  ],
}

// ─────────────────────────────────────────────────────────────────────────────
// args 계약 (전부 필수 아님 — 없으면 아래 기본값):
//   tag      : 'bank_t7336'            런 태그 접두(파일명 bank_<tag>_<half>_<suffix>.results.json.gz)
//   suffix   : '20260821b'
//   tasks    : [{id:'task_073', half:'halfA', fails:'양 trial 실패 ★…'}]   실패 태스크(실패 sim 전부)
//   prior    : ['T7335_NT1_FORENSIC_HALFA_2026_08_21.md', …]             선행 포렌식 파일명(reports/facet_rft_2026/)
//   baseline : {halfA:'bank_t7328_halfA_20260819r', halfB:'bank_t7328_halfB_20260819r2'}  기준선 gz 접두
//   prev     : {halfA:'bank_t7335_halfA_20260821', halfB:'bank_t7335_halfB_20260821'}     직전 런 gz 접두
//   scores   : '성적·부호표 문장(마스터에 그대로 인용·수치는 gz 직독으로 검산한 것만)'
//   probes   : '이미 잰 격리 결과 요약(있으면)'
//   out_dir  : 't7336_tasks'            태스크 보고서 디렉터리(reports/facet_rft_2026/ 하위)
//   master   : 'T7336_FAILURE_MASTER_2026_08_22.md'
// ─────────────────────────────────────────────────────────────────────────────
const A = args || {}
const TAG = A.tag || 'bank_t7336'
const SUF = A.suffix || '20260821b'
const TASKS = A.tasks || []
const PRIOR = A.prior || []
const BASELINE = A.baseline || { halfA: 'bank_t7328_halfA_20260819r', halfB: 'bank_t7328_halfB_20260819r2' }
const PREV = A.prev || {}
const SCORES = A.scores || '(성적 문장 미제공 — 마스터는 입력 수치만 쓴다)'
const PROBES = A.probes || '(없음)'
const OUT_DIR = A.out_dir || 'tasks_' + SUF
const MASTER = A.master || 'FAILURE_MASTER_' + SUF + '.md'
const BASE = String.raw`C:\workspace\ba-frft`

if (!TASKS.length) { log('⛔ args.tasks 가 비어 있다 — 실패 태스크 목록을 넘겨라'); }

const TRACE_SCHEMA = {
  type: 'object',
  required: ['task', 'sims', 'our_layer_claims', 'summary'],
  properties: {
    task: { type: 'string' },
    sims: {
      type: 'array',
      items: {
        type: 'object',
        required: ['trial', 'reward', 'missing', 'wrongarg', 'dup', 'blocked',
                   'decision_point', 'cause_primary', 'evidence'],
        properties: {
          trial: { type: 'integer' },
          reward: { type: 'number' },
          missing: { type: 'array', items: { type: 'string' } },
          wrongarg: { type: 'array', items: { type: 'string' } },
          dup: { type: 'array', items: { type: 'string' } },
          blocked: { type: 'array', items: { type: 'string' } },
          decision_point: { type: 'string' },
          cause_primary: { type: 'string', enum: ['our_layer', 'model', 'env', 'user_sim'] },
          cause_secondary: { type: 'string' },
          lever_firing: { type: 'string' },
          evidence: { type: 'array', items: { type: 'string' } },
          prescription: { type: 'string' },
        },
      },
    },
    our_layer_claims: { type: 'array', items: { type: 'string' } },
    vs_prior: { type: 'string' },
    summary: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['task', 'claims'],
  properties: {
    task: { type: 'string' },
    claims: {
      type: 'array',
      items: {
        type: 'object',
        required: ['claim', 'verdict', 'reason'],
        properties: {
          claim: { type: 'string' },
          verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'UNPROVEN'] },
          reason: { type: 'string' },
          counter_evidence: { type: 'string' },
        },
      },
    },
  },
}

function tracePrompt(t) {
  const prevLine = PREV[t.half] ? `대조(직전 런·같은 계열): ${PREV[t.half]}.results.json.gz` : '대조(직전 런): (미제공)'
  return `tau2 banking 실험 **${TAG}** 의 ${t.id} 실패를 **궤적 per-step 정밀 추적**하고 원인을 확정하라. 응답·보고서 한국어(기술용어 원어).

## ⛔환경 규율
- **전부 로컬**이다. SSH·리모트 접속 **금지**(다른 에이전트와 동시 실행 중). 작업 디렉터리 = \`${BASE}\\scripts\\distill\\tau2\`, 파이썬 = \`py -3\` + \`PYTHONIOENCODING=utf-8\`(stdin 스크립트 첫 줄에 \`# -*- coding: utf-8 -*-\`).
- git **커밋·push 금지**(여러 에이전트 동시 실행 — index lock 다툼). 보고서 파일만 쓰고 끝낸다.

## 데이터 (전부 로컬 gz · ${BASE}\\reports\\facet_rft_2026\\sim_results\\)
\`\`\`
결과: ${TAG}_${t.half}_${SUF}.results.json.gz
로그: ${TAG}_${t.half}_${SUF}.log.gz   (줄 접두 [sim=${t.id}#...] 로 이 sim 만 grep)
${prevLine}
대조(기준선·sha 상이): ${BASELINE[t.half]}.results.json.gz
\`\`\`
읽기 예: \`import gzip,json; d=json.load(gzip.open(path,'rt',encoding='utf-8')); sims=[x for x in d['simulations'] if x['task_id']=='${t.id}']\`

## 이 태스크의 상태
${t.fails || '(미기재)'}

## 방법 (순서 고정)
1. **채점 축 먼저**: \`sim['reward_info']\` 를 직접 열어 DB-해시 축인지 ACTION 축인지 확인(\`reward_basis=ACTION\` 태스크는 \`action_checks\` 직독). 축을 틀리면 표가 거짓말을 한다(C583ⓖ).
2. **변이 집합은 정본으로만**: \`sys.path.insert(0,'.'); import t2_forensic as F; mut=F.mutating_tools(); m=F.mutation_diff(sim, mut)\` → keys: missing/wrongarg/extra/dup/blocked/matched. ⛔손 비교기 금지(C583ⓐ). WRONGARG 는 보낸 인자 ↔ gold 인자를 **필드별로** 대조.
3. **궤적을 처음부터 끝까지 순서대로 따라가라**(messages: role/content/tool_calls/tool 출력). 실패 변이마다 결정된 **정확한 지점**을 특정하고 직전 몇 턴을 **축자 인용**. 확인: 필요한 값이 그 시점 문맥에 **실재했는가** · 어떤 read 를 했고 안 했나 · write 를 왜 안 했나/틀리게 했나 · user-sim 이 무엇을 요구·오도했나.
4. **레버 발화 대조**(로그에서 이 sim 줄만): \`T2_SG_DOCS\`·\`T2_PIN_READ\`·\`T2_DEMANDED_STEP\`·\`T2_CLAIMPROV\`·\`T2_FOLLOWUP\`·\`T2_SEARCH_AGENT\`(침묵 포함)·\`FAB_STRIP\`·\`T2_ARG_PRODUCERS\`·READ-FIRST·\`T2_REQUIRE_DOC_DELIVER\`·\`T2_SEARCH_REARM\`. 각각 **발화했는데 무시 / 미발화 / 오발화** 를 가려라. 직전 런 이후 들어간 수리·레버가 **이 궤적에 개입했는지, 개입하고도 왜 못 샀는지**가 핵심 질문이다.
5. **선행 판정과 대조**: \`${BASE}\\reports\\facet_rft_2026\\\` 의 ${PRIOR.length ? PRIOR.map(x => '`' + x + '`').join('·') : '(선행 보고서 미제공)'} 중 이 태스크를 다룬 절을 읽고 **같은 원인인가 달라졌는가** 명시.
6. trial 이 둘이면 **둘 다** 추적. 한쪽만 실패면 **분기점**(어느 턴에서 갈렸나) 특정.

## 규율
- 원인 귀속 4주체(**our_layer / model / env / user_sim**) · 근거는 궤적 축자 인용만([[08]]).
- \`our_layer\` 귀속은 **코드 경로(파일:줄) 또는 선언 키를 반드시 지목**. 못 대면 \`model\` 또는 UNPROVEN.
- gold(\`reward_info\`)는 진단용으로만([[23]]). 수리 실행·코드 수정 금지(제안까지).

## 산출물
1. \`${BASE}\\reports\\facet_rft_2026\\${OUT_DIR}\\TASK_${t.id.replace('task_', '')}.md\` — 채점축 → 변이표(trial 별) → **step-by-step 결정 지점 추적(축자 인용)** → 레버 발화표 → 선행 대조 → 원인 확정 → 처방 후보. (디렉터리 없으면 생성.)
2. StructuredOutput(스키마 준수). \`our_layer_claims\` = **코드 경로를 지목한 우리-층 주장만**(없으면 빈 배열).`
}

function refutePrompt(task, claims) {
  return `너는 **반증자**다. tau2 banking ${TAG} ${task} 포렌식이 내놓은 **우리-층(scaffold) 귀속 주장**을 궤적과 코드로 **반증하려 시도**하라. 확증이 아니라 반증이 임무다.

## 검증 대상 주장
${claims.map((c, i) => `${i + 1}. ${c}`).join('\n')}

## 규율
- **전부 로컬**(SSH 금지·git 커밋 금지). 데이터 = \`${BASE}\\reports\\facet_rft_2026\\sim_results\\${TAG}_*.{results.json,log}.gz\`, 보고서 = \`${BASE}\\reports\\facet_rft_2026\\${OUT_DIR}\\TASK_${task.replace('task_', '')}.md\`, 코드 = \`${BASE}\\scripts\\distill\\tau2\\\`.
- 각 주장마다: ⑴그 코드 경로가 **이 sim 에서 실제로 실행됐다는 증거**(발화 마크·문면)가 있나? ⑵같은 결과가 **우리 층 없이도**(모델 결손만으로) 설명되나? ⑶지목한 파일:줄을 **직접 읽어** 그 분기가 정말 그렇게 도는지 확인했나? ⑷반대 사례(다른 sim 에서 같은 레버가 발화했는데 실패하지 않았다 등)가 있나?
- 이 프로젝트는 같은 함정에 두 번 빠졌다(C583ⓐ 손 비교기 · C584 약한 인터페이스) — *"수치가 나왔다"* 로 넘어갈 뻔했고 **정본으로 갈아끼우자 뒤집혔다**. 근거가 관측이 아니라 그럴듯함이면 UNPROVEN.
- 판정: **CONFIRMED**(실행 증거 + 코드 확인 + 대안 설명 배제) / **REFUTED**(반대 증거) / **UNPROVEN**. 불확실하면 UNPROVEN 쪽.

StructuredOutput 으로 반환하라.`
}

phase('Trace')
const traced = await pipeline(
  TASKS,
  (t) => agent(tracePrompt(t), { label: `trace:${t.id}`, phase: 'Trace', schema: TRACE_SCHEMA }),
  (res, t) => {
    if (!res || !(res.our_layer_claims || []).length) return { trace: res, verdicts: null }
    return agent(refutePrompt(t.id, res.our_layer_claims),
                 { label: `refute:${t.id}`, phase: 'Refute', schema: VERDICT_SCHEMA })
      .then((v) => ({ trace: res, verdicts: v }))
  },
)

const rows = traced.filter(Boolean)
log(`추적 완료 ${rows.length}/${TASKS.length} · 우리-층 주장 있는 태스크 ${rows.filter(r => r.verdicts).length}`)

phase('Synthesize')
const bundle = JSON.stringify(rows.map(r => ({
  task: r.trace && r.trace.task,
  summary: r.trace && r.trace.summary,
  vs_prior: r.trace && r.trace.vs_prior,
  sims: r.trace && r.trace.sims,
  our_layer_claims: (r.trace && r.trace.our_layer_claims) || [],
  verdicts: (r.verdicts && r.verdicts.claims) || [],
})), null, 1)

const master = await agent(
  `tau2 banking **${TAG} 전수 실패 포렌식 종합**. 아래는 실패 태스크를 태스크당 한 에이전트가 궤적 per-step 추적한 구조화 결과와, 우리-층 귀속에 대한 반증자 판정이다. **마스터 보고서 한 편**으로 종합하라. 한국어.

## 입력 (구조화 결과)
\`\`\`json
${bundle}
\`\`\`

## 성적·부호표(gz 직독 검산본 — 그대로 인용)
${SCORES}

## 이미 잰 격리 결과(인용 가능)
${PROBES}

## 쓸 것 (\`${BASE}\\reports\\facet_rft_2026\\${MASTER}\`)
1. 성적 표(위 문장 그대로·없는 수치 만들지 말 것).
2. **원인 축별 군집표**: 실패 sim 을 원인 축으로 묶어라(태스크·sim 수·대표 축자 근거·귀속 주체). 축은 데이터에서 나오게.
3. **직전 런 이후 들어간 수리·레버의 실측 성적표**: 각각 **발화했나 / 발화하고도 못 샀나 / 발화 기회 자체가 없었나**. 死배선과 무효과를 구분([[55]]).
4. **회귀 전용 절**: 직전 런 대비 내려간 태스크마다 **무엇을 팔았나**([[70]] 의무)·확정 못 하면 "미상".
5. **반증자 판정 반영**: CONFIRMED 만 우리-층 결손으로 승격, UNPROVEN·REFUTED 는 등급 그대로 표에 남긴다.
6. **처방 큐**: 무료 수리 가능 / 격리 프로브 선행 필요 / 레버 없음(경계) 3분할 — 표적 태스크·기대 상한·[[62]] 순서. 새 레버는 **격리로 잰 뒤** 조건 명시.
7. **이 종합이 못 사는 것**(정직 절).

규율: 근거 없는 승격 금지·수치는 입력에 있는 것만. git 커밋·push 금지(파일만). SSH 금지.
반환은 마스터 표의 핵심 5줄 요약만.`,
  { label: 'synthesize:master', phase: 'Synthesize' })

return { traced: rows.length, master }
