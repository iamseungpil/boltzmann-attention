#!/bin/bash
# eref_gpu0_batch.sh — 바인딩-공격 오염 축 배치 (GPU0:8140 기존 32B·client-only·밤샘·무료)
#
# 목적: E-REF 오염-사다리(GPU1·eref_scale_ladder.sh)가 못 하는 "바인딩 표적 자체를 애매하게 하는"
#   강한 축을 32B에 실측 → 오염-축 그리드 완성(논문 figure). 약한 축(C 부하·A distractor)은
#   바인딩 유일성을 안 건드려 32B서 1.00. 여기선 바인딩을 부수는 두 축:
#     ① B(near-miss 같은-차원): 진짜 anchor(delivered·소유중)와 같은 상품·인접 key 값 "반품 주문"
#        디코이 K개 → "same size as mine"의 anchor 식별을 exact-match서 의미판별로. level=K=1,2,4.
#     ② P(paraphrase 약화): utterance서 필드명 토큰 제거("same size"→"matching my current one").
#        이 카탈로그 단일-변주-차원 상품=0 → anchor를 key로 투영해 field 유일추론 공정성 확보. level=1.
#   + ③ fexec 실행-채점 full(--tasks all·안정 stats).
#
# gold 전부 결정론·불변(소음은 바인딩 표적만 애매하게·gold-유일성 유지). gpt-4.1 user-sim = 0(무료).
# ★client-only: 8140 서버(32B·4지선다 프로브와 공유·vLLM 다중화) 서빙/킬 안 함 — 살아있는지 확인만.
# workers=3(4지선다와 GPU 공유 배려). setsid·로그·[[30]] PipeTimeout 주의.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
GZ=$REPO/reports/facet_rft_2026/sim_results/comp_retail_t4.results.json.gz
OUT=$REPO/reports/facet_rft_2026/sim_results
P=/home/woori/venvs/seka_env/bin/python
BASE=http://localhost:8140/v1
MODEL=Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
LOG=/home/woori/scratch/eref_gpu0_batch.log
exec > $LOG 2>&1
set -x
date
cd $REPO && git pull --ff-only -q

# 기존 32B 서버(8140) 살아있는지 확인만 (서빙 안 함·4지선다와 공유)
curl -s --max-time 8 localhost:8140/v1/models | grep -q "$MODEL" \
  || { echo "SERVER_8140_DOWN — 4지선다 서버 확인 필요·중단"; touch /home/woori/scratch/EREF_GPU0_ABORT; exit 1; }
echo "=== 8140 32B alive (client-only) ==="; date

# ① 축 B near-miss (K=1,2,4)
echo "===== [1/3] axis B near-miss B:1,2,4 ====="; date
$P -u $T2/eref_probe.py --gz $GZ --n 36 --v2 B:1,2,4 --base $BASE --model "$MODEL" \
   --workers 3 --out $OUT/eref_gpu0_nearmissB.jsonl 2>&1 | tail -24
echo "===== [1/3] DONE nearmissB ====="; date

# ② 축 P paraphrase (level 1·anchor-투영)
echo "===== [2/3] axis P paraphrase P:1 ====="; date
$P -u $T2/eref_probe.py --gz $GZ --n 36 --v2 P:1 --base $BASE --model "$MODEL" \
   --workers 3 --out $OUT/eref_gpu0_paraphraseP.jsonl 2>&1 | tail -18
echo "===== [2/3] DONE paraphraseP ====="; date

# ③ fexec 실행-채점 full (변형-선택 클래스 전수·안정 stats)
echo "===== [3/3] fexec exec-probe --tasks all ====="; date
$P -u $T2/fexec_exec_probe.py --gz $GZ --tasks all --base $BASE --model "$MODEL" \
   --workers 3 --out $OUT/eref_gpu0_fexec_all.jsonl 2>&1 | tail -40
echo "===== [3/3] DONE fexec_all ====="; date

# 결과 영속 (scratch=gitignored 아님·이건 sim_results=커밋대상)
cd $REPO
git add -f $OUT/eref_gpu0_nearmissB.jsonl $OUT/eref_gpu0_paraphraseP.jsonl $OUT/eref_gpu0_fexec_all.jsonl 2>/dev/null
git commit -q -m "persist: E-REF GPU0 binding-attack axes (near-miss B·paraphrase P·fexec full·밤샘·무료)" 2>/dev/null
git pull --rebase -q 2>/dev/null; git push -q origin facet-rft-2026 && echo PERSISTED

touch $OUT/../EREF_GPU0_DONE; touch /home/woori/scratch/EREF_GPU0_DONE
echo "===== EREF_GPU0_ALLDONE ====="; date
