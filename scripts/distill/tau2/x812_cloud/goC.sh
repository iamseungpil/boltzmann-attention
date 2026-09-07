#!/usr/bin/env bash
# 8141 후속 연쇄 (2026-09-07): 우선군 소진 → ①수리검증 rep1 합류(공유 큐·flock)
#   → ②base 잔여 30(결손 7 + 022 + 후순위 24). 사용자 지시: "쉬면 안된다".
bash /root/lane_rep1.sh 8141
bash /root/lane.sh A 8141 /root/q_base_rest.txt
