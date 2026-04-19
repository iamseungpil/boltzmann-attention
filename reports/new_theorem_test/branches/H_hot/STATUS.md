# H-HOT — Higher-order ablation Status

**Status**: **dormant** (2026-04-19). Depended on F14 pilot positive — closed by F13b +3.85pp. **Do not execute.** Most speculative branch; dormancy is low-regret. Q5 "higher-order abstraction = 상위 attention 이 하위 관찰" remains as §2.5.2 theoretical coverage only; F13's staged rotation (early K / late Q) is a weaker-but-empirical version of the hierarchical-observation intuition.
**Primary spec**: memory `user_5question_synthesis_f14_metafocus_2026_04_19.md` §5 H-HOT
**Concept**: Q5 "상위 attention 이 하위 관찰". Test if explicit meta-layer is qualitatively different from pure transformer depth.
**3 variants**:
  1. Identical stacked attention (control, pure depth)
  2. Ontology-typed routing (axis-labeled Q/K)
  3. Step-state aware routing (decoding history-conditioned)
**Prediction**: `(ii), (iii) ≥ (i) + 2pp`. Else "transformer depth already implicitly implements HOT".
**Gate**: F14 pilot positive (otherwise no empirical anchor for "depth vs meta-layer" comparison).
**Decision**:
  - `(ii)/(iii) > (i) by ≥+2pp` → qualitatively distinct meta-attention verified; paper §5.X "HOT-inspired"
  - `(i) ≈ (ii) ≈ (iii)` → depth suffices; H-HOT dormant; Q5 answered "transformer already does this"
**Risks**:
  - **HOT philosophical citation trap** (Dennett/Block): use "HOT-inspired" weakly, do not claim consciousness mechanism.
  - **Ambition vs empirics mismatch**: this is the most speculative branch; fallback to "depth suffices" is fine.
**Cost**: 12–18 GPU-hr (3 variants × train + eval).
**Log**: (empty)
