# Utterance inventory (agent 2, 2026-09-08)
- 96 surfaces. Via t2_arbitrate.merge / Stack.speak: 0. Direct t2_dominance.merged_text: 2 (gate_patch:10850, :11295). admit()-dedup only: 31 (29 _ap_regen tags + fb :13112 + :13301). Bypass everything: 63.
- Engine-computed ~46 · A2 template+state ~28 · fixed prose ~22.
- 14 emitting modules never touch the arbiter: t2_eplan_patch t2_procedure t2_resolve t2_prekb_patch t2_scaffold_get t2_source t2_ledger t2_transfer_prereq t2_claim_block t2_unknown_bool t2_repeat_gov t2_compute t2_signature t2_speak.
- Stack.speak: route() never fills s["reqs"] so merge() always got [] (second dead path). t2_window.opened: 0 live calls. surface_bus: 0 live flushes.
- 048: gate_patch unified() :11751-11752 `role in ("user","tool")`. 049: eplan chain phrase = a2 settings.json:151 intent_chains[0].phrase (fixed prose) emitted t2_eplan_patch.py:467-480 (chain_reminder) reached via gate_patch:9394; [PROCEDURE] = t2_procedure.decide → proc_fb gate_patch:9668; _pchain at :9636 puts eplan ahead of proc → proc logged "suppressed by=eplan" (:9662).
- Exclusive chain _SRC8 at gate_patch:13040-13051 (18 sources), elif at :13060-13110, admit at :13112.
- _ap_regen :14032-14107 = generation-side exit (29 call sites :14603-15961).
