set -u
source ~/.openrouter_key
[ -f ~/.openai_key ] && source ~/.openai_key
export PYTHONPATH="src:$HOME/t2/tau2"
export T2_MAX_MODEL_LEN=131072
export HF_HOME=/workspace/.hf_home
cd ~/tau2-bench
rm -rf data/simulations/x812_g4
~/tau2-bench/venv/bin/python ~/t2/tau2/t2_run_gated.py --gate 0 --domain banking_knowledge   --retrieval_config alltools --agent_model Qwen/Qwen3.8-27B-FP8   --agent_base http://localhost:8141/v1   --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low   --task_ids task_001,task_002,task_005 --num_trials 4 --max_concurrency 4 --max_steps 200   --save_to x812_g4
echo G4_DRIVER_DONE
