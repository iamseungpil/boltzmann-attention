set -u
source ~/.openrouter_key; [ -f ~/.openai_key ] && source ~/.openai_key
export HF_HOME=/workspace/.hf_home
export T2_PROBE_URL=http://localhost:8143/v1/chat/completions
export T2_PROBE_MODEL=Qwen/Qwen3.8-27B-FP8
cd ~/t2/tau2
PYTHONPATH=~/t2/tau2 ~/tau2-bench/venv/bin/python x166_our_text_narrowing.py 10
echo X166_DONE
