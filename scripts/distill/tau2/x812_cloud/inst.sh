set -e
python3 -m venv ~/venv_vllm
~/venv_vllm/bin/pip install -q --upgrade pip
~/venv_vllm/bin/pip install --no-cache-dir "vllm==0.27.1"
~/venv_vllm/bin/python -c "import vllm,torch;print(\"OK vllm\",vllm.__version__,\"torch\",torch.__version__,torch.version.cuda)"
