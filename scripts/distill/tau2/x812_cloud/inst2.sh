set -e
cd ~/tau2-bench
python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install --no-cache-dir -q -e . litellm
./venv/bin/python -c "import tau2,litellm;print(\"RUNNER_OK litellm\",litellm.__version__)"
