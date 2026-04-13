# Node Connection and GPU Usage Policy (2026-04-05)

This file documents how to connect to the current remote experiment node used by
the `boltzmann-attention` project, how to inspect GPU availability, and how to
launch work without polluting the node state.

The current active remote target is:

- node alias: `tops-caiman`
- transport: AzureML websocket SSH
- last verified on: `2026-04-05`
- last confirmed host response: `node-0`

## 1. Core Rule

Do not treat the remote node like a disposable shell.

Before launching any new experiment:

1. confirm the node is reachable
2. confirm which GPUs are occupied
3. confirm whether an old run is still producing logs/results
4. only then assign a free GPU

Do not launch a new run with `custom` benchmark metadata if the run belongs to a
named experimental track. Use a hypothesis-aligned preset whenever possible.

## 2. Connection Method

The node is reached through AzureML's websocket SSH connector, not through a
plain hostname in `~/.ssh/config`.

Required local paths:

- Azure Python:
  - `/opt/az/bin/python3`
- AzureML SSH connector:
  - `/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py`
- SSH key:
  - `/home/v-seungplee/.ssh/id_rsa`

Current websocket URL:

- `wss://ssh-2etszrmvdrq4cwqdql4al50f30b0d16gxsfm589ld0y9dm0l6bc.westus2.nodes.azureml.ms`
- (previous, expired: `wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms`)

Canonical connection command:

```bash
AZ_PYTHON=/opt/az/bin/python3
CONNECTOR=/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py
SSH_KEY=/home/v-seungplee/.ssh/id_rsa
URL_TOPS='wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms'

ssh -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_TOPS" \
    -i "$SSH_KEY" \
    azureuser@placeholder 'hostname; whoami; uptime'
```

Expected success signal:

- `hostname` returns the AML compute host such as `node-0`

## 3. Remote Project Layout

The current remote working directory used by the reproducible Qwen runs is:

- `/scratch/boltzmann-attention-v3-repro`

Important subpaths:

- scripts:
  - `/scratch/boltzmann-attention-v3-repro/scripts`
- logs:
  - `/scratch/boltzmann-attention-v3-repro/logs`
- results:
  - `/scratch/boltzmann-attention-v3-repro/results/v3`
- virtualenv:
  - `/scratch/boltzmann-attention-v3-repro/.venv/bin/python`

The local develop-side launcher and harness currently mirrored to the remote
node are:

- [scripts/exp4_2_v3_full_quant_ppl.py](/home/v-seungplee/boltzmann-attention-develop/scripts/exp4_2_v3_full_quant_ppl.py)
- [scripts/launch_remote_qwen_20260404_gap.sh](/home/v-seungplee/boltzmann-attention-develop/scripts/launch_remote_qwen_20260404_gap.sh)

## 4. GPU Inspection

Always inspect GPU state before choosing a device.

Basic status command:

```bash
ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_TOPS" \
    -i "$SSH_KEY" \
    azureuser@placeholder \
    'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader;
     echo __PROCS__;
     nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

Interpretation:

1. a GPU with `0 MiB` and no listed compute app is usually free
2. a GPU with tens of GiB allocated is occupied by a live model run
3. if memory is nonzero but the process table looks stale, verify with `ps`

Process check:

```bash
ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_TOPS" \
    -i "$SSH_KEY" \
    azureuser@placeholder \
    'ps -p <PID> -o pid=,etime=,cmd='
```

## 5. Launch Pattern

Use `CUDA_VISIBLE_DEVICES` to bind exactly one run to exactly one GPU.

Recommended pattern:

```bash
nohup env CUDA_VISIBLE_DEVICES=<GPU_ID> \
  /scratch/boltzmann-attention-v3-repro/.venv/bin/python \
  /scratch/boltzmann-attention-v3-repro/scripts/exp4_2_v3_full_quant_ppl.py \
  ...args... \
  > /scratch/boltzmann-attention-v3-repro/logs/<run_name>.log 2>&1 &
echo $!
```

Rules:

1. one long run per GPU unless the job is tiny smoke verification
2. write a dedicated log file per run
3. write a dedicated `model-key` per run
4. keep benchmark preset aligned with the experiment plan

## 6. Smoke Before Real Run

Before any expensive run:

1. sync the latest harness to the remote node
2. run remote `--self-test`
3. run one tiny smoke with the exact preset family
4. confirm:
   - finite outputs
   - log header is correct
   - `benchmark_meta` is written to JSON

Example remote self-test:

```bash
ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_TOPS" \
    -i "$SSH_KEY" \
    azureuser@placeholder \
    'cd /scratch/boltzmann-attention-v3-repro &&
     .venv/bin/python scripts/exp4_2_v3_full_quant_ppl.py --self-test'
```

## 7. Current Experiment Tracks

The current Qwen full-K v3 tracks are organized as:

1. `hamiltonian_descriptive`
   - purpose: descriptive Hamiltonian-style geometry diagnostics
   - mode: `ppl`

2. `practical_gap_residual`
   - purpose: test whether a recent FP16 tail reduces the practical gap
   - mode: `ppl`

3. `retrieval_residual`
   - purpose: test whether retrieval depth shows gains that PPL can miss
   - mode: `niah`

Do not launch these tracks under `custom` unless the plan is explicitly revised.

## 8. Log and Result Checks

After launch, check:

```bash
ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_TOPS" \
    -i "$SSH_KEY" \
    azureuser@placeholder \
    'tail -n 30 /scratch/boltzmann-attention-v3-repro/logs/<run_name>.log'
```

For finished runs:

```bash
ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL_TOPS" \
    -i "$SSH_KEY" \
    azureuser@placeholder \
    'ls -lt /scratch/boltzmann-attention-v3-repro/results/v3 | head -20'
```

Required header checks:

1. intended `Benchmark preset`
2. intended `Intent`
3. intended `Hypothesis`
4. intended `Verification method`
5. intended `Acceptance rule`

If any of the above are wrong, stop treating the run as authoritative.

## 9. Safety Rules

1. Do not kill a remote process unless:
   - the user asked to stop it, or
   - the run is plan-misaligned and must be relaunched correctly
2. Do not overwrite a live run's log file with an unrelated experiment.
3. Do not launch duplicate runs on the same GPU.
4. Do not reuse stale JSON from a buggy harness wave as current evidence.
5. Prefer relaunch over hand-waving if a preset or metadata header is wrong.

## 10. Recommended Workflow

Use this order every time:

1. connect
2. inspect GPUs
3. inspect live logs/results
4. sync code
5. run self-test
6. run tiny smoke
7. launch real run on a free GPU
8. verify the first 20-30 log lines
9. only then treat the run as active evidence
