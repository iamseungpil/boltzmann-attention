# Node Registry for boltzmann-attention experiments (2026-04-09)

## Connection Method

All nodes use AzureML websocket SSH:

```bash
AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="$HOME/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
SSH_KEY="$HOME/.ssh/id_rsa"

ssh -T -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $URL" \
    -i "$SSH_KEY" azureuser@placeholder '<command>'
```

## Nodes

| Node | URL | GPUs | Status (2026-04-09) |
|------|-----|------|---------------------|
| **E8** | `wss://ssh-2etszrmvdrq4cwqdql4al50f38gyq2afb9nhuq49bngbf1buj3c.westus2.nodes.azureml.ms` | 4× A100 80GB | **IDLE — boltzmann experiments** |
| TOPS | `wss://ssh-2etszrmvdrq4cwqdql4al50f30b0d16gxsfm589ld0y9dm0l6bc.westus2.nodes.azureml.ms` | 4× A100 80GB | Busy (other jobs) |
| EVAL | `wss://ssh-2etszrmvdrq4cwqdql4al50f30o4458xqprr3ccl017imp6anpc.westus2.nodes.azureml.ms` | 4× A100 80GB | GPU 0 busy, 1-3 idle |
| TRAIN_B | `wss://ssh-2etszrmvdrq4cwqdql4al50f3c67aahzqkey85y2iajsy6y4t5c.westus2.nodes.azureml.ms` | 4× A100 80GB | Busy (RL training) |

## E8 Node Details

- Hostname: node-0
- Python: 3.10.12
- Working dir: /scratch/
- Conda envs: check with `conda env list`
- No boltzmann-attention repo yet — need to clone or upload scripts
