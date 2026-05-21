# Distributed HRL Reproduction for Embodied Cooperation

This repository tracks the reproduction work for the paper:

**Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation**
(`2407.06499v1.pdf`).

The key implementation lives in `dhrl/`. It is not a single-RL baseline: it implements the
paper's distributed hierarchical controller:

- **Upper Layer (UL)**: encodes each agent's exteroceptive observation, including task
  features and nearest-neighbour information.
- **Middle Layer (ML)**: recurrent command policy with spatiotemporal memory.
- **Lower Layer (LL)**: locomotion-command interface. In the proxy environments this is a
  velocity/position command adapter; in IsaacSim/Gym this should be connected to the
  pretrained Ant/legged locomotion policy.

The code also includes the ablations reported by the paper:

- `dhrl`: distributed hierarchical policy with recurrent memory.
- `no_memory`: hierarchical policy without the recurrent middle layer.
- `no_hierarchy`: single flat MLP policy.

## Quick Start

Install dependencies in an environment with PyTorch:

```bash
pip install -r requirements.txt
```

Run a fast proxy reproduction:

```bash
python -m dhrl.train --config configs/dhrl/cooperative_transport.json --variant dhrl
python -m dhrl.train --config configs/dhrl/cooperative_transport.json --variant no_memory
python -m dhrl.train --config configs/dhrl/cooperative_transport.json --variant no_hierarchy
```

Run all three paper scenarios:

```bash
bash scripts/run_dhrl_proxy_4seeds.sh
```

Plot a paper-style shadow curve:

```bash
python scripts/plot_dhrl_shadow.py \
  --log-dir results/dhrl_proxy/cooperative_transport \
  --metric success_rate \
  --out results/dhrl_proxy/cooperative_transport_shadow.png
```

## IsaacSim/Gym Boundary

The paper's final numbers require IsaacSim/Gym environments and a pretrained lower-level
locomotion controller. Those assets are not included in this local project snapshot. The included
proxy environments reproduce the **algorithmic structure** and ablation protocol, while
`dhrl/isaac_adapter.py` documents the observation/action contract needed to plug the same policy
into IsaacSim/Gym tasks.

## Repository Contents

- `dhrl/models.py`: UL/ML/LL actor-critic policies and ablations.
- `dhrl/ippo.py`: independent PPO training loop for distributed multi-agent control.
- `dhrl/proxy_envs.py`: Cooperative Transport, Corridor Crossing, and Ravine Bridging proxy tasks.
- `configs/dhrl/*.json`: scenario configs following the paper.
- `scripts/run_dhrl_proxy_4seeds.sh`: multi-seed runs for shadow plots.
- `scripts/plot_dhrl_shadow.py`: mean/std curve plotting.
