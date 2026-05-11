"""Pre-edit operations for omnisafe package.

1. Register CustomLag algorithm in naive_lagrange __init__.py
2. Create CustomLag.yaml config (based on PPOLag defaults, 1M steps)
3. Create train_safe_rl.py training script
"""

# ── 1. Register CustomLag import ────────────────────────────────────

_CUSTOM_LAG_IMPORT = (
    "from omnisafe.algorithms.on_policy.naive_lagrange.custom_lag import CustomLag\n"
)

# ── 2. CustomLag.yaml config (based on PPOLag, reduced to 1M steps) ──

_CUSTOM_LAG_YAML = """\
defaults:
  seed: 0
  train_cfgs:
    device: cuda:0
    torch_threads: 16
    vector_env_nums: 1
    parallel: 1
    total_steps: 2000000
  algo_cfgs:
    steps_per_epoch: 20000
    update_iters: 40
    batch_size: 64
    target_kl: 0.02
    entropy_coef: 0.0
    reward_normalize: False
    cost_normalize: False
    obs_normalize: True
    kl_early_stop: True
    use_max_grad_norm: True
    max_grad_norm: 40.0
    use_critic_norm: True
    critic_norm_coef: 0.001
    gamma: 0.99
    cost_gamma: 0.99
    lam: 0.95
    lam_c: 0.95
    clip: 0.2
    adv_estimation_method: gae
    standardized_rew_adv: True
    standardized_cost_adv: True
    penalty_coef: 0.0
    use_cost: True
  logger_cfgs:
    use_wandb: False
    wandb_project: omnisafe
    use_tensorboard: True
    save_model_freq: 100
    log_dir: "./runs"
    window_lens: 100
  model_cfgs:
    weight_initialization_mode: "kaiming_uniform"
    actor_type: gaussian_learning
    linear_lr_decay: True
    exploration_noise_anneal: False
    std_range: [0.5, 0.1]
    actor:
      hidden_sizes: [64, 64]
      activation: tanh
      lr: 0.0003
    critic:
      hidden_sizes: [64, 64]
      activation: tanh
      lr: 0.0003
  lagrange_cfgs:
    cost_limit: 25.0
    lagrangian_multiplier_init: 0.001
    lambda_lr: 0.035
    lambda_optimizer: "Adam"
  env_cfgs: {}
"""

# ── 3. Training script ──────────────────────────────────────────────

_TRAIN_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"Train CustomLag for MLS-Bench safe-rl task.\"\"\"
import argparse
import omnisafe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='CustomLag')
    parser.add_argument('--env-id', default='SafetyPointGoal1-v0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total-steps', type=int, default=1000000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        custom_cfgs={
            'seed': args.seed,
            'train_cfgs': {
                'total_steps': args.total_steps,
                'device': args.device,
            },
        },
    )
    agent.learn()
"""

# ── OPS (ordered bottom-to-top within each file) ────────────────────

_NAIVE_LAG_INIT = """\
\"\"\"Naive Lagrange algorithms.\"\"\"

from omnisafe.algorithms.on_policy.naive_lagrange.pdo import PDO
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.algorithms.on_policy.naive_lagrange.rcpo import RCPO
from omnisafe.algorithms.on_policy.naive_lagrange.trpo_lag import TRPOLag
from omnisafe.algorithms.on_policy.naive_lagrange.custom_lag import CustomLag


__all__ = [
    'RCPO',
    'PDO',
    'PPOLag',
    'TRPOLag',
    'CustomLag',
]
"""

_ON_POLICY_INIT_LINE32 = (
    "from omnisafe.algorithms.on_policy.naive_lagrange import PDO, RCPO, PPOLag, TRPOLag, CustomLag\n"
)

OPS = [
    # 1. Rewrite naive_lagrange __init__.py (lines 15-28) to include CustomLag
    {
        "op": "replace",
        "file": "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/__init__.py",
        "start_line": 15,
        "end_line": 28,
        "content": _NAIVE_LAG_INIT,
    },
    # 2. Add CustomLag import in on_policy __init__.py (replace line 32)
    {
        "op": "replace",
        "file": "omnisafe/omnisafe/algorithms/on_policy/__init__.py",
        "start_line": 32,
        "end_line": 32,
        "content": _ON_POLICY_INIT_LINE32,
    },
    # 3. Create CustomLag.yaml config
    {
        "op": "create",
        "file": "omnisafe/omnisafe/configs/on-policy/CustomLag.yaml",
        "content": _CUSTOM_LAG_YAML,
    },
    # 4. Create training script
    {
        "op": "create",
        "file": "omnisafe/train_safe_rl.py",
        "content": _TRAIN_SCRIPT,
    },
]
