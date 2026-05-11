# Online RL: Off-Policy Actor-Critic for Continuous Control

## Research Question
Design and implement an off-policy actor-critic RL algorithm for
continuous control. Your code goes in `custom_offpolicy_continuous.py`.
Several reference implementations are provided as read-only `*.edit.py`
baselines.

## Background
Off-policy methods maintain a replay buffer of past transitions and
update the policy using data collected under previous policies. They are
typically more sample-efficient than on-policy methods, but they expose
well-known failure modes: overestimation bias in Q-value estimates,
instability of the actor under noisy critic targets, and
exploration-exploitation tradeoffs. Different design points address
these via twin / ensemble critics, target smoothing, entropy
regularization, delayed updates, or batch-normalized critics.

Reference baselines spanning the design space:
- **DDPG** — Lillicrap et al., "Continuous Control with Deep
  Reinforcement Learning" (arXiv:1509.02971, ICLR 2016). Single
  deterministic actor and critic with target networks and
  Ornstein-Uhlenbeck (or Gaussian) exploration noise.
- **TD3** — Fujimoto et al., "Addressing Function Approximation Error in
  Actor-Critic Methods" (arXiv:1802.09477, ICML 2018). Twin critics with
  clipped-double-Q targets, target-policy smoothing, and delayed actor
  updates (default policy delay `d = 2`).
- **SAC** — Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum
  Entropy Deep Reinforcement Learning with a Stochastic Actor"
  (arXiv:1801.01290, ICML 2018). Stochastic actor with maximum-entropy
  objective and twin-Q targets.
- **TQC** — Kuznetsov et al., "Controlling Overestimation Bias with
  Truncated Mixture of Continuous Distributional Quantile Critics"
  (arXiv:2005.04269, ICML 2020). Distributional critic with truncation
  of the top quantiles for overestimation control.
- **CrossQ** — Bhatt et al., "CrossQ: Batch Normalization in Deep
  Reinforcement Learning for Greater Sample Efficiency and Simplicity"
  (arXiv:1902.05605, ICLR 2024). Removes target networks and uses Batch
  Renormalization across `(s,a)` and `(s',a')` for stable Q updates at
  UTD = 1.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified.
- Total parameter count is enforced at runtime; the contribution must be
  algorithmic (losses, target construction, exploration, update rules)
  rather than capacity.
- Do **not** simply copy a reference implementation with minor changes.

## Evaluation
Trained and evaluated on Gymnasium MuJoCo continuous-control
environments including HalfCheetah-v4, Hopper-v4 and Walker2d-v4 within
a fixed interaction budget. Metric: mean episodic return over evaluation
episodes (higher is better). Strong methods should transfer across
environments with different dynamics and action effects.
