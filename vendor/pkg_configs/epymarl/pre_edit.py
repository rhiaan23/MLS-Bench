"""Pre-edit operations for epymarl package.

Applied before the agent starts:
1. Register CustomMixer in q_learner.py (import + elif branch)
2. Inject TRAIN_METRICS / TEST_METRICS prints in episode_runner.py
3. Create custom.yaml algorithm config (based on qmix.yaml, mixer="custom")
4. Register CustomCritic (try/except; no-op for tasks that don't create it)
5. Inject TRAIN_METRICS / TEST_METRICS prints in parallel_runner.py (MAPPO path)
6. Create custom_mappo.yaml algorithm config (based on mappo.yaml, critic_type="custom_critic")
"""

# ── 1. q_learner.py: add CustomMixer import after line 9 ────────────────
# Wrapped in try/except so non-mixer tasks (e.g. marl-centralized-critic
# which uses ppo_learner with custom_critic) don't fail at import time
# when mixers/custom.py doesn't exist.

_CUSTOM_IMPORT = (
    "try:\n"
    "    from modules.mixers.custom import CustomMixer\n"
    "except ImportError:\n"
    "    CustomMixer = None\n"
)

# ── 2. q_learner.py: replace lines 30-31 (else: raise) with custom branch
_CUSTOM_MIXER_BRANCH = """\
            elif args.mixer == "custom":
                assert args.common_reward, "CustomMixer only supports common reward setting"
                if CustomMixer is None:
                    raise ImportError("mixer='custom' requires modules/mixers/custom.py to exist (created by mid_edit)")
                self.mixer = CustomMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
"""

# ── 3. episode_runner.py: inject metrics print after line 159 ────────────

_METRICS_PRINT = (
    '            _mtag = "TEST_METRICS" if prefix == "test_" else "TRAIN_METRICS"\n'
    '            _n_eps = max(stats.get("n_episodes", 1), 1)\n'
    '            _bw = stats.get("battle_won", 0) / _n_eps\n'
    '            try:\n'
    '                print(f"{_mtag} t_env={self.t_env} '
    'return_mean={np.mean(returns):.4f} '
    'return_std={np.std(returns):.4f} '
    'battle_won_mean={_bw:.4f}", flush=True)\n'
    '            except (BrokenPipeError, OSError):\n'
    '                pass  # apptainer/SLURM cleanup may have closed stdout\n'
)

# ── 4. custom.yaml algorithm config ─────────────────────────────────────

_CUSTOM_YAML = """\
# --- Custom mixer parameters ---

# use epsilon greedy action selector
action_selector: "epsilon_greedy"
epsilon_start: 1.0
epsilon_finish: 0.05
epsilon_anneal_time: 500000
evaluation_epsilon: 0.0

runner: "episode"

buffer_size: 5000

# update the target network every {} episodes
target_update_interval_or_tau: 200


obs_agent_id: True
obs_last_action: False
obs_individual_obs: False


# use the Q_Learner to train
standardise_returns: False
standardise_rewards: True

agent_output_type: "q"
learner: "q_learner"
double_q: True
mixer: "custom"
use_rnn: True
mixing_embed_dim: 32
hypernet_layers: 2
hypernet_embed: 64

name: "custom"
"""

# ── 5. envs/__init__.py: make smaclite import optional ─────────────────

_SMACLITE_FIX = """\
try:
    from .smaclite_wrapper import SMACliteWrapper
except ImportError:
    SMACliteWrapper = None
"""

# ── 6. critics/__init__.py: register CustomCritic with try/except ───────
# Marl tasks that don't create custom_critic.py (e.g. marl-mixing-network,
# which uses Q-learning mixers) silently skip via the ImportError branch.

_CUSTOM_CRITIC_REGISTER = """\
try:
    from .custom_critic import CustomCritic
    REGISTRY["custom_critic"] = CustomCritic
except ImportError:
    pass
"""

# ── 7. parallel_runner.py: inject metrics print (MAPPO / parallel runners)
# Placed at end of the common_reward branch of _log (after return_std log_stat),
# BEFORE returns.clear() on the next line. Mirrors the episode_runner injection
# format but adds battle_won_mean (from smaclite info dict via cur_stats).

_PARALLEL_METRICS_PRINT = (
    '            _mtag = "TEST_METRICS" if prefix == "test_" else "TRAIN_METRICS"\n'
    '            _n_eps = max(stats.get("n_episodes", 1), 1)\n'
    '            _bw = stats.get("battle_won", 0) / _n_eps\n'
    '            try:\n'
    '                print(f"{_mtag} t_env={self.t_env} '
    'return_mean={np.mean(returns):.4f} '
    'return_std={np.std(returns):.4f} '
    'battle_won_mean={_bw:.4f}", flush=True)\n'
    '            except (BrokenPipeError, OSError):\n'
    '                pass  # apptainer/SLURM cleanup may have closed stdout\n'
)

# ── 8. custom_mappo.yaml algorithm config ───────────────────────────────
# Based on upstream src/config/algs/mappo.yaml with critic_type swapped
# to the custom registry entry. Harmless if unused.

_CUSTOM_MAPPO_YAML = """\
# --- Custom MAPPO parameters ---
# Hyperparameters match Yu et al. 2022 "The Surprising Effectiveness of
# PPO in Cooperative Multi-Agent Games" (arXiv 2103.01955) SMAC setup.

action_selector: "soft_policies"
mask_before_softmax: True

runner: "parallel"

# 32 parallel envs matches Yu et al. 2022 SMAC setup. EPyMARL default is 10;
# we bump to 32 for SMAC rigor.
buffer_size: 32
batch_size_run: 32
batch_size: 32

# update the target network every {} training steps
target_update_interval_or_tau: 0.01

lr: 0.0005
hidden_dim: 128

obs_agent_id: True
obs_last_action: True
obs_individual_obs: False

agent_output_type: "pi_logits"
learner: "ppo_learner"
entropy_coef: 0.01
use_rnn: True
standardise_returns: True
standardise_rewards: True
q_nstep: 5
critic_type: "custom_critic"
epochs: 15
eps_clip: 0.2
grad_norm_clip: 10
name: "custom_mappo"
"""

# ── 9. gymma.yaml: add local_ratio for PettingZoo MPE envs ─────────────
# Papoudakis et al. 2021 benchmarks used OpenAI MPE with purely global
# reward. PettingZoo defaults to local_ratio=0.5 (mixed local+global)
# which causes QMIX to underperform VDN on spread, contradicting theory.
# Declare local_ratio as an env_arg so tasks can override it.

_GYMMA_YAML = """\
env: "gymma"

env_args:
  key: null
  time_limit: 100
  pretrained_wrapper: null
  local_ratio: null

test_greedy: True
test_nepisode: 100
test_interval: 50000
log_interval: 50000
runner_log_interval: 10000
learner_log_interval: 10000
t_max: 2050000
"""

# ── 10. gymma.py: pop local_ratio before gym.make ───────────────────────
# Only some PettingZoo envs accept local_ratio (e.g. simple_spread does,
# simple_tag does NOT). Pop it from kwargs, then try passing it to gym.make;
# if the env rejects it, fall back to creating without it.

_GYMMA_PY_FIX = (
    "        _local_ratio = kwargs.pop('local_ratio', None)\n"
    "        kwargs = {k: v for k, v in kwargs.items() if v is not None}\n"
    "        if _local_ratio is not None:\n"
    "            try:\n"
    "                self._env = gym.make(f\"{key}\", local_ratio=_local_ratio, **kwargs)\n"
    "            except TypeError:\n"
    "                self._env = gym.make(f\"{key}\", **kwargs)\n"
    "        else:\n"
    "            self._env = gym.make(f\"{key}\", **kwargs)\n"
)

# Ops ordered bottom-to-top within each file for line-number stability.
OPS = [
    # --- q_learner.py ---
    # Replace lines 30-31: add custom mixer elif before the else
    {
        "op": "replace",
        "file": "epymarl/src/learners/q_learner.py",
        "start_line": 30,
        "end_line": 31,
        "content": _CUSTOM_MIXER_BRANCH,
    },
    # Insert CustomMixer import after line 9
    {
        "op": "insert",
        "file": "epymarl/src/learners/q_learner.py",
        "after_line": 9,
        "content": _CUSTOM_IMPORT,
    },
    # --- episode_runner.py ---
    # Insert METRICS print after line 159 (inside common_reward branch)
    {
        "op": "insert",
        "file": "epymarl/src/runners/episode_runner.py",
        "after_line": 159,
        "content": _METRICS_PRINT,
    },
    # --- Create custom.yaml ---
    {
        "op": "create",
        "file": "epymarl/src/config/algs/custom.yaml",
        "content": _CUSTOM_YAML,
    },
    # --- envs/__init__.py: make smaclite import optional ---
    {
        "op": "replace",
        "file": "epymarl/src/envs/__init__.py",
        "start_line": 6,
        "end_line": 6,
        "content": _SMACLITE_FIX,
    },
    # --- critics/__init__.py: register CustomCritic with try/except ---
    # Insert after line 20 (end of REGISTRY entries, before register_pac_critics).
    {
        "op": "insert",
        "file": "epymarl/src/modules/critics/__init__.py",
        "after_line": 20,
        "content": _CUSTOM_CRITIC_REGISTER,
    },
    # --- parallel_runner.py: inject metrics print in _log() common_reward branch ---
    # Insert after line 258 (the return_std log_stat) at indent 12, before the else branch.
    {
        "op": "insert",
        "file": "epymarl/src/runners/parallel_runner.py",
        "after_line": 258,
        "content": _PARALLEL_METRICS_PRINT,
    },
    # --- Create custom_mappo.yaml ---
    {
        "op": "create",
        "file": "epymarl/src/config/algs/custom_mappo.yaml",
        "content": _CUSTOM_MAPPO_YAML,
    },
    # --- gymma.py: filter None kwargs on line 39 (self._env = gym.make ...) ---
    {
        "op": "replace",
        "file": "epymarl/src/envs/gymma.py",
        "start_line": 39,
        "end_line": 39,
        "content": _GYMMA_PY_FIX,
    },
    # --- Overwrite gymma.yaml to declare local_ratio ---
    {
        "op": "create",
        "file": "epymarl/src/config/envs/gymma.yaml",
        "content": _GYMMA_YAML,
    },
]
