"""Pre-edit for oyster package.

1. Fix envs/__init__.py to handle missing optional dependencies (rand_param_envs)
2. Inject TRAIN_METRICS and TEST_METRICS prints in rl_algorithm.py evaluate()
3. Fix detach_z to handle encoders without hidden state
4. Fix mujoco_env.py for gym 0.25.2 API compatibility (observation_space arg)
5. Fix pytorch_util.py set_gpu_mode: don't override CUDA_VISIBLE_DEVICES
6. Fix np.stack(generator) for NumPy 2.x compatibility in rl_algorithm.py
7. Replace deprecated np.bool alias on the one line that uses it.
"""

OPS = [
    # Replace deprecated `np.bool` alias (removed in NumPy >=1.20) with the
    # underscored modern equivalent. The H100 GPU forces a torch>=2.2 upgrade
    # in install_cmds, which transitively pulls numpy>=1.20 where the bare
    # alias was removed. Only one line in pytorch_util.py uses it.
    {
        "op": "replace",
        "file": "oyster/rlkit/torch/pytorch_util.py",
        "start_line": 54,
        "end_line": 54,
        "content": "        if v.dtype == np.bool_:\n",
    },
    # Fix pytorch_util.py: set_gpu_mode() overrides CUDA_VISIBLE_DEVICES with
    # the --gpu CLI arg, clobbering the scheduler's GPU assignment.
    # Replace lines 75-83 to respect the existing CUDA_VISIBLE_DEVICES.
    {
        "op": "replace",
        "file": "oyster/rlkit/torch/pytorch_util.py",
        "start_line": 75,
        "end_line": 83,
        "content": (
            "def set_gpu_mode(mode, gpu_id=0):\n"
            "    global _use_gpu\n"
            "    global device\n"
            "    global _gpu_id\n"
            "    _gpu_id = gpu_id\n"
            "    _use_gpu = mode\n"
            "    device = torch.device('cuda:0' if _use_gpu else 'cpu')\n"
        ),
    },
    # Fix envs/__init__.py: wrap auto-imports in try/except for missing packages
    {
        "op": "replace",
        "file": "oyster/rlkit/envs/__init__.py",
        "start_line": 22,
        "end_line": 26,
        "content": (
            "# automatically import any envs in the envs/ directory\n"
            "for file in os.listdir(os.path.dirname(__file__)):\n"
            "    if file.endswith('.py') and not file.startswith('_'):\n"
            "        module = file[:file.find('.py')]\n"
            "        try:\n"
            "            importlib.import_module('rlkit.envs.' + module)\n"
            "        except (ImportError, ModuleNotFoundError):\n"
            "            pass\n"
        ),
    },
    # Inject TRAIN_METRICS and TEST_METRICS prints in evaluate()
    # After line 460: self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
    {
        "op": "insert",
        "file": "oyster/rlkit/core/rl_algorithm.py",
        "after_line": 460,
        "content": (
            "        print(f'TRAIN_METRICS iteration={epoch} "
            "avg_train_return={avg_train_return:.4f}', flush=True)\n"
            "        print(f'TEST_METRICS iteration={epoch} "
            "meta_test_return={avg_test_return:.4f}', flush=True)\n"
        ),
    },
    # Fix detach_z to gracefully handle encoders without hidden state
    {
        "op": "replace",
        "file": "oyster/rlkit/torch/sac/agent.py",
        "start_line": 90,
        "end_line": 94,
        "content": (
            "    def detach_z(self):\n"
            "        ''' disable backprop through z '''\n"
            "        self.z = self.z.detach()\n"
            "        if self.recurrent and hasattr(self.context_encoder, 'hidden'):\n"
            "            self.context_encoder.hidden = self.context_encoder.hidden.detach()\n"
        ),
    },
    # Fix mujoco_env.py for gym 0.25.2 API compatibility
    # gym 0.25.2's MujocoEnv.__init__ requires observation_space parameter.
    # Replace the auto-setup branch to pass observation_space from _get_obs.
    {
        "op": "replace",
        "file": "oyster/rlkit/envs/mujoco_env.py",
        "start_line": 28,
        "end_line": 29,
        "content": (
            "        if automatically_set_obs_and_action_space:\n"
            "            import inspect\n"
            "            sig = inspect.signature(mujoco_env.MujocoEnv.__init__)\n"
            "            if 'observation_space' in sig.parameters:\n"
            "                # gym >= 0.25: must pass observation_space\n"
            "                from gym import spaces\n"
            "                mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip,\n"
            "                    observation_space=spaces.Box(-np.inf, np.inf, shape=(999,)))\n"
            "                # Override with actual obs space after init\n"
            "                obs = self._get_obs()\n"
            "                self.observation_space = spaces.Box(-np.inf, np.inf,\n"
            "                    shape=obs.shape, dtype=np.float64)\n"
            "            else:\n"
            "                mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)\n"
        ),
    },
    # Fix np.stack(generator) → np.stack(list(...)) for NumPy 2.x compatibility.
    # sparse-point-robot uses sparse_rewards which hits this code path.
    {
        "op": "replace",
        "file": "oyster/rlkit/core/rl_algorithm.py",
        "start_line": 366,
        "end_line": 366,
        "content": (
            "                sparse_rewards = np.stack("
            "[e['sparse_reward'] for e in p['env_infos']]"
            ").reshape(-1, 1)\n"
        ),
    },
    {
        "op": "replace",
        "file": "oyster/rlkit/core/rl_algorithm.py",
        "start_line": 432,
        "end_line": 432,
        "content": (
            "                    sparse_rewards = np.stack("
            "[e['sparse_reward'] for e in p['env_infos']]"
            ").reshape(-1, 1)\n"
        ),
    },
    # Fix simple_replay_buffer.random_sequence: line 79 references
    # `self.episode_starts` (no underscore), but the attribute is consistently
    # named `self._episode_starts` everywhere else in the file. The typo is
    # harmless when nothing calls random_sequence, but VariBAD and the PEARL
    # recurrent encoder ablation both need ordered trajectory sampling and
    # trip on it (AttributeError).
    {
        "op": "replace",
        "file": "oyster/rlkit/data_management/simple_replay_buffer.py",
        "start_line": 79,
        "end_line": 79,
        "content": (
            "            start = np.random.choice(self._episode_starts[:-1])\n"
        ),
    },
    # Fix HalfCheetah observation_space mismatch: oyster adds torso COM (3 dims)
    # to _get_obs but inherits the parent's 17-dim observation_space.
    {
        "op": "insert",
        "file": "oyster/rlkit/envs/half_cheetah.py",
        "after_line": 3,
        "content": (
            "from gym import spaces\n"
            "\n"
        ),
    },
    {
        "op": "insert",
        "file": "oyster/rlkit/envs/half_cheetah.py",
        "after_line": 6,
        "content": (
            "    def __init__(self, *args, **kwargs):\n"
            "        super().__init__(*args, **kwargs)\n"
            "        # Fix obs space: _get_obs adds torso COM (3 dims) on top of parent's 17\n"
            "        obs = self._get_obs()\n"
            "        self.observation_space = spaces.Box(\n"
            "            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64\n"
            "        )\n"
            "\n"
        ),
    },
]
