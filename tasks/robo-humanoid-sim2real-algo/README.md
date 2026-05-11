# Robo-Humanoid Sim2Real Algorithm Task

This task evaluates an agent's ability to design and optimize reinforcement learning algorithms for humanoid robot locomotion.

## Task Description

The agent is given freedom to modify the PPO algorithm implementation in humanoidgym, including:
- **Network Architecture** (actor_critic_custom.py): Modify the actor and critic neural network architectures, activation functions, initialization strategies, etc.
- **Algorithm Implementation** (ppo_custom.py): Adjust the PPO algorithm including loss functions, optimization strategies, learning rate schedules, clipping parameters, etc.
- **Experience Buffer** (rollout_storage_custom.py): Modify how experiences are stored and sampled, including advantage computation, return calculation, and mini-batch generation.

## Evaluation

The task uses a sim2sim evaluation protocol:
- Training and evaluation on the XBotL humanoid robot
- Performance measured by locomotion success in simulation and sim-to-real transfer

## Files

### Editable Files
- `humanoid-gym/humanoid/algo/ppo/actor_critic_custom.py` - Neural network architecture
- `humanoid-gym/humanoid/algo/ppo/ppo_custom.py` - PPO algorithm implementation
- `humanoid-gym/humanoid/algo/ppo/rollout_storage_custom.py` - Experience buffer

### Reference Files (Read-only)
- `humanoid-gym/humanoid/algo/ppo/actor_critic.py` - Original actor-critic implementation
- `humanoid-gym/humanoid/algo/ppo/ppo.py` - Original PPO implementation
- `humanoid-gym/humanoid/algo/ppo/rollout_storage.py` - Original rollout storage implementation

## Baseline

The default baseline uses the official PPO implementation from humanoid-gym with:
- 3-layer MLP with 256 hidden units and ELU activation
- Standard PPO with clipped surrogate objective
- GAE for advantage estimation
- Standard experience buffer with mini-batch sampling
