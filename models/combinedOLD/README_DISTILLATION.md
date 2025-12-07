# Skill Distillation for Combined Model

This directory contains the implementation of skill distillation, which combines the movement, aiming, and clicking skills into a single model using supervised learning.

## Overview

### Why Skill Distillation?

The traditional approach (in `baseline_model.py`) tries to load pre-trained PPO models and combine them directly. This has several limitations:

-   Different observation spaces between environments
-   Difficult weight transfer
-   Skills may interfere during joint training

**Skill distillation solves these issues by:**

1. Creating datasets from expert policies (trained PPO models)
2. Transforming all observations to a unified observation space
3. Training a single policy network with supervised learning
4. Optionally fine-tuning with PPO for online improvement

## Architecture

### Observation Space Unification

Each skill environment has different observations:

-   **Movement** (20 dims): Spatial position, velocities, target look direction
-   **Aiming** (5 dims): Yaw/pitch errors, distance (trigonometric encoding)
-   **Clicking** (9 dims): Aim errors, range, invulnerability, positions

All are transformed to the **Combined observation space** (20 dims):

```
[to_target_local(3), distance(1),
 agent_vel_local(3), agent_speed(1),
 target_vel_local(3), target_speed(1),
 agent_health(1), target_health(1),
 agent_on_ground(1), target_on_ground(1),
 target_look_local(3), invulnerability(1)]
```

### Action Space Unification

Each skill outputs different actions:

-   **Movement**: [w, a, s, d, space, sprint] (6 actions)
-   **Aiming**: [dyaw, dpitch] (2 actions)
-   **Clicking**: [click] (1 action)

Combined to **Combined action space** (9 dims):

```
[w, a, s, d, space, sprint, click, dyaw, dpitch]
```

### Network Architecture

The distilled policy has:

1. **Shared Encoder** (256 → 256 → 128): Processes unified observations
2. **Skill-Specific Heads**:
    - Movement head: 64 → 6 (sigmoid for binary actions)
    - Aiming head: 32 → 2 (tanh for continuous angles)
    - Clicking head: 16 → 1 (sigmoid for binary click)
3. **Output**: Concatenated 9-dim action vector

## Usage

### Quick Start (Recommended)

```bash
# Run complete workflow
chmod +x train_distilled.sh
./train_distilled.sh
```

This will:

1. Create datasets from trained models (100 episodes each)
2. Train distilled model (50 epochs)
3. Optionally fine-tune with PPO

### Step-by-Step

#### 1. Create Datasets

```bash
# Collect 100 episodes per skill
python -m models.combined.create_skill_datasets 100
```

This creates:

-   `datasets/movement_dataset.pkl`
-   `datasets/aiming_dataset.pkl`
-   `datasets/clicking_dataset.pkl`
-   `datasets/combined_dataset.pkl` (all skills merged)

#### 2. Train Distilled Model

```bash
# Train for 50 epochs
python -m models.combined.distill_skills 50
```

This trains a policy network using supervised learning on the combined dataset.

Output: `distilled_model_best.pth`

#### 3. Fine-tune with PPO (Optional)

```bash
# Fine-tune for 100K timesteps
python -m models.combined.distill_skills finetune 100000
```

This loads the distilled model and uses it to initialize a PPO policy for online learning.

Output: `distilled_ppo_finetuned_final.zip`

### Using the Distilled Model

#### Evaluate Distilled Model

```python
from models.combined.distill_skills import DistillationModel

# Load model
model = DistillationModel()
model.load("distilled_model_best.pth")

# Evaluate
model.evaluate(n_episodes=10, render=True)
```

#### Use with Visual Demo

```python
from models.combined.distill_skills import DistillationModel

# Load distilled model
distilled = DistillationModel()
distilled.load("distilled_model_best.pth")

# Use in demo
# Note: The visual demo expects PPO model, so use fine-tuned version
python visual_combined_demo.py distilled_ppo_finetuned_final
```

#### Load Fine-tuned PPO Model

```python
from stable_baselines3 import PPO

# Load fine-tuned model
model = PPO.load("distilled_ppo_finetuned_final")

# Use like any PPO model
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

## Files

-   **`create_skill_datasets.py`**: Dataset creation from trained PPO models

    -   Loads movement, aiming, clicking models
    -   Runs episodes and collects (obs, action) pairs
    -   Transforms observations to unified space
    -   Saves individual and combined datasets

-   **`distill_skills.py`**: Distilled model training and fine-tuning

    -   `DistilledPolicyNetwork`: Multi-head policy network
    -   `DistillationModel`: Main training class
    -   Supervised learning on datasets
    -   PPO fine-tuning support

-   **`train_distilled.sh`**: Complete workflow script

    -   Automated pipeline
    -   Interactive prompts
    -   Error handling

-   **`baseline_model.py`**: OLD approach (kept for reference)
    -   Direct weight transfer from PPO models
    -   Less effective than distillation

## Configuration

### Dataset Creation

Edit `create_skill_datasets.py`:

```python
# Number of episodes per skill
num_episodes = 100

# Model paths
movement_model = "best_baseline_ppo_movement_final"
aiming_model = "best_baseline_ppo_aiming_final"
clicking_model = "best_baseline_ppo_clicking_final"
```

### Supervised Training

Edit `distill_skills.py`:

```python
# Training hyperparameters
epochs = 50
batch_size = 256
lr = 1e-3
hidden_dim = 256
```

### PPO Fine-tuning

Edit `distill_skills.py`:

```python
# PPO hyperparameters
learning_rate = 3e-4
n_steps = 1024
batch_size = 512
n_epochs = 10
```

## Advantages Over Direct Transfer

1. **Unified Observation Space**: All skills see the same state representation
2. **No Weight Compatibility Issues**: Fresh network trained end-to-end
3. **Supervised Learning**: Faster initial training than pure RL
4. **Skill Integration**: Network learns how skills interact
5. **Fine-tuning Flexibility**: Can improve with online PPO learning

## Troubleshooting

### "Model file not found"

Make sure you have trained the individual skill models first:

```bash
python -m models.movement.baseline_model
python -m models.aiming.baseline_model
python -m models.clicking.baseline_model
```

### "Out of memory"

Reduce batch size in training:

```python
batch_size = 128  # instead of 256
```

### Poor performance after distillation

The distilled model learns from demonstrations, so:

1. Ensure source models are well-trained
2. Collect more episodes (increase from 100)
3. Try PPO fine-tuning for online improvement

### Fine-tuning doesn't improve

Distillation might already be near-optimal. Try:

1. Lower PPO learning rate
2. More timesteps
3. Adjust PPO hyperparameters

## Expected Results

-   **After Distillation**: Model should perform reasonably well, mimicking expert skills
-   **After Fine-tuning**: Performance should improve as model adapts to actual environment
-   **Training Time**:
    -   Dataset creation: ~10-20 min (depends on episodes)
    -   Supervised training: ~5-10 min (50 epochs)
    -   PPO fine-tuning: ~30-60 min (100K timesteps)

## References

-   Policy Distillation: Extracting knowledge from large networks into smaller ones
-   Imitation Learning: Learning from expert demonstrations
-   PPO: Proximal Policy Optimization for fine-tuning
