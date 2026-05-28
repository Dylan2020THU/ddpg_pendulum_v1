# DDPG for Pendulum-v1

A PyTorch implementation of **Deep Deterministic Policy Gradient (DDPG)** for the classic inverted pendulum control task in OpenAI Gym.

The agent learns a continuous control policy to swing the pendulum upright and keep it balanced. Training uses an actor–critic architecture with experience replay and soft target network updates.

## Project Structure

```
ddpg_pendulum_v1/
├── ddpg_agent.py   # Actor, Critic, replay buffer, and DDPG agent
├── ddpg_train.py   # Training loop and model checkpoint saving
├── ddpg_test.py    # Evaluation with Pygame visualization
├── models/         # Saved actor/critic weights (created during training)
└── requirements.txt
```

## Requirements

- Python 3.9 (recommended)
- See `requirements.txt` for package versions

| Package     | Purpose                          |
|-------------|----------------------------------|
| PyTorch     | Neural networks and optimization |
| Gym         | Pendulum-v1 environment          |
| NumPy       | Array operations                 |
| Matplotlib  | Optional reward plotting         |
| Pygame      | Test-time rendering              |

### Installation

```bash
pip install -r requirements.txt
```

For GPU support with CUDA 12.1:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

## Training

Run the training script from the project root:

```bash
python ddpg_train.py
```

Default settings:

| Parameter        | Value  |
|------------------|--------|
| Episodes         | 1000   |
| Steps per episode| 200    |
| Actor LR         | 1e-4   |
| Critic LR        | 1e-3   |
| Discount (γ)     | 0.99   |
| Soft update (τ)  | 0.005  |
| Replay buffer    | 100000 |
| Batch size       | 64     |

After training, the script saves:

- Actor and critic weights under `models/` (timestamped `.pth` files)
- Episode rewards to `ddpg_reward_<timestamp>.txt`

## Testing

1. Place a trained actor checkpoint in the `models/` directory.
2. Update `actor_path` in `ddpg_test.py` to match your checkpoint filename.
3. Run:

```bash
python ddpg_test.py
```

The test script loads the actor, runs episodes in the Pendulum environment, and displays the simulation in a Pygame window.

## Algorithm Overview

DDPG is an off-policy actor–critic method for continuous action spaces:

- **Actor** maps states to deterministic actions in `[-2, 2]`.
- **Critic** estimates Q-values for state–action pairs.
- **Target networks** are slowly updated via Polyak averaging for stable learning.
- **Experience replay** breaks temporal correlation in training samples.
