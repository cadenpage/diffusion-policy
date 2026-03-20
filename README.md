# diffusion-policy
Train a vision-conditioned diffusion policy to control a real robot (Rotom) for tabletop manipulation tasks.  This project implements a minimal pipeline for:  collecting real robot demonstrations  training a diffusion-based imitation learning policy deploying it using receding-horizon control

## Rotom LeRobot Workflow

This repo is now wired to train directly from a local LeRobot dataset instead of the older hand-rolled `dataset/episode_*/data.csv` sketch below.

For Hub-based training, treat the local dataset copy only as the staging area for the first upload. After that, use a Hugging Face dataset repo and a Hugging Face model repo as the primary training and checkpoint paths.

Local dataset mapping:

- Jetson source: `Berra:~/Documents/rotom/data/lerobot/local/rotom_task_teleop/`
- Mac sync destination: `~/dev/diffusion-policy/data/local/rotom_task_teleop/`
- LeRobot `dataset.root`: `~/dev/diffusion-policy/data`
- LeRobot `dataset.repo_id`: `local/rotom_task_teleop`
- Hub dataset repo: `caden-ut/rotom_task_teleop`
- Suggested Hub policy repo: `caden-ut/rotom_task_teleop_diffusion`

The Rotom task is expected to expose only:

- `observation.images.front`
- `observation.state` with trailing dimension `4`
- `action` with trailing dimension `2`

`justfile` commands:

```bash
just venv-create
just venv-install
just hf-login
just hf-whoami
just sync-rotom-dataset
just check-rotom-dataset
just push-rotom-dataset-hub
just check-rotom-dataset-hub
just train-rotom-diffusion-hub
just infer-rotom-diffusion-hub
just push-rotom-policy-hub
just train-rotom-diffusion steps=5000 batch_size=8
just infer-rotom-diffusion
just push-rotom-policy
```

Notes:

- Training can use either the local dataset root above or a Hugging Face dataset repo.
- `just push-rotom-dataset-hub` uploads the local LeRobot dataset folder to a Hub dataset repo.
- `just train-rotom-diffusion-hub` trains directly from the Hub dataset repo and pushes checkpoints to the Hub policy repo via `policy.repo_id`.
- `just infer-rotom-diffusion-hub` loads both the dataset sample and the policy straight from the Hub, which is the same pattern you would use from Colab or another remote machine.
- The default device is `mps` for Apple Silicon. Override with `device=cpu` if needed.
- `policy.use_amp=false` and `policy.n_obs_steps=2` are intentional. This keeps the setup conservative on macOS and avoids recent LeRobot diffusion issues reported with `n_obs_steps=1`.
- Trained checkpoints land under `outputs/train/rotom_task_teleop_diffusion/`. The default offline inference smoke test uses `outputs/train/rotom_task_teleop_diffusion/checkpoints/last/pretrained_model`.
- `just push-rotom-policy` syncs that `pretrained_model` directory to `Berra:~/Documents/rotom/policies/rotom_task_teleop_diffusion/pretrained_model/`.
- Private Hub repos require `hf auth login` anywhere you train or infer, including Colab and the Jetson.
- Use Python `3.12` for this repo. Current LeRobot releases require `Python >=3.12`; `3.14` is too new for this stack today and is more likely to break PyTorch/robotics dependencies.
- Current LeRobot install docs also say `ffmpeg 8.x` is not supported yet. If your local dataset uses encoded video and you hit decode errors, install `ffmpeg 7.x` instead.

key idea: 

Diffusion Policy

Instead of predicting a single action, the model:

starts with a noisy sequence of actions

iteratively denoises it

outputs a short action plan

At runtime:

only the first action is executed

the process repeats every timestep (receding horizon)

---
Action Chunking

Each training target is a sequence:

a_t, a_{t+1}, ..., a_{t+H-1}

Typical:

horizon H = 8

These come directly from demonstrations (no manual planning).


Observations

The model uses:

overhead RGB image (workspace crop)

robot joint state (proprioception)

---

Image → small CNN → visual feature

Joint state → MLP → proprio feature

Concatenate features → conditioning vector

Diffusion model → predicts action sequence

--- 

Dataset Format

Episode-based structure:

dataset/
  episode_000/
    images/
      000000.png
      000001.png
      ...
    data.csv
  episode_001/
    ...

Each row in data.csv:

timestep,
q0, q1, q2, q3,
action_0, action_1

Each timestep corresponds to:

(image_t, state_t, action_t)

Task Setup

fixed overhead camera

tabletop workspace

fixed goal region (marked on table)

block starts in random positions

demonstrations push block into goal

---

Data Collection


control rate = 20 Hz

camera FPS = 20 Hz

Each timestep:

image

joint state

commanded action



i will log commanded actions, not resulting motion

remove idle time from demos

keep camera framing consistent

Suggested dataset size:

minimum: 10 demos

recommended: 20–30 demos

Training Pipeline
1. Dataset Loader

Samples:
(image_t, state_t) → action_chunk[t : t+H]

2. Baseline (Behavior Cloning)

Train a simple model:
(image, state) → action_chunk

Goal:

verify data is correct

confirm model can overfit

3. Diffusion Training

For each batch:

sample action chunk

add Gaussian noise

condition on observation

predict noise

minimize MSE

Inference (Robot)

Loop:

capture image + state

sample noisy action sequence

denoise with model

execute first action

repeat

---

Starter Hyperparameters

image size: 112x112

horizon: 8

action dim: 2 (dx, dy) or joint deltas

learning rate: 1e-4

diffusion steps:

training: 50

inference: 10–20
