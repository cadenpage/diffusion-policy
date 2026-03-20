# diffusion-policy
Train a vision-conditioned diffusion policy to control a real robot (Rotom) for tabletop manipulation tasks.  This project implements a minimal pipeline for:  collecting real robot demonstrations  training a diffusion-based imitation learning policy deploying it using receding-horizon control

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

