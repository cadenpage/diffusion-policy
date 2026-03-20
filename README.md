# diffusion-policy

This repo is the machine-learning side of my Rotom project. The robot repo covers the hardware, ROS 2 stack, teleoperation, perception, and demonstration collection. This repo focuses on the learning side: how I turn those demonstrations into a vision-conditioned diffusion policy, why the reduced action space matters, and how imitation learning and reinforcement learning fit together for this robot.

Training and publishing now happen primarily in [`scripts/rotom_lerobot.ipynb`](scripts/rotom_lerobot.ipynb). I intentionally wanna keep the rest of this repo small: a helper script for dataset checks and offline inference, a minimal `justfile`, and this README as the ML-side explanation. The justfile was the initial idea for running a start python file I got from a template, but the notebook allows me to use Colab GPUs and is more flexible for iterating on the training process, so it became the main interface.

## Setup Photo

<div align="center">
  <img src="imgs/IL_layout.png" alt="Imitation learning setup" height="700"/>
</div>

## What I Am Actually Learning

I am not trying to learn raw torques or a generic 6D Cartesian controller. I am learning a policy in the same reduced task space that already makes sense for my robot and for tabletop tasks.

At each timestep, I treat the learning problem as:

```text
o_t = {I_t, q_t}
q_t = [O, A, B, C]
a_t = [dx, dy]
```

where:

- `I_t` is the current overhead RGB image,
- `q_t` is the current joint-state vector,
- and `a_t` is a small planar task-space action.

That design choice matters. Rotom is a 4-DoF arm, but the tasks I care about are mostly planar tabletop manipulation. I already know from teleoperation that this lower-dimensional action space is easier to control, easier to collect demos in, and better aligned with the actual geometry of the mechanism. So instead of asking a policy to rediscover all of that from scratch, I encode those assumptions directly into the learning interface.

## End-To-End Pipeline

The full learning pipeline is:

```text
human teleop on Rotom
-> raw real-robot demonstrations
-> LeRobot dataset conversion
-> diffusion-policy notebook training
-> checkpoint export / Hub publish
-> robot-side policy deployment
```

More concretely:

1. I collect demonstrations on the real robot using the reduced task-space teleop interface from the Rotom stack.
2. The robot repo converts those raw episodes into a LeRobot dataset with a small policy-facing interface.
3. This repo consumes that dataset, usually through the notebook, and trains a vision-conditioned diffusion policy.
4. The trained checkpoint can be uploaded to the Hugging Face Hub or loaded back on the robot side for inference.

The important boundary is that this repo is not the robot controller. The robot-side controller, kinematics, pitch closure, smoothing, and hardware constraints already exist elsewhere. This repo sits one level above that and learns the action policy that feeds that controller.

## Why Imitation Learning Is The Right Starting Point

For this project, imitation learning is the right first move because I already have a human teleoperation interface that can produce demonstrations in the exact action space I want the policy to use.

That gives me three advantages:

- I can encode task strategy with demonstrations instead of inventing a fragile reward function first.
- I can stay in a low-dimensional, safety-aware action space that already works on hardware.
- I can get useful behavior from a small real-world dataset much faster than starting from online RL.

Diffusion policy is still imitation learning. The "diffusion" part is not a separate training paradigm. It is the way the policy models the conditional distribution over short action sequences given the current observation.

## Learning Problem And Math

### 1. Supervised imitation target

Instead of predicting only the next action, you predict an action chunk (common in diffusion policy). For a horizon `H`, the target action chunk is:

```text
A_t^(H) = [a_t, a_(t+1), etc , a_(t+H-1)]
```

So each training example is:

```text
(o_t, A_t^(H))
```

This is useful because robot control is inherently sequential. A short horizon lets the policy represent local intent (just like how image backbones work), not just a single isolated action.

A plain behavior cloning baseline would learn a direct mapping:

```text
f_theta(o_t) -> A_t^(H)
```

with a loss such as:

```text
L_BC(theta) = E[ ||f_theta(o_t) - A_t^(H)||^2 ]
```

That baseline is worth understanding because it exposes the core supervised-learning structure of the problem. But it also has a weakness: when multiple short-term action sequences are plausible, direct regression tends to average them together.

### 2. Diffusion policy objective

Diffusion policy replaces direct chunk regression with denoising. The clean action chunk `x_0` is the demonstration chunk `A_t^(H)`. During training, Gaussian noise is added at a random diffusion step `k`:

```text
x_k = sqrt(alpha_bar_k) * x_0 + sqrt(1 - alpha_bar_k) * epsilon
epsilon ~ N(0, I)
```

The model sees:

- the noisy action chunk `x_k`,
- the diffusion step `k`,
- and the conditioning information from the observation `o_t`.

It then predicts the noise that was added:

```text
epsilon_theta(x_k, k, cond(o_t))
```

and is trained with the standard denoising loss:

```text
L_diff(theta) = E[ ||epsilon - epsilon_theta(x_k, k, cond(o_t))||^2 ]
```

Conceptually, that means the model learns how to move from "noisy possible action sequence" back toward "valid action sequence for this observation."

### 3. Receding-horizon execution

At inference time, I start from noise, iteratively denoise an action chunk, and then execute only the first action or first few actions before replanning.

```text
noise
-> denoised action chunk
-> execute first action
-> observe again
-> replan
```

That receding-horizon structure matters. The model does not need to produce a perfect long-horizon plan open loop. It only needs to propose a good local action sequence, then it gets corrected again on the next observation.

## What The Network Is Doing In Practice

At a high level, the implementation is:

```text
image -> visual encoder
joint state -> MLP
fused observation features -> conditioning vector
conditioning + noisy action chunk + timestep -> diffusion denoiser
denoised chunk -> action plan
```

In LeRobot, the details of the scheduler, checkpointing, and policy wrapper are handled by the library. The important conceptual point is that the model is learning a conditional distribution over short action sequences, not just a single deterministic control output.

For Rotom, that is a good fit because the same visual scene can admit several locally reasonable motions. A pushing task may allow a left-approach or right-approach correction that are both valid. Diffusion models handle that multimodality better than plain MSE regression.

## Why This Representation Fits Rotom Specifically

This repo makes the most sense when viewed together with the robot-side controller design.

On the robot, I already reduced the control problem to a lower-dimensional task interface that matches the mechanism and the tabletop tasks I care about. That means:

- the policy does not need to learn full inverse kinematics from pixels,
- the policy does not need to output raw joint torques or motor-space commands,
- and the hardware-facing controller still handles the geometric and safety structure downstream.

That separation is one of the main reasons I expect diffusion policy to be practical here. The learning problem is no longer "map image to everything." It is "map image + current arm configuration to the next useful planar action sequence."

This is also why `observation.state` still matters even with vision. The camera tells the policy about the object and workspace. The joint state tells the policy where the robot currently is, which changes reachability, approach direction, and the meaning of the next planar move.

## The Dataset Interface I Am Using

The policy-facing dataset is deliberately minimal:

- `observation.images.front`
- `observation.state`
- `action`

In this repo, I expect:

- `observation.state` trailing dimension `4`
- `action` trailing dimension `2`

Those assumptions are checked in [`scripts/rotom_lerobot.py`](scripts/rotom_lerobot.py).

In practice, that means:

- image: fixed-view workspace observation,
- state: current joint configuration `[O, A, B, C]`,
- action: reduced task-space delta, typically `[dx, dy]`.

This is the main implementation choice that ties the ML repo back to the robot repo. I am learning in the same action space that I already proved is meaningful during human teleoperation.

## What Good Training Should Mean Here

For this project, good training does not just mean driving the loss down. It should mean:

- the policy can overfit a tiny subset of demos when I want a smoke test,
- the predicted action chunks are smooth and directionally sensible,
- the policy reacts to object position changes in the image,
- and the policy stays compatible with the reduced controller interface used on the robot.

Because the real-world dataset is initially small, consistency matters more than scale at first:

- same camera viewpoint,
- same workspace framing,
- same reset pose,
- same object and table setup,
- and mostly successful demonstrations.

That reduces the burden on the model and makes the imitation-learning assumption more realistic.

## Where Reinforcement Learning Fits Later

I do think reinforcement learning can be useful here, but as a second stage, not the starting point.

The standard RL objective is:

```text
J(pi) = E_pi [ sum_t gamma^t r_t ]
```

^^ the expected discounted return under the policy's trajectory distribution

For this task, a reasonable reward would likely combine:

```text
r_t =
  w_goal * task_progress
  - w_smooth * ||Delta a_t||^2
  - w_limit * joint_limit_cost
  - w_fail * failure_events
```

^^ this is similar to how my controller's cost function is structured, but with a task progress term added in.

The role of RL would be to fine-tune a policy that already knows the task at a basic level from demonstrations. That is much more realistic on a real robot than learning from scratch online.

The main ways RL could help are:

- improving robustness when the state drifts away from the demonstration distribution,
- refining contact behavior and recovery motions,
- and optimizing for a task-specific reward after the imitation policy already works.

The main risks are:

- poor sample efficiency on real hardware,
- reward hacking if the reward is underspecified,
- and unsafe exploratory behavior if the action interface is not tightly constrained.

For this reason, the most practical path for Rotom is:

1. collect demonstrations in the reduced task space,
2. train a diffusion imitation policy,
3. validate that the policy behaves sensibly offline and on-hardware,
4. then consider RL fine-tuning on top of the same action interface.

That keeps the robot-side controller, action clipping, and task-space structure as safety priors instead of throwing them away.

## Minimal Repo Workflow

The notebook is now the main place for training and publishing:

```bash
just venv-create
just venv-install
just notebook
```

If I want to stage a local dataset from the robot machine first:

```bash
just sync-rotom-dataset
just check-rotom-dataset
```

If I already have a trained checkpoint and want a quick offline action prediction:

```bash
just infer-rotom-checkpoint
```

The notebook can train from either a local staged dataset or a Hugging Face dataset repo. Before running the notebook, I should update any hard-coded dataset/model repo ids there to match the current experiment.

## Repo Map

- [`scripts/rotom_lerobot.ipynb`](scripts/rotom_lerobot.ipynb): canonical notebook for training and publishing checkpoints
- [`scripts/rotom_lerobot.py`](scripts/rotom_lerobot.py): dataset verification and offline inference smoke tests
- [`justfile`](justfile): minimal convenience commands for environment setup and local checks
- [`requirements.txt`](requirements.txt): Python dependencies for the notebook and helper script

## Summary

This repo is intentionally narrow. The robot repo handles the physical system and demonstration collection. This repo explains and runs the learning stage: use real demonstrations, keep the action space aligned with the reduced controller, train a diffusion policy that predicts short action chunks from image + state, and leave reinforcement learning as a targeted fine-tuning tool after imitation learning already works.
