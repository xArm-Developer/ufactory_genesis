# 1. Reach: robot arm target reaching

A robot arm without an end-effector tool starts from its default joint pose and moves its end effector to a random target inside the target workspace. The policy is trained from scratch in simulation with PPO; success requires an end-effector-to-target distance strictly below **5 mm**. Five robots are supported: `xarm5`, `xarm6` (default), `xarm7`, `uf850` and `lite6`.

[中文说明](README_zh.md) · [RL examples](../README.md)

## Requirements

- Linux with an NVIDIA GPU; install with `pip install -e ".[sim,rl]"` (see the [examples overview](../../README.md)).
- Run the commands as modules from the repository root, for example `python -m examples.rl.reach.train`.
- Real-time GLB inference requires a desktop display.

## Task definition

1. **Episode**: 20 ms control period, 5 s maximum episode length. Each episode starts from the initial joint pose corresponding to end-effector flange TCP `(300, 0, 300) mm`, `RPY = (180°, 0°, 0°)`, with a target sampled uniformly from the target workspace. The episode ends with success once the distance is below 5 mm; reaching the time limit counts as failure.
2. **Observation** (`3 * dof + 6` dimensions, concatenated in order):
   - Joint position `joint_pos` (`dof`, rad)
   - Joint velocity `joint_vel` (`dof`, rad/s)
   - World-frame end-effector position `ee_pos` (3, m)
   - Target displacement relative to the end effector `target_pos - ee_pos` (3, m)
   - Joint position command error `position_command - joint_pos` (`dof`, rad)
3. **Action** (`dof` dimensions): normalized joint-position increments in `[-1, 1]`, mapped linearly to at most 0.01 rad per step (0.5 rad/s joint speed limit × 20 ms). The accumulated joint position command is clamped to the URDF joint limits.
4. **Reward**: distance reward `exp(-10 d)` (weight 1.0) and joint velocity penalty `-Σ q̇²` (weight 0.001), both scaled by the control period, plus a one-time success reward of 2.0.
5. **Training algorithm**: rsl-rl PPO with `[256, 256, 128]` ELU MLPs for both actor and critic and a Beta action distribution. Each iteration collects 24 steps per environment. Defaults: 300 iterations, seed 7, a checkpoint every 25 iterations, ending with `model_299.pt`.

## Robots and benchmark

| `--robot` | DOF (`dof`) | Obs / Action dim | Target workspace (m) |
|---|---|---|---|
| `xarm5` | 5 | 21 / 5 | x `0.15–0.55`, y `-0.30–0.30`, z `0.05–0.50` |
| `xarm6` (default) | 6 | 24 / 6 | x `0.15–0.55`, y `-0.30–0.30`, z `0.05–0.50` |
| `xarm7` | 7 | 27 / 7 | x `0.15–0.55`, y `-0.30–0.30`, z `0.05–0.50` |
| `uf850` | 6 | 24 / 6 | x `0.15–0.55`, y `-0.30–0.30`, z `0.05–0.50` |
| `lite6` | 6 | 24 / 6 | x `0.15–0.35`, y `-0.20–0.20`, z `0.05–0.35` |

Measured on an **NVIDIA GeForce RTX 4060 Ti (8 GB)** with the [one-command run](#option-1-one-command) (`-B 4096`, seed 7, 300 iterations):

| `--robot` | VRAM | 300-iteration training time (total / PPO iterations) | Selected checkpoint | Validation success rate (17000 / 17001) | Mean terminal distance (17000 / 17001) |
|---|---|---|---|---|---|
| `xarm5` | 1.82 GB | 192 s / 159 s | `model_299.pt` | 99.6% / 99.8% | 4.48 / 4.50 mm |
| `xarm6` | 2.16 GB | 257 s / 240 s | `model_299.pt` | 100% / 100% | 4.23 / 4.24 mm |
| `xarm7` | 2.60 GB | 272 s / 255 s | `model_299.pt` | 99.8% / 99.2% | 4.39 / 4.52 mm |
| `uf850` | 2.32 GB | 297 s / 280 s | `model_299.pt` | 99.6% / 99.6% | 4.26 / 4.34 mm |
| `lite6` | 1.94 GB | 200 s / 167 s | `model_299.pt` | 99.8% / 99.6% | 4.66 / 4.71 mm |

- VRAM is the training process peak; total time includes Genesis initialization and scene building.
- Reduce `-B` (number of parallel environments) if VRAM is insufficient. When resuming, `-B` must match the original run.

## Train, select and validate

The one-command run and the step-by-step run produce the same output layout; use either one. The examples below use `xarm6` and the output directory `outputs/rl/reach/xarm6-reach`; for another robot, change both `--robot` and the directory name.

### Option 1: one command

```bash
python -m examples.rl.reach.run_experiments \
  --robot xarm6 \
  --output outputs/rl/reach/xarm6-reach \
  --seeds 7 -B 4096 --max-iterations 300
```

The steps run in order and stop at the first failure:

1. **Smoke test**: train 2 iterations with 64 environments, resume for 1 iteration and evaluate 8 episodes.
2. **Training**: each seed writes to `<output>/seed-<seed>/`.
3. **Selection**: evaluate every checkpoint on 128 targets with seed 17002 and rank by success rate (descending), mean terminal distance (ascending) and mean success time (ascending). The best checkpoint is written to `<output>/seed-<seed>/checkpoint-selection.json`.
4. **Validation**: evaluate the selected checkpoint on 512 targets each with seeds 17000 and 17001. Each set requires ≥95% success and ≤10 mm mean terminal distance; the result is written to `<output>/seed-<seed>/acceptance-result.json`.

- The `--output` directory must not exist; if omitted, it defaults to `outputs/rl/reach/reach-<timestamp>`.
- For three training seeds, replace `--seeds 7` with `--seeds 1 7 17`.
- Logs, process records and GPU samples of every step are saved in `<output>/`; the overall status is in `experiment-status.json`.
- Evaluation targets are generated on the CPU from a fixed seed, independent of the number of environments and the GPU, so they are reproducible.

### Option 2: step by step

```bash
# 1. Train
python -m examples.rl.reach.train \
  --robot xarm6 \
  --log-dir outputs/rl/reach/xarm6-reach/seed-7 \
  --seed 7 -B 4096 --max-iterations 300

# 2. Select: seed 17002, 128 targets
python -m examples.rl.reach.run_experiments \
  --stage select --robot xarm6 --output outputs/rl/reach/xarm6-reach --seeds 7

# 3. Validate: seeds 17000 and 17001, 512 targets each
python -m examples.rl.reach.run_experiments \
  --stage accept --robot xarm6 --output outputs/rl/reach/xarm6-reach --seeds 7
```

- Selection and validation rules are the same as in option 1. `--robot` defaults to `xarm6` and must match the trained robot.
- Selection and validation look for checkpoints in `<output>/seed-<seed>/`, so name `--log-dir` accordingly.
- An existing training directory raises an error; choose another name, or pass `--overwrite` to delete it and retrain.
- Training curves: `tensorboard --logdir outputs/rl/reach/xarm6-reach`.

### Resume training (optional)

```bash
python -m examples.rl.reach.train \
  --resume outputs/rl/reach/xarm6-reach/seed-7/model_299.pt \
  --log-dir outputs/rl/reach/xarm6-reach/seed-7-resume \
  --seed 7 -B 4096 --max-iterations 100
```

The robot is detected from `config.yaml` in the checkpoint directory. `--max-iterations 100` adds 100 iterations, and checkpoint numbering continues from the original run (`model_300.pt` to `model_399.pt`); `-B` must match the original run.

## Evaluation and inference

If you do not need to train from scratch locally, download the pretrained checkpoints from Hugging Face ([`UFACTORY-Developer/ufactory_genesis_rl`](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl)) into `outputs/rl/reach/`:

```bash
# Install the Hugging Face CLI if not already available: pip install huggingface_hub
# Download pretrained checkpoints for all 5 robots (~15 MB)
hf download UFACTORY-Developer/ufactory_genesis_rl --include "reach/**" --local-dir outputs/rl

# Or download a single robot (xarm6 example)
hf download UFACTORY-Developer/ufactory_genesis_rl --include "reach/xarm6-reach/**" --local-dir outputs/rl
```

The selected checkpoint is not necessarily the last one. Check the selection result first:

```bash
# Example output: "selected_checkpoint": "model_299.pt",
grep selected_checkpoint outputs/rl/reach/xarm6-reach/seed-7/checkpoint-selection.json
```

The examples below use `outputs/rl/reach/xarm6-reach/seed-7/model_299.pt`; replace `--checkpoint` with the file name shown above.

### Evaluate a checkpoint

```bash
python -m examples.rl.reach.evaluate \
  --checkpoint outputs/rl/reach/xarm6-reach/seed-7/model_299.pt \
  -B 128 --episodes 512 --seed 17000 \
  --min-success-rate 0.95 --max-mean-distance-m 0.010 \
  --quiet-episodes --summary-json outputs/rl/reach/xarm6-reach/seed-7/evaluation-17000.json
```

- The robot is detected from `config.yaml` in the checkpoint directory.
- The exit code is 1 if `--min-success-rate` or `--max-mean-distance-m` is not met; an existing report file raises an error.
- For the second set, change `--seed` to `17001` and the report name to `evaluation-17001.json`.

### Real-time GLB inference

```bash
# Single arm
python -m examples.rl.reach.visualize \
  --checkpoint outputs/rl/reach/xarm6-reach/seed-7/model_299.pt

# Four arms
python -m examples.rl.reach.visualize \
  --checkpoint outputs/rl/reach/xarm6-reach/seed-7/model_299.pt -b 4
```

- Robots are rendered with GLB visual models. Each arm holds for 1 s after success before starting its next episode. Close the window or press Ctrl+C to exit.
- Keep the checkpoint, `config.yaml` and its `*.checkpoint_manifest.json` in the same directory (the training output directory already satisfies this).

## Configuration and files

| File | Description |
|---|---|
| `recipe.yaml` | Number of parallel environments, reward weights, PPO hyperparameters |
| `assets/configs/runtime/tasks/reach.yaml` | Shared task parameters: control period, episode length, success distance, action scaling, default target workspace |
| `assets/configs/runtime/tasks/robots/*_reach.yaml` | Per-robot bare-arm URDF and initial joint pose; `lite6` also sets a smaller target workspace |
| `env.py` | Training and evaluation environment: state initialization, action execution, rewards and episode metrics |
| `realtime_env.py`, `realtime_viewer.py` | Real-time inference: reuses the training environment with its own scene layout, success hold and real-time stepping |
| `train.py`, `evaluate.py`, `visualize.py`, `run_experiments.py` | Command-line entry points |

- Use `--recipe` and `--runtime-config` to specify custom files.
- Checkpoint loading verifies the task configuration and action contract; retrain after changing the task configuration.
- Training logs, checkpoint manifests and cache setup share the `ufactory/training` implementation with Pick & Place.
