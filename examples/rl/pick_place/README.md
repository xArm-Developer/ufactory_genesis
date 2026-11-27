# 2. Pick & Place: tabletop pick and place

An xArm6 with Gripper G2 grasps a randomly spawned cube in simulation, transfers it to a fixed target, and sets it down. The policy is trained from scratch with PPO.

[中文说明](README_zh.md) · [RL examples](../README.md) · [Previous: Reach](../reach/README.md)

## Setup

Run the following commands from the repository root. Requires Linux and an NVIDIA GPU; interactive inference also requires a desktop display.

```bash
pip install -e ".[sim,rl]"
```

## Task overview

- Each episode starts at end-effector TCP `(300, 0, 300) mm` with a `50 mm` gripper gap. The `30 mm` cube spawn is sampled uniformly in `x ∈ [300, 500] mm`, `y ∈ [-200, 200] mm`, and the target is at `(300, 300) mm`.
- Control period `20 ms`, up to `400` steps. The policy takes a `70`-dimensional observation and outputs `4`-dimensional increment actions for end-effector XYZ and gripper gap, with per-step maximum increments of `5 mm` and `3 mm`.
- Success requirements: pre-lift drag ≤10 mm and tilt ≤15°; lift the cube's lowest point to ≥65 mm above the table; landing error ≤20 mm and placement tilt ≤15°; detach both fingers at 0–3 mm table clearance with gripper gap ≥35 mm, ≤1 mm additional end-effector pressing after touchdown, and hold stably for 0.2 s. Any robot-table collision counts as failure.

## Training

Training outputs are saved under `outputs/rl/pick_place/`. Run all commands below from the repository root.

### One-command training and validation

```bash
python -m examples.rl.pick_place.run_seeds \
  --seeds 1 2 3 -B 2048 --max-iterations 2000
```

Trains three seeds sequentially, runs an independent 64-episode evaluation, and records a single-environment replay. Evaluates every 50 iterations (with dual-set holdout validation on seeds `17000` + `17500` totaling 192 episodes, requiring combined success rate ≥99.4%, max pressing ≤0.8 mm, and recent training stability) and updates `model_best.pt`; stops early after two consecutive passing evaluations. Outputs are saved under each `pick_place_e2e_s<seed>/` directory, with the overall summary in `e2e_validation/summary.json`. Exits with a non-zero code if validation fails.

### Single-seed training or resume

```bash
python -m examples.rl.pick_place.train \
  -e pick_place_e2e_s1 --seed 1 -B 2048 --max-iterations 2000
```

To resume training, specify the source checkpoint and a new experiment name (`-B` must match the original run):

```bash
python -m examples.rl.pick_place.train \
  --resume outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt \
  -e pick_place_e2e_s1_resume --seed 1 -B 2048 --max-iterations 200
```

`-B` is the number of parallel environments; reduce it if VRAM is insufficient. `--max-iterations` caps the training iterations; `--eval-start-iteration` defers periodic strict evaluations until that iteration; `--early-stop-evaluations 0` disables early stopping. Existing experiment directories are not overwritten; pass `-e` to use a new directory for standalone runs.

## Benchmark

Across a 10-seed `C10` benchmark (`seeds 1..10`, `-B 2048`, up to `1200` iterations, `v0.3.11`), **8/10** seeds passed strict blind evaluation (`seed = 18000 + s`, 64 episodes, ≥98% success rate, ≤1.0 mm max pressing) and single-environment `GLB` replay, achieving **`508/512 (99.22%)`** combined blind success across passing seeds with **`0.00`** IK rejects (four seeds at `64/64 = 100.0%`, four at `63/64 = 98.44%`).

The pretrained checkpoint hosted on Hugging Face (`pick_place/xarm6-pick-place/seed-7/model_best.pt`, early-stopped at `650` iterations) was verified on both Colab L4 and a local **NVIDIA GeForce RTX 4060 Ti (8 GB)**:

| Robot + Gripper | Obs / Act Dim | Checkpoint | Dual Holdout (`17000` / `17500`) | Blind Eval (`18007`, 64 eps) | Max Press (`18007`) | Mean / Median Target Err (`18007`) | GLB `B=1` Replay |
|---|---:|---|---|---|---|---|---|
| `xarm6` + `gripper_g2` | 70 / 4 | `seed-7/model_best.pt` | **100%** (64/64) / **100%** (128/128) | **100%** (64/64) | `0.477 mm` (≤1.0 mm) | `6.98` / `6.14 mm` (≤20 mm) | **PASS** |

## Evaluation and inference

If you do not need to train from scratch locally, download the pretrained checkpoint from Hugging Face ([`UFACTORY-Developer/ufactory_genesis_rl`](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl)) into `outputs/rl/pick_place/`:

```bash
# Install the Hugging Face CLI if not already available: pip install huggingface_hub
hf download UFACTORY-Developer/ufactory_genesis_rl --include "pick_place/**" --local-dir outputs/rl
```

The examples below use the downloaded checkpoint `outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt` (for a locally trained run, replace `--checkpoint` with `outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt` or pass `-e pick_place_e2e_s1`). Evaluate the checkpoint before running inference.

### Evaluation

The example below evaluates the hosted seed-7 checkpoint, requiring ≥98% success rate and ≤1 mm maximum pressing over 64 episodes:

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt \
  -B 32 --episodes 64 --seed 18007 \
  --min-success-rate 0.98 --max-press-mm 1.0 \
  --quiet-episodes --out outputs/rl/pick_place/xarm6-pick-place/seed-7/test.json
```

`--out` writes both a JSON summary and a per-episode CSV table; exits with code `1` if the thresholds are not met.

### Real-time inference and recording

```bash
# Open interactive viewer
python -m examples.rl.pick_place.visualize \
  --checkpoint outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt -B 1

# Record a single-episode headless video
python -m examples.rl.pick_place.visualize \
  --checkpoint outputs/rl/pick_place/xarm6-pick-place/seed-7/model_best.pt \
  -B 1 --episodes 1 --headless \
  --record outputs/rl/pick_place/xarm6-pick-place/seed-7/replay.mp4
```

`-e` specifies an experiment name under `outputs/rl/pick_place/` and loads `model_best.pt` from that directory by default; use `--checkpoint` to specify any checkpoint path explicitly. `--headless` runs without a desktop window, and `--record` specifies the MP4 output path. Keep `model_best.pt`, `config.yaml`, and `model_best.checkpoint_manifest.json` in the same directory.

When running `train`, `evaluate`, or `visualize` directly, use `--recipe` and `--runtime-config` to specify custom training recipes and scene configurations. Checkpoints verify the configuration on load; retrain after modifying the task configuration.
