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

Trains three seeds sequentially, runs an independent 64-episode evaluation, and records a single-environment replay. Evaluates every 50 iterations and updates `model_best.pt`; stops early after two consecutive evaluations with ≥98% success rate and ≤1 mm maximum pressing. Outputs are saved under each `pick_place_e2e_s<seed>/` directory, with the overall summary in `e2e_validation/summary.json`. Exits with a non-zero code if validation fails.

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

`-B` is the number of parallel environments; reduce it if VRAM is insufficient. `--max-iterations` caps the training iterations; `--early-stop-evaluations 0` disables early stopping. Existing experiment directories are not overwritten; pass `-e` to use a new directory for standalone runs.

## Local checkpoint evaluation and inference

The commands below use `model_best.pt` produced by local training. Evaluate the checkpoint before running inference.

### Evaluation

The example below evaluates the best checkpoint of seed 1, requiring ≥98% success rate and ≤1 mm maximum pressing over 64 episodes:

```bash
python -m examples.rl.pick_place.evaluate \
  --checkpoint outputs/rl/pick_place/pick_place_e2e_s1/model_best.pt \
  -B 32 --episodes 64 --seed 18001 \
  --min-success-rate 0.98 --max-press-mm 1.0 \
  --quiet-episodes --out outputs/rl/pick_place/pick_place_e2e_s1/test.json
```

`--out` writes both a JSON summary and a per-episode CSV table; exits with code `1` if the thresholds are not met.

### Real-time inference and recording

```bash
# Open interactive viewer
python -m examples.rl.pick_place.visualize -e pick_place_e2e_s1 -B 1

# Record a single-episode headless video
python -m examples.rl.pick_place.visualize \
  -e pick_place_e2e_s1 -B 1 --episodes 1 --headless \
  --record outputs/rl/pick_place/pick_place_e2e_s1/replay.mp4
```

`-e` specifies the experiment name and loads `model_best.pt` from that directory by default; use `--checkpoint` to specify a file explicitly. `--headless` runs without a desktop window, and `--record` specifies the MP4 output path.

When running `train`, `evaluate`, or `visualize` directly, use `--recipe` and `--runtime-config` to specify custom training recipes and scene configurations. Checkpoints verify the configuration on load; retrain after modifying the task configuration.
