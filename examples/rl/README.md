# Reinforcement-learning examples

Follow the examples in increasing order of difficulty. These workflows run in simulation; run their commands as modules from the repository root.

Policy weights are not committed to Git. Pretrained Reach checkpoints are hosted on Hugging Face ([`UFACTORY-Developer/ufactory_genesis_rl`](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl)); Pick & Place must be trained locally. Training artifacts are saved under `outputs/rl/` (ignored by Git).

1. **[Reach — target reaching](reach/README.md)**: move the bare-arm end effector (`xarm5`, `xarm6`, `xarm7`, `uf850`, `lite6`) to a target without a gripper. Success requires a distance strictly below **5 mm**; the default training seed is 7. The guide covers downloading pretrained checkpoints, training, checkpoint selection, acceptance and real-time GLB inference.
2. **[Pick & Place — tabletop](pick_place/README.md)**: use xArm6 + Gripper G2 to grasp, lift and transfer a cube, then set it down and release it at a tabletop target. The cube start position is sampled uniformly at random in `x ∈ [300, 500] mm`, `y ∈ [-200, 200] mm` with a fixed target at `(300, 300) mm`. The guide covers local training, evaluation and real-time GLB inference.
3. **Box packing — planned**: place an object inside a box, adding box geometry and contact constraints. The RL environment and training/inference entries remain to be implemented.

The existing [packaging showcase](../packaging/README.md) uses trajectory control.

[中文说明](README_zh.md)
