# Security policy

## Supported version

Only the current `0.3.6` development line receives security fixes. This is Alpha robotics software and is not safety-certified.

## Reporting

Do not disclose a suspected vulnerability in a public issue when it could enable unintended robot motion, code execution, calibration/manifest bypass, or credential exposure. Use GitHub's private security advisory workflow for this repository and include a minimal reproduction, affected revision and impact. Do not include real robot IP addresses, serial numbers or proprietary calibration data.

## Trust boundaries

- Treat YAML, URDF, external checkpoints, SDK feedback and inventory commands as untrusted.
- Public RL entries require a hashed config and checkpoint manifest, validate both before deserialization, and use PyTorch's weights-only loader. Do not bypass these checks for downloaded `.pt` files.
- The public tree does not ship a legacy pickle migration CLI. If a private checkout still contains one, use it only when `--trusted-input` truthfully applies.
- Never weaken `ApprovedProgram` binding, collision checks, complete-serial calibration binding, or explicit real-motion confirmation.
- Repository assets and safety exemptions are integrity-checked; a mismatch fails closed.
- Lite6 excludes only its two finger-root collision boxes from each other in Genesis and Coal. Finger pads and external collisions retain the normal checks and 5 mm margin; no link-wide Lite6 exemption applies.
- RL checkpoints are simulation-only artifacts stored under `outputs/rl/` (trained locally or downloaded from `UFACTORY-Developer/ufactory_genesis_rl`); their integrity, manifest, runtime configuration body, and task-specific contracts (`5 mm` `success_distance_m` for Reach; `g2_calibrated_v1` physics profile for Pick & Place) are verified before a weights-only load. Checkpoints trained under earlier contact physics or incompatible environment contracts fail that binding by design; never bypass the mismatch for evaluation or real use. The public package contains no online real-policy execution, random-action deployment, policy session, or SDK policy-action adapter. `ufactory.deploy` was removed in v0.2.7.

Operators remain responsible for physical guarding, payload/tool validation, emergency-stop access, speed limits and site-specific risk assessment.
