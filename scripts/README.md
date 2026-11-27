# scripts/

用户可用的辅助脚本，以及少量维护者发版工具。资产 vendor / relocalize 流水线属于内部维护材料，不随公开面发布；机械臂与末端组合 URDF 由 `ufactory.robots.assembly` 运行时生成，不再提供生成脚本。

## 用户脚本

| 脚本 | 模块 | 用途 |
|------|------|------|
| `gen_kinematics_params.py` | `ufactory.kinematics.controller_fk` | 从控制器 FK 辨识逐台运动学 YAML（严格 v0.2.5 校准 schema）；只读查询，用 100 组未参与辨识的位姿校验，超过 1 mm / 0.5° 不写文件 |
| `ft_sensor_check.py` | `ufactory.hardware.ft_sensor` | 六维力矩传感器真机诊断与验证（`enable` / `disable` 开关采样，`status` 只读查配置/实时力，`zero` 静止置零，`identify-load` / `admittance-test` / `force-test` 需 `--confirm-real`） |

真机 hold 电流/力矩时间序列观察由 `observe-hold-current` 控制台入口（`ufactory.hardware.observe:cli_observe_hold_current`）提供，安装后可直接调用。

## 维护者脚本

| 脚本 | 模块 | 用途 |
|------|------|------|
| `export_release_evidence_summary.py` | `ufactory.quality.evidence_summary` | 从本地 `project-check` JSON 报告生成脱敏 Release 附件摘要 |
