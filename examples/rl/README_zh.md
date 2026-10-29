# 强化学习示例

按以下顺序从易到难运行示例。所有流程均在仿真中执行，请在项目根目录通过模块方式运行指令。

仓库内不直接提交策略权重文件。Reach 预训练检查点托管于 Hugging Face（[`UFACTORY-Developer/ufactory_genesis_rl`](https://huggingface.co/UFACTORY-Developer/ufactory_genesis_rl)）；Pick & Place 需在本地训练。训练产物保存在 Git 忽略的 `outputs/rl/` 下。

1. **[Reach：目标到达](reach/README_zh.md)**：不装夹爪，让机械臂（`xarm5`、`xarm6`、`xarm7`、`uf850`、`lite6`）末端到达目标点。成功距离严格小于 **5 mm**，默认训练种子 7。指南提供预训练检查点下载、训练、筛选、验收和 GLB 实时推理指令。
2. **[Pick & Place：桌面抓放](pick_place/README_zh.md)**：使用 xArm6 + Gripper G2 抓取、抬升并搬运方块，在桌面目标位置放下、松爪。方块起点在 `x ∈ [300, 500] mm`、`y ∈ [-200, 200] mm` 矩形区域内均匀随机采样，目标点固定在 `(300, 300) mm`。指南提供本地训练、评估和 GLB 实时推理指令。
3. **装箱：待实现**：将物体放入箱内，增加箱体几何与接触约束。RL 环境及训练、推理入口尚未实现。

现有[装箱展示](../packaging/README_zh.md)采用轨迹控制。

[English](README.md)
