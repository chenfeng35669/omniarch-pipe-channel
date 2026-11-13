# OmniArch 圆管与平行板槽道速度预测

给定 128 个空间节点上的 **10 帧轴向速度**，连续预测后续 **20 帧**。实验采用恒定压差驱动的层流，以解析解计算误差，用同网格 Crank–Nicolson 有限差分作对照。

**A** 用三帧速度变化估计下一步，再用分步长更新和 OmniArch 特征纠偏。**E** 在 A 的时间步或扰动形状覆盖不足时，调用用扩散方程参考值训练的物理更新层。触发后由新层独立推进，其他轨迹沿用 A。

### 主要结果

原测试集中，在前六阶扰动上加入训练时未见的第 7、8 阶。每种几何 48 条轨迹，下表为三个适配器种子的平均**扰动相对误差**。

| 几何 | A | E（加入物理后训练） | 有限差分 |
|---|---:|---:|---:|
| 圆管 | 0.0616% | 0.003098% | 0.003665% |
| 平行板槽道 | 1.541% | 0.004203% | 0.003218% |

这批轨迹全部触发新层。物理层训练时用 24 个差分子步生成每个单步标签，推理时使用已拟合的更新矩阵。结果比较预测精度。

### 快速复现

在仓库根目录运行。已验证环境为 Python 3.10、PyTorch 2.6.0 和 NVIDIA CUDA。以下为 CUDA 11.8 的安装示例。

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python benchmark.py validate
```

从上游下载 **OmniArch Base v0.4、72,000 步**权重（约 1.23 GB），放到 `weights/omniarch-base.pt`，也可用环境变量 `OMNIARCH_CKPT` 指向已有文件。复现入口会核对权重 SHA-256。[Base 权重下载](https://bhpan.buaa.edu.cn/link/AA4C3D921A4EC6439B952F161E819F3938) · [上游代码与模型说明](https://github.com/Tarpelite/omniarch_pt/tree/5fba14f1ce93ec26d5c2b75b74ef17bf6c60dc02)

```bash
python specialization_20261107/specialize.py train-plain
python reproduce.py
```

第一条命令生成解析训练轨迹并训练圆管、槽道各三个适配器。第二条首次运行时重建 A 的纠偏权重，分别输出 A、E、有限差分的整场误差、扰动误差及新层触发数。只看槽道可运行 `python reproduce.py --geometry channel`。运行 `python run_ae.py` 可在两种几何的 A 权重都生成后复核其余七类工况。

### 代码入口

- [benchmark.py](benchmark.py) — 解析速度、稳态求解、有限差分和通用模型基线。
- [specialize.py](specialization_20261107/specialize.py) — 轨迹生成、普通微调与物理特化。
- [base_pipeline.py](base_pipeline.py) — A 的稳态处理、分步长纠偏和预测。
- [integrated.py](integrated.py) — E 的物理后训练及按轨迹选择更新方式的入口。
- [reproduce.py](reproduce.py) — 原 48 条轨迹的 A/E 对照与结果表。
- [ablation.py](ablation.py) — 训练适配器后运行 `python ablation.py`，对照完整 A、等量晚期样本 B、共享时间公式 C、关闭网络残差 D。

整场速度误差与扰动误差均以解析解为参照，后者用扣除稳态后的真实扰动作为分母。原始基线可分别用 `python benchmark.py baseline`（有限差分与复制末帧）和 `python benchmark.py model --device cuda`（通用 Base）运行。

仓库内使用的上游推理源码及其 [MIT 许可证](vendor/omniarch_pt/LICENSE) 位于 `vendor/omniarch_pt/`。训练数据可以由脚本生成。
