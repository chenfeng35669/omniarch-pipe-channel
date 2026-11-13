# OmniArch 圆管与平行板槽道速度预测

这个仓库研究恒定压差驱动的两种层流：圆管中的径向速度剖面，以及两块平行板之间的横向速度剖面。预测量都是**轴向速度**。给模型 128 个节点上的 **10 帧观测**，让它用自己的预测连续推进 **20 帧**，再逐帧与解析解比较。

### 从 A 到 E

两种流动的稳态速度都是抛物线，叠加的小扰动会随时间衰减。直接预测完整剖面时，即使曲线看起来很接近，模型也可能算错相邻两帧之间的小变化，并在滚动预测中累积误差。

- **A · 逐步纠偏。** 用最后三帧估计速度变化的衰减，再按时间步分别学习各节点的速度更新。OmniArch 从最近十帧扣除稳态后的历史中提取特征，修正剩余偏差。每步检查扰动能量，限制本应衰减的扰动异常增大。
- **E · 增加物理后训练。** 冻结 A，在每种几何上用 256 个局部扰动、16 个时间步生成 4,096 对单步样本，训练额外的物理更新层。若步长超出 A 的训练区间 0.001–0.025，或初始扰动明显偏离 A 学过的前六阶空间形状，就由新层独立推进；其他轨迹沿用 A。

### 主要结果

复现入口使用原来暴露外推误差的那批测试轨迹：保持已训练的步长，在前六阶扰动上叠加第 7、8 阶更细的空间变化。圆管和槽道各 48 条，A、E 与同网格有限差分从相同观测出发。

| 几何 | 20 步评价指标 | A | E | 有限差分 |
|---|---|---:|---:|---:|
| 圆管 | 整场误差 | 0.005385% | 0.000351% | 0.000296% |
| 圆管 | 扰动误差 | 0.0616% | 0.003098% | 0.003665% |
| 平行板槽道 | 整场误差 | 0.3386% | 0.001021% | 0.000762% |
| 平行板槽道 | 扰动误差 | 1.541% | 0.004203% | 0.003218% |

两个指标使用相同的预测偏差，**整场误差**除以真实速度的范数，**扰动误差**除以扣除稳态后的真实扰动范数。先对每条轨迹的 20 帧计算相对误差，再平均轨迹和三个适配器种子。

这批轨迹全部触发物理层，槽道 48 条轨迹的扰动误差均下降，均值从 **1.541% 降至 0.004203%**，仍略高于有限差分的 **0.003218%**。原四阶和已覆盖的六阶测试则沿用 A。物理层用 24 个差分子步生成训练标签，推理时只应用拟合好的矩阵。其步长训练区间为 0.00055–0.034，更远范围的精度尚待检验。

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

四组消融中，两种几何在未训练的中间步长上改用共享时间公式后，整场误差都增大了两个数量级以上。关闭 OmniArch 修正后，圆管的增幅约为 6%。这组对照可用来区分分步长更新与网络修正各自的贡献。

原始基线可分别用 `python benchmark.py baseline`（有限差分与复制末帧）和 `python benchmark.py model --device cuda`（通用 Base）运行。

仓库内使用的上游推理源码及其 [MIT 许可证](vendor/omniarch_pt/LICENSE) 位于 `vendor/omniarch_pt/`。训练数据可以由脚本生成。
