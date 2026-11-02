# MotionTrend 中文说明

本仓库整理的是你确认的 **父模型30轮＋SRR20轮** 默认完整版本，仅包含这一版的权重与训练记录。三个创新模块分别为：

- `model/ebte.py`：Energy-Bounded Trend Encoding（EBTE）
- `model/ctr.py`：Context and Trend Retrieval（CTR，包含内容读取分支）
- `model/srr.py`：Spatial Relation Refinement（SRR，包含限幅出口）

主干及组装在 `model/backbone.py`；三个开关集中在 `config/switches.py`。工程按 `data/`、`model/`、`config/`、`results/` 组织，另有训练、工具、测试和第三方源码目录。

| 结果口径 | 正确数 | 精度 |
|---|---:|---:|
| 完整50轮运行内的最高完整模型，epoch30（从0计数） | 283/287 | 98.61% |
| 最后一轮，epoch49 | 282/287 | 98.26% |

参数量为 **408,682**。这是分阶段训练的单模型结果。父模型选择和完整模型选择都使用这套固定评测集，因此论文中需要如实注明选优口径。SRR实际更新过、出口非零；这一选中版本保住父模型原有的283条正确案例，没有额外修复原4条错误。

## 安装和复现

参考环境为Linux、Python3.10.8、Torch2.1.2+cu118、NumPy1.26.4、einops0.8.2及RTX4090D。原生PointNet++二进制和编译源码均已提供。

```bash
conda create -n motiontrend python=3.10.8 -y
conda activate motiontrend
python -m pip install -r requirements.txt
python -B reproduce.py --verify-only

CUDA_VISIBLE_DEVICES=0 python -B reproduce.py \
  --data /path/to/full567_npz --output /path/to/new_replay
```

数据本身不打包。`--data`需要原567个NPZ文件，包内有读取顺序和SHA清单。复现入口恢复局部54、上下文14对应的实际随机状态，逐窗口核对5356个概率及287条视频结果。重新预处理的数据或仅设置两个seed，不等价于这份固定输入。

## 从零训练和续跑

```bash
# 只验证输入与梯度，测试更新会完整恢复，不保留训练步数
CUDA_VISIBLE_DEVICES=0 python -B train.py \
  --data /path/to/full567_npz --output /path/to/new_preflight --preflight-only

# 正式50轮
CUDA_VISIBLE_DEVICES=0 python -B train.py \
  --data /path/to/full567_npz --output /path/to/new_run

# 同一目录续跑
CUDA_VISIBLE_DEVICES=0 python -B train.py \
  --data /path/to/full567_npz --output /path/to/new_run --resume
```

训练从保存的epoch−1随机初始化开始，不使用`best.pth`初始化。完成前30轮后，恢复本次训练选出的父模型参数，并保留第30轮结束时的随机状态；后20轮冻结父模型，只更新SRR。训练SRR系数1.0，评测0.5，并且系数在`tanh`之前。

准确复现入口锁定完整三模块。关闭开关后是新的模型结构，需要另行训练，不能直接沿用283的权重和精度。单独用模型API加载完整权重后，应调用`model.set_phase(30)`和`model.set_srr_scale(0.5)`，然后`model.eval()`。

详细内容见[英文首页](README.md)、[训练与评价口径](docs/PROTOCOL.md)、[安装依赖](docs/INSTALL.md)及[验证记录](docs/VALIDATION.md)。
